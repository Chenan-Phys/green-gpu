#include "green/gpu/thc_gpu_ops.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <cuComplex.h>
#include <stdexcept>
#include <string>
#include <algorithm>

namespace green::gpu {
  namespace {
    using colmatrix=Eigen::Matrix<std::complex<double>,Eigen::Dynamic,Eigen::Dynamic,Eigen::ColMajor>;
    void check(int status,const char* operation) {
      if(status) throw std::runtime_error(std::string("native THC CUDA ")+operation+" failed with code "+std::to_string(status));
    }
    struct buffer {
      void* pointer=nullptr; size_t capacity=0;
      ~buffer(){if(pointer) cudaFree(pointer);}
      void reserve(size_t size) {
        if(size<=capacity)return;
        if(pointer) {check(cudaFree(pointer),"free");pointer=nullptr;capacity=0;}
        size_t free,total;check(cudaMemGetInfo(&free,&total),"memory preflight");
        if(size+512ul*1024*1024>free) throw std::runtime_error("native THC CUDA tile needs more memory/headroom");
        check(cudaMalloc(&pointer,size),"allocation");capacity=size;
      }
      cuDoubleComplex* complex_data(){return static_cast<cuDoubleComplex*>(pointer);}
      void clear(){if(pointer) {check(cudaFree(pointer),"free");pointer=nullptr;capacity=0;}}
    };
    __global__ void multiply_elements(const cuDoubleComplex* a,const cuDoubleComplex* b,cuDoubleComplex* c,size_t count) {
      const size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
      if(i<count)c[i]=cuCmul(a[i],b[i]);
    }
  }
  struct thc_gpu_ops::implementation {
    cublasHandle_t blas=nullptr;cusolverDnHandle_t solver=nullptr;
    buffer a,b,c;bool low;
    size_t gemms=0,solves=0,peak=0;
    explicit implementation(bool low_memory):low(low_memory) {
      check(cudaSetDevice(0),"select device");
      check(cublasCreate(&blas),"create cuBLAS");check(cusolverDnCreate(&solver),"create cuSOLVER");
      // No TF32/FP16 or implicit tensor-core mode is enabled.
      check(cublasSetMathMode(blas,CUBLAS_DEFAULT_MATH),"double math mode");
    }
    ~implementation(){if(blas)cublasDestroy(blas);if(solver)cusolverDnDestroy(solver);}
    void prepare(const colmatrix& A,const colmatrix& B,size_t output_count) {
      a.reserve(A.size()*sizeof(cuDoubleComplex));b.reserve(B.size()*sizeof(cuDoubleComplex));c.reserve(output_count*sizeof(cuDoubleComplex));
      peak=std::max(peak,a.capacity+b.capacity+c.capacity);
      check(cudaMemcpy(a.pointer,A.data(),A.size()*sizeof(cuDoubleComplex),cudaMemcpyHostToDevice),"A upload");
      check(cudaMemcpy(b.pointer,B.data(),B.size()*sizeof(cuDoubleComplex),cudaMemcpyHostToDevice),"B upload");
    }
    matrix download(void* source,size_t rows,size_t cols) {
      colmatrix output(rows,cols);
      check(cudaMemcpy(output.data(),source,output.size()*sizeof(cuDoubleComplex),cudaMemcpyDeviceToHost),"result download");
      matrix result=output;
      if(low){a.clear();b.clear();c.clear();}
      return result;
    }
  };
  thc_gpu_ops::thc_gpu_ops(bool low):_impl(new implementation(low)){}
  thc_gpu_ops::~thc_gpu_ops()=default;
  thc_gpu_ops::matrix thc_gpu_ops::gemm(const matrix& A,const matrix& B) {
    if(A.cols()!=B.rows())throw std::runtime_error("native THC GEMM shape mismatch");
    colmatrix ca=A,cb=B;
    _impl->prepare(ca,cb,A.rows()*B.cols());
    const cuDoubleComplex alpha=make_cuDoubleComplex(1,0),beta=make_cuDoubleComplex(0,0);
    check(cublasZgemm(_impl->blas,CUBLAS_OP_N,CUBLAS_OP_N,A.rows(),B.cols(),A.cols(),&alpha,
                     _impl->a.complex_data(),A.rows(),_impl->b.complex_data(),B.rows(),&beta,_impl->c.complex_data(),A.rows()),"GEMM");
    ++_impl->gemms;
    return _impl->download(_impl->c.pointer,A.rows(),B.cols());
  }
  thc_gpu_ops::matrix thc_gpu_ops::hadamard(const matrix& A,const matrix& B) {
    if(A.rows()!=B.rows() || A.cols()!=B.cols())throw std::runtime_error("native THC Hadamard shape mismatch");
    colmatrix ca=A,cb=B;_impl->prepare(ca,cb,A.size());
    multiply_elements<<<(A.size()+255)/256,256>>>(_impl->a.complex_data(),_impl->b.complex_data(),_impl->c.complex_data(),A.size());
    check(cudaGetLastError(),"Hadamard launch");
    return _impl->download(_impl->c.pointer,A.rows(),A.cols());
  }
  thc_gpu_ops::matrix thc_gpu_ops::solve(const matrix& A,const matrix& B) {
    if(A.rows()!=A.cols() || A.rows()!=B.rows())throw std::runtime_error("native THC solve shape mismatch");
    colmatrix ca=A,cb=B;_impl->prepare(ca,cb,0);
    int work_count=0;
    check(cusolverDnZgetrf_bufferSize(_impl->solver,A.rows(),A.cols(),_impl->a.complex_data(),A.rows(),&work_count),"LU workspace");
    buffer work,piv,info;work.reserve(size_t(work_count)*sizeof(cuDoubleComplex));piv.reserve(A.rows()*sizeof(int));info.reserve(sizeof(int));
    _impl->peak=std::max(_impl->peak,_impl->a.capacity+_impl->b.capacity+_impl->c.capacity+work.capacity+piv.capacity+info.capacity);
    check(cusolverDnZgetrf(_impl->solver,A.rows(),A.cols(),_impl->a.complex_data(),A.rows(),work.complex_data(),static_cast<int*>(piv.pointer),static_cast<int*>(info.pointer)),"LU");
    int host_info=0;check(cudaMemcpy(&host_info,info.pointer,sizeof(int),cudaMemcpyDeviceToHost),"LU status");
    if(host_info)throw std::runtime_error("native THC CUDA singular/invalid screening LU");
    check(cusolverDnZgetrs(_impl->solver,CUBLAS_OP_N,A.rows(),B.cols(),_impl->a.complex_data(),A.rows(),static_cast<int*>(piv.pointer),_impl->b.complex_data(),B.rows(),static_cast<int*>(info.pointer)),"screening solve");
    check(cudaMemcpy(&host_info,info.pointer,sizeof(int),cudaMemcpyDeviceToHost),"solve status");
    if(host_info)throw std::runtime_error("native THC CUDA screening solve rejected arguments");
    ++_impl->solves;
    return _impl->download(_impl->b.pointer,B.rows(),B.cols());
  }
  size_t thc_gpu_ops::gemm_calls()const{return _impl->gemms;}
  size_t thc_gpu_ops::solve_calls()const{return _impl->solves;}
  size_t thc_gpu_ops::peak_workspace_bytes()const{return _impl->peak;}
}

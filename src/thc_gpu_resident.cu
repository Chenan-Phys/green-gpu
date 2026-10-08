#include "green/gpu/thc_gpu_resident.h"
#include <green/tensors/thc_gw_fft.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <cufft.h>
#include <cuComplex.h>
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace green::gpu {
namespace {
  using z=cuDoubleComplex;
  using colmatrix=Eigen::Matrix<std::complex<double>,Eigen::Dynamic,Eigen::Dynamic,Eigen::ColMajor>;
  void check(int status,const char* name){if(status)throw std::runtime_error(std::string("resident THC CUDA ")+name+" status "+std::to_string(status));}
  size_t product(size_t a,size_t b){if(b && a>std::numeric_limits<size_t>::max()/b)throw std::runtime_error("THC device size overflow");return a*b;}
  int dimension(size_t n){if(!n || n>size_t(std::numeric_limits<int>::max()))throw std::runtime_error("THC CUDA dimension outside supported range");return int(n);}
  struct block {void* pointer=nullptr;size_t bytes=0;};
  struct arena {
    size_t budget,total=0,peak=0;std::vector<block> idle;
    explicit arena(size_t b):budget(b){}
    ~arena(){trim();}
    void trim(){for(auto b:idle){cudaFree(b.pointer);total-=b.bytes;}idle.clear();}
    block acquire(size_t bytes) {
      auto best=idle.end();
      for(auto it=idle.begin();it!=idle.end();++it)if(it->bytes>=bytes && (best==idle.end() || it->bytes<best->bytes))best=it;
      if(best!=idle.end()){auto result=*best;idle.erase(best);return result;}
      if(bytes>budget || total>budget-bytes)trim();
      if(bytes>budget || total>budget-bytes)throw std::runtime_error("resident THC GPU workspace exceeds declared budget");
      size_t available,all;check(cudaMemGetInfo(&available,&all),"memory preflight");
      if(bytes>available || available-bytes<512ul*1024*1024)throw std::runtime_error("resident THC CUDA workspace/headroom exhausted");
      block result;result.bytes=bytes;check(cudaMalloc(&result.pointer,bytes),"allocation");total+=bytes;peak=std::max(peak,total);return result;
    }
    void release(block b){idle.push_back(b);}
  };
  __global__ void elements(z* c,const z* a,const z* b,size_t n) {
    size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i<n)c[i]=cuCmul(a[i],b[i]);
  }
  __global__ void diagonal_sum_kernel(z* out,const z* in,size_t r,size_t count,double scale) {
    size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=r)return;z sum=make_cuDoubleComplex(0,0);
    for(size_t k=0;k<count;++k)sum=cuCadd(sum,in[k*r*r+i*(r+1)]);
    out[i]=make_cuDoubleComplex(sum.x*scale,sum.y*scale);
  }
  __global__ void diagonal_kernel(z* out,const z* in,size_t r) {
    size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i<r)out[i*(r+1)]=in[i];
  }
  __global__ void identity_minus(z* out,const z* p,size_t n,size_t count) {
    size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=n*n*count)return;
    size_t e=i%(n*n);z v=p[i];out[i]=make_cuDoubleComplex((e/n==e%n?1.:0.)-v.x,-v.y);
  }
  __global__ void pointers(z** a,z** b,z* av,z* bv,size_t stride,size_t count) {
    size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i<count){a[i]=av+i*stride;b[i]=bv+i*stride;}
  }
  __global__ void residual_norm(const z* residual,const z* rhs,double* result,size_t entries) {
    __shared__ double numerator[256],denominator[256];
    double a=0,b=0;
    for(size_t e=threadIdx.x;e<entries;e+=blockDim.x){z x=residual[size_t(blockIdx.x)*entries+e],y=rhs[size_t(blockIdx.x)*entries+e];a+=x.x*x.x+x.y*x.y;b+=y.x*y.x+y.y*y.y;}
    numerator[threadIdx.x]=a;denominator[threadIdx.x]=b;__syncthreads();
    for(unsigned s=128;s;s/=2){if(threadIdx.x<s){numerator[threadIdx.x]+=numerator[threadIdx.x+s];denominator[threadIdx.x]+=denominator[threadIdx.x+s];}__syncthreads();}
    if(!threadIdx.x)result[blockIdx.x]=sqrt(numerator[0])/fmax(1.,sqrt(denominator[0]));
  }
  __global__ void direct_correlation(z* out,const z* a,const z* b,const long* pair,const size_t* transfer,
                                    size_t nk,size_t r,size_t first,size_t nq,bool sigma,bool transpose) {
    size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x,entries=r*r,count=sigma?nk:nq;
    if(i>=count*entries)return;size_t index=i/entries,e=i%entries,ae=transpose?(e%r)*r+e/r:e;z sum=make_cuDoubleComplex(0,0);
    for(size_t j=0;j<nk;++j){
      if(sigma){size_t q=transfer[index*nk+j];if(q<first || q>=first+nq)continue;sum=cuCadd(sum,cuCmul(a[j*entries+ae],b[(q-first)*entries+e]));}
      else {long k=pair[(first+index)*nk+j];if(k<0)continue;sum=cuCadd(sum,cuCmul(a[size_t(k)*entries+ae],b[j*entries+e]));}
    }
    out[i]=make_cuDoubleComplex(sum.x/nk,sum.y/nk);
  }
  __global__ void reorder(z* out,const z* in,const size_t* map,size_t nk,size_t r,bool transpose,bool unpack,double scale) {
    size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x,entries=r*r;if(i>=nk*entries)return;
    size_t k=i/entries,e=i%entries,source_e=transpose?(e%r)*r+e/r:e;
    z v=unpack?in[map[k]*entries+source_e]:in[k*entries+source_e];
    size_t target=unpack?i:map[k]*entries+e;out[target]=make_cuDoubleComplex(v.x*scale,v.y*scale);
  }
  __global__ void fourier_product(z* out,const z* a,const z* b,const size_t* negative,size_t nk,size_t entries) {
    size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=nk*entries)return;z v=cuCmul(a[i],b[negative[i/entries]*entries+i%entries]);
    out[i]=make_cuDoubleComplex(v.x/nk,v.y/nk);
  }
  __global__ void accumulate_time_kernel(z* out,const z* value,size_t count,size_t entries,size_t nt,size_t t,double scale) {
    size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=count*entries)return;
    size_t target=(i/entries*nt+t)*entries+i%entries;z v=value[i];out[target]=cuCadd(out[target],make_cuDoubleComplex(v.x*scale,v.y*scale));
  }
  __global__ void symmetrize_time_kernel(z* out,size_t count,size_t r,size_t nt,size_t t) {
    size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x,entries=r*r;if(i>=count*entries)return;
    size_t q=i/entries,e=i%entries,row=e%r,col=e/r;if(row>col)return;
    size_t base=(q*nt+t)*entries,other=col+row*r;z a=out[base+e],b=out[base+other];
    z v=make_cuDoubleComplex((a.x+b.x)*.5,(a.y-b.y)*.5),vh=cuConj(v);
    out[base+e]=v;out[base+other]=vh;size_t mirror=(q*nt+nt-t-1)*entries;
    out[mirror+e]=v;out[mirror+other]=vh;
  }
  __global__ void time_slice_kernel(z* out,const z* in,size_t count,size_t entries,size_t nt,size_t t) {
    size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i<count*entries)out[i]=in[(i/entries*nt+t)*entries+i%entries];
  }
  __global__ void mirror_time_kernel(z* out,size_t count,size_t entries,size_t nt,size_t t) {
    size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i<count*entries)
      out[(i/entries*nt+nt-t-1)*entries+i%entries]=out[(i/entries*nt+t)*entries+i%entries];
  }
  unsigned grid(size_t n){if((n+255)/256>size_t(std::numeric_limits<int>::max()))throw std::runtime_error("THC CUDA launch too large");return unsigned((n+255)/256);}
}
struct thc_gpu_resident::storage {
  std::shared_ptr<arena> pool;block memory;
  storage(std::shared_ptr<arena> p,size_t bytes):pool(std::move(p)),memory(pool->acquire(bytes)){}
  ~storage(){pool->release(memory);}
};
namespace {
  z* data(const thc_gpu_resident::field& f){return static_cast<z*>(f.owner->memory.pointer)+f.offset;}
  size_t entries(const thc_gpu_resident::field& f){return product(product(f.rows,f.cols),f.count);}
  void same(const thc_gpu_resident::field& a,const thc_gpu_resident::field& b){if(a.rows!=b.rows || a.cols!=b.cols || a.count!=b.count)throw std::runtime_error("THC CUDA field mismatch");}
}
thc_gpu_resident::field thc_gpu_resident::field::slice(size_t first,size_t number)const {
  if(!number || first>count || number>count-first)throw std::out_of_range("THC device field slice");
  field result=*this;result.offset+=product(first,product(rows,cols));result.count=number;return result;
}
thc_gpu_resident::field thc_gpu_resident::field::reshape(size_t r,size_t c,size_t batches)const {
  if(!r || !c || !batches || product(product(r,c),batches)!=entries(*this))throw std::runtime_error("THC device reshape mismatch");
  field result=*this;result.rows=r;result.cols=c;result.count=batches;return result;
}
struct thc_gpu_resident::implementation {
  std::shared_ptr<arena> pool;bool low,fft=false,auxiliary_gemm3m=false;cublasHandle_t blas=nullptr;cusolverDnHandle_t solver=nullptr;
  cufftHandle plan=0;std::shared_ptr<storage> fft_work;
  std::shared_ptr<storage> transfer,pairs,kmap,qmap,negative;
  size_t nk=0,nq=0,rank=0,gemms=0,solves=0,copies=0,bytes=0;
  implementation(bool l,size_t b,bool gemm3m):pool(std::make_shared<arena>(b)),low(l),auxiliary_gemm3m(gemm3m){
    try {
      check(cudaSetDevice(0),"device");
      if(auxiliary_gemm3m){
        cudaDeviceProp properties{};check(cudaGetDeviceProperties(&properties,0),"GEMM3M device capability");
        if(properties.major<5)throw std::runtime_error("thc_cuda_aux_gemm3m requires CUDA compute capability >= 5.0");
      }
      check(cublasCreate(&blas),"cuBLAS create");check(cusolverDnCreate(&solver),"cuSOLVER create");
      check(cublasSetMathMode(blas,CUBLAS_DEFAULT_MATH),"double math");
    } catch(...) {
      if(solver)cusolverDnDestroy(solver);if(blas)cublasDestroy(blas);throw;
    }
  }
  ~implementation(){cudaDeviceSynchronize();if(plan)cufftDestroy(plan);if(blas)cublasDestroy(blas);if(solver)cusolverDnDestroy(solver);}
  template<class T> std::shared_ptr<storage> metadata(const std::vector<T>& values){
    auto result=std::make_shared<storage>(pool,product(values.size(),sizeof(T)));
    check(cudaMemcpy(result->memory.pointer,values.data(),values.size()*sizeof(T),cudaMemcpyHostToDevice),"metadata upload");
    ++copies;bytes+=values.size()*sizeof(T);return result;
  }
};
thc_gpu_resident::thc_gpu_resident(bool low,size_t budget,bool auxiliary_gemm3m):_impl(new implementation(low,budget,auxiliary_gemm3m)){}
thc_gpu_resident::~thc_gpu_resident()=default;
thc_gpu_resident::field thc_gpu_resident::allocate(size_t r,size_t c,size_t count,bool clear){
  dimension(r);dimension(c);dimension(count);field f{std::make_shared<storage>(_impl->pool,product(product(product(r,c),count),sizeof(z))),r,c,count,0};
  if(clear)zero(f);return f;
}
thc_gpu_resident::field thc_gpu_resident::upload(const matrix& value){return upload(std::vector<matrix>{value});}
thc_gpu_resident::field thc_gpu_resident::upload(const std::vector<matrix>& values){
  if(values.empty())throw std::runtime_error("empty THC GPU upload");size_t r=values[0].rows(),c=values[0].cols();
  auto f=allocate(r,c,values.size());std::vector<std::complex<double>> packed(entries(f));
  for(size_t i=0;i<values.size();++i){if(values[i].rows()!=r || values[i].cols()!=c)throw std::runtime_error("ragged THC CUDA upload");Eigen::Map<colmatrix>(packed.data()+i*r*c,r,c)=values[i];}
  check(cudaMemcpy(data(f),packed.data(),packed.size()*sizeof(z),cudaMemcpyHostToDevice),"field upload");++_impl->copies;_impl->bytes+=packed.size()*sizeof(z);return f;
}
std::vector<thc_gpu_resident::matrix> thc_gpu_resident::download(const field& f){
  std::vector<std::complex<double>> packed(entries(f));check(cudaMemcpy(packed.data(),data(f),packed.size()*sizeof(z),cudaMemcpyDeviceToHost),"field download");
  ++_impl->copies;_impl->bytes+=packed.size()*sizeof(z);std::vector<matrix> result;result.reserve(f.count);
  for(size_t i=0;i<f.count;++i)result.emplace_back(Eigen::Map<const colmatrix>(packed.data()+i*f.rows*f.cols,f.rows,f.cols));return result;
}
thc_gpu_resident::field thc_gpu_resident::multiply(const field& a,const field& b,char ta,char tb,double scale){
  return multiply_impl(a,b,ta,tb,scale,false);
}
thc_gpu_resident::field thc_gpu_resident::multiply_impl(const field& a,const field& b,char ta,char tb,double scale,bool gemm3m){
  auto op=[](char c){
    if(c=='N')return CUBLAS_OP_N;if(c=='T')return CUBLAS_OP_T;if(c=='C')return CUBLAS_OP_C;
    throw std::runtime_error("THC CUDA invalid GEMM transpose");
  };
  const auto opa=op(ta),opb=op(tb);
  size_t ar=ta=='N'?a.rows:a.cols,ac=ta=='N'?a.cols:a.rows,br=tb=='N'?b.rows:b.cols,bc=tb=='N'?b.cols:b.rows,count=std::max(a.count,b.count);
  if(!a.owner || !b.owner || ac!=br || (a.count!=1 && a.count!=count) || (b.count!=1 && b.count!=count))throw std::runtime_error("THC batched GEMM mismatch");
  const int rows=dimension(ar),cols=dimension(bc),inner=dimension(ac),lda=dimension(a.rows),ldb=dimension(b.rows),batch=dimension(count);
  const size_t stride_a=a.count==1?0:product(a.rows,a.cols),stride_b=b.count==1?0:product(b.rows,b.cols),stride_c=product(ar,bc);
  auto out=allocate(ar,bc,count);z alpha=make_cuDoubleComplex(scale,0),beta=make_cuDoubleComplex(0,0);
  if(gemm3m){
    // cuBLAS has no double-complex GEMM3M batched entry point. Each call
    // receives one column-major matrix; zero offsets broadcast single fields.
    for(size_t i=0;i<count;++i)
      check(cublasZgemm3m(_impl->blas,opa,opb,rows,cols,inner,&alpha,data(a)+i*stride_a,lda,
                         data(b)+i*stride_b,ldb,&beta,data(out)+i*stride_c,rows),"auxiliary GEMM3M");
  }else{
    check(cublasZgemmStridedBatched(_impl->blas,opa,opb,rows,cols,inner,&alpha,data(a),lda,stride_a,
                                   data(b),ldb,stride_b,&beta,data(out),rows,stride_c,batch),"batched GEMM");
  }
  _impl->gemms+=count;return out;
}
thc_gpu_resident::field thc_gpu_resident::project(const field& x,const field& g){auto first=multiply(x,g);return multiply(first,x,'N','C');}
thc_gpu_resident::field thc_gpu_resident::backproject(const field& x,const field& g,double scale){auto first=multiply(x,g,'C','N');return multiply(first,x,'N','N',scale);}
thc_gpu_resident::field thc_gpu_resident::hadamard(const field& a,const field& b){same(a,b);auto out=allocate(a.rows,a.cols,a.count);elements<<<grid(entries(a)),256>>>(data(out),data(a),data(b),entries(a));check(cudaGetLastError(),"Hadamard");return out;}
thc_gpu_resident::field thc_gpu_resident::diagonal_sum(const field& a,double scale) {
  if(a.rows!=a.cols)throw std::runtime_error("THC density matrix shape");auto out=allocate(a.rows,1);
  diagonal_sum_kernel<<<grid(a.rows),256>>>(data(out),data(a),a.rows,a.count,scale);check(cudaGetLastError(),"density diagonal");return out;
}
thc_gpu_resident::field thc_gpu_resident::diagonal(const field& a) {
  if(a.cols!=1 || a.count!=1)throw std::runtime_error("THC potential shape");auto out=allocate(a.rows,a.rows,1,true);
  diagonal_kernel<<<grid(a.rows),256>>>(data(out),data(a),a.rows);check(cudaGetLastError(),"potential diagonal");return out;
}
void thc_gpu_resident::add(const field& out,const field& value,double scale){same(out,value);z alpha=make_cuDoubleComplex(scale,0);check(cublasZaxpy(_impl->blas,dimension(entries(out)),&alpha,data(value),1,data(out),1),"accumulate");}
void thc_gpu_resident::copy(const field& out,const field& value){same(out,value);check(cudaMemcpyAsync(data(out),data(value),entries(out)*sizeof(z),cudaMemcpyDeviceToDevice),"field copy");}
void thc_gpu_resident::zero(const field& out){check(cudaMemsetAsync(data(out),0,entries(out)*sizeof(z)),"field zero");}
thc_gpu_resident::field thc_gpu_resident::compress(const field& m,const field& response){
  if(response.rows!=response.cols || m.rows!=response.rows)throw std::runtime_error("THC compression dimensions");
  auto left=multiply_impl(m,response,'C','N',1.,_impl->auxiliary_gemm3m);
  return multiply_impl(left,m,'N','N',1.,_impl->auxiliary_gemm3m);
}
thc_gpu_resident::field thc_gpu_resident::expand(const field& m,const field& core){
  if(core.rows!=core.cols || m.cols!=core.rows)throw std::runtime_error("THC expansion dimensions");
  auto left=multiply_impl(m,core,'N','N',1.,_impl->auxiliary_gemm3m);
  return multiply_impl(left,m,'N','C',1.,_impl->auxiliary_gemm3m);
}
thc_gpu_resident::field thc_gpu_resident::screen_core(const field& polarization){
  return solve_response(polarization,polarization);
}
thc_gpu_resident::field thc_gpu_resident::screen(const field& m,const field& response,bool auxiliary){
  if(m.count!=1 || response.rows!=response.cols || m.rows!=response.rows)throw std::runtime_error("THC screening dimensions");
  if(auxiliary)return expand(m,screen_core(compress(m,response)));
  auto Z=multiply(m,m,'N','C'),original=multiply(Z,response),rhs=multiply(original,Z);
  return solve_response(std::move(original),std::move(rhs));
}
thc_gpu_resident::field thc_gpu_resident::solve_response(field original,field rhs){
  same(original,rhs);if(original.rows!=original.cols)throw std::runtime_error("THC dielectric dimensions");
  const size_t n=original.rows,nw=original.count;
  auto A=allocate(n,n,nw);identity_minus<<<grid(entries(A)),256>>>(data(A),data(original),n,nw);check(cudaGetLastError(),"dielectric");
  original={};
  auto lu=allocate(n,n,nw),solution=allocate(n,n,nw);copy(lu,A);copy(solution,rhs);
  auto info=std::make_shared<storage>(_impl->pool,2*nw*sizeof(int));
  check(cudaMemsetAsync(info->memory.pointer,0,2*nw*sizeof(int)),"solve status zero");
  auto piv=std::make_shared<storage>(_impl->pool,n*nw*sizeof(int));
  // Batch medium dielectric systems when enough frequencies amortize setup.
  // The workstation probe measures 64/150/192/256 with 107 frequencies;
  // keep cuSOLVER for large systems and small nontrivial batches.
  if(n<=32 || (n<=256 && nw>=32)){
    auto aa=std::make_shared<storage>(_impl->pool,nw*sizeof(z*)),bb=std::make_shared<storage>(_impl->pool,nw*sizeof(z*));
    pointers<<<grid(nw),256>>>(static_cast<z**>(aa->memory.pointer),static_cast<z**>(bb->memory.pointer),data(lu),data(solution),n*n,nw);
    check(cudaGetLastError(),"batched solve pointers");
    check(cublasZgetrfBatched(_impl->blas,dimension(n),static_cast<z**>(aa->memory.pointer),dimension(n),static_cast<int*>(piv->memory.pointer),static_cast<int*>(info->memory.pointer),dimension(nw)),"batched LU");
    int host_status=0;
    check(cublasZgetrsBatched(_impl->blas,CUBLAS_OP_N,dimension(n),dimension(n),static_cast<const z* const*>(aa->memory.pointer),dimension(n),static_cast<int*>(piv->memory.pointer),
                            static_cast<z**>(bb->memory.pointer),dimension(n),&host_status,dimension(nw)),"batched solve");
    if(host_status)throw std::runtime_error("THC batched screening argument error");
  }else{
    int work_count=0;check(cusolverDnZgetrf_bufferSize(_impl->solver,dimension(n),dimension(n),data(lu),dimension(n),&work_count),"LU workspace size");
    auto work=std::make_shared<storage>(_impl->pool,size_t(work_count)*sizeof(z));
    for(size_t w=0;w<nw;++w){
      check(cusolverDnZgetrf(_impl->solver,dimension(n),dimension(n),data(lu)+w*n*n,dimension(n),static_cast<z*>(work->memory.pointer),
                            static_cast<int*>(piv->memory.pointer)+w*n,static_cast<int*>(info->memory.pointer)+w),"LU");
      // getrs uses a separate status slot so it cannot erase a singular LU flag.
      check(cusolverDnZgetrs(_impl->solver,CUBLAS_OP_N,dimension(n),dimension(n),data(lu)+w*n*n,dimension(n),static_cast<int*>(piv->memory.pointer)+w*n,
                            data(solution)+w*n*n,dimension(n),static_cast<int*>(info->memory.pointer)+nw+w),"solve");
    }
  }
  std::vector<int> status(2*nw);check(cudaMemcpy(status.data(),info->memory.pointer,2*nw*sizeof(int),cudaMemcpyDeviceToHost),"LU status");++_impl->copies;_impl->bytes+=2*nw*sizeof(int);
  for(int value:status)if(value)throw std::runtime_error("THC screening singular/invalid LU");
  _impl->solves+=nw;lu={};
  auto residual=multiply(A,solution);add(residual,rhs,-1);
  auto norms=std::make_shared<storage>(_impl->pool,nw*sizeof(double));
  residual_norm<<<dimension(nw),256>>>(data(residual),data(rhs),static_cast<double*>(norms->memory.pointer),n*n);check(cudaGetLastError(),"screening residual");
  std::vector<double> errors(nw);check(cudaMemcpy(errors.data(),norms->memory.pointer,nw*sizeof(double),cudaMemcpyDeviceToHost),"residual status");++_impl->copies;_impl->bytes+=nw*sizeof(double);
  for(double e:errors)if(!std::isfinite(e) || e>1e-9)throw std::runtime_error("native THC CUDA screening residual failed");
  return solution;
}
void thc_gpu_resident::configure_momentum(size_t nk,size_t nq,size_t r,const std::vector<size_t>& transfer,
                                         const std::vector<double>& k,const std::vector<double>& q,bool fft){
  if(transfer.size()!=nk*nk)throw std::runtime_error("THC CUDA transfer shape");_impl->nk=nk;_impl->nq=nq;_impl->rank=r;_impl->fft=fft;
  if(_impl->plan){synchronize();check(cufftDestroy(_impl->plan),"old FFT plan");_impl->plan=0;_impl->fft_work.reset();}
  std::vector<long> pairs(nq*nk,-1);for(size_t i=0;i<nk;++i)for(size_t j=0;j<nk;++j){size_t iq=transfer[j*nk+i];if(iq>=nq || pairs[iq*nk+j]!=-1)throw std::runtime_error("THC GPU nonunique transfer");pairs[iq*nk+j]=long(i);}
  _impl->transfer=_impl->metadata(transfer);_impl->pairs=_impl->metadata(pairs);
  if(fft){
    tensors::thc_momentum_fft mesh(k,q,nk);if(nq!=nk)throw std::runtime_error("THC cuFFT needs closed full q mesh");
    _impl->kmap=_impl->metadata(mesh.k_map());_impl->qmap=_impl->metadata(mesh.q_map());_impl->negative=_impl->metadata(mesh.negative_map());
    if(nk==1){_impl->fft=false;return;} // A length-one Fourier transform is identity.
    auto shape=mesh.shape();int dims[3]={dimension(shape[0]),dimension(shape[1]),dimension(shape[2])};
    check(cufftCreate(&_impl->plan),"cuFFT create");check(cufftSetAutoAllocation(_impl->plan,0),"cuFFT explicit workspace");
    size_t work=0;check(cufftMakePlanMany(_impl->plan,3,dims,dims,dimension(r*r),1,dims,dimension(r*r),1,CUFFT_Z2Z,dimension(r*r),&work),"cuFFT batch plan");
    if(work){_impl->fft_work=std::make_shared<storage>(_impl->pool,work);check(cufftSetWorkArea(_impl->plan,_impl->fft_work->memory.pointer),"cuFFT workspace");}
  }
}
thc_gpu_resident::field thc_gpu_resident::correlate(const field& a,const field& b,bool sigma,bool transpose,size_t first,size_t count){
  size_t nk=_impl->nk,r=_impl->rank;if(!count)count=_impl->nq;
  if(a.rows!=r || a.cols!=r || a.count!=nk || b.rows!=r || b.cols!=r || b.count!=(sigma?count:nk) || first>_impl->nq || count>_impl->nq-first)throw std::runtime_error("THC CUDA momentum field shape");
  if(!_impl->fft){auto result=allocate(r,r,sigma?nk:count);direct_correlation<<<grid(entries(result)),256>>>(data(result),data(a),data(b),static_cast<long*>(_impl->pairs->memory.pointer),
                     static_cast<size_t*>(_impl->transfer->memory.pointer),nk,r,first,count,sigma,transpose);check(cudaGetLastError(),"direct momentum");return result;}
  if(first || count!=nk)throw std::runtime_error("THC cuFFT requires all q");
  auto left=allocate(r,r,nk),right=allocate(r,r,nk);
  reorder<<<grid(entries(left)),256>>>(data(left),data(a),static_cast<size_t*>(_impl->kmap->memory.pointer),nk,r,transpose,false,1);
  reorder<<<grid(entries(right)),256>>>(data(right),data(b),static_cast<size_t*>((sigma?_impl->qmap:_impl->kmap)->memory.pointer),nk,r,false,false,1);
  check(cudaGetLastError(),"FFT packing");check(cufftExecZ2Z(_impl->plan,data(left),data(left),CUFFT_FORWARD),"left FFT");check(cufftExecZ2Z(_impl->plan,data(right),data(right),CUFFT_FORWARD),"right FFT");
  auto product=allocate(r,r,nk);
  fourier_product<<<grid(entries(product)),256>>>(data(product),data(left),data(right),static_cast<size_t*>(_impl->negative->memory.pointer),nk,r*r);check(cudaGetLastError(),"Fourier correlation");
  check(cufftExecZ2Z(_impl->plan,data(product),data(product),CUFFT_INVERSE),"inverse FFT");
  auto result=allocate(r,r,nk);reorder<<<grid(entries(result)),256>>>(data(result),data(product),static_cast<size_t*>((sigma?_impl->kmap:_impl->qmap)->memory.pointer),nk,r,false,true,1./nk);check(cudaGetLastError(),"FFT unpacking");return result;
}
void thc_gpu_resident::accumulate_time(const field& out,const field& value,size_t t,size_t nt,double scale){
  if(out.rows!=value.rows || out.cols!=value.cols || out.count!=value.count*nt || t>=nt)throw std::runtime_error("THC time accumulation shape");
  accumulate_time_kernel<<<grid(entries(value)),256>>>(data(out),data(value),value.count,value.rows*value.cols,nt,t,scale);check(cudaGetLastError(),"time accumulation");
}
void thc_gpu_resident::symmetrize_time(const field& out,size_t t,size_t nt){
  if(out.rows!=out.cols || out.count%nt || t>=nt/2)throw std::runtime_error("THC time symmetrization shape");
  symmetrize_time_kernel<<<grid(out.count/nt*out.rows*out.cols),256>>>(data(out),out.count/nt,out.rows,nt,t);check(cudaGetLastError(),"time symmetry");
}
void thc_gpu_resident::symmetrize(const field& out){
  if(out.rows!=out.cols)throw std::runtime_error("THC symmetry shape");
  symmetrize_time_kernel<<<grid(entries(out)),256>>>(data(out),out.count,out.rows,1,0);check(cudaGetLastError(),"matrix symmetry");
}
void thc_gpu_resident::mirror_time(const field& out,size_t t,size_t nt){
  if(!nt || out.count%nt || t>=nt/2)throw std::runtime_error("THC time mirror shape");
  mirror_time_kernel<<<grid(out.count/nt*out.rows*out.cols),256>>>(data(out),out.count/nt,out.rows*out.cols,nt,t);check(cudaGetLastError(),"time mirror");
}
thc_gpu_resident::field thc_gpu_resident::time_slice(const field& in,size_t t,size_t nt){
  if(in.count%nt || t>=nt)throw std::runtime_error("THC time slice shape");auto out=allocate(in.rows,in.cols,in.count/nt);
  time_slice_kernel<<<grid(entries(out)),256>>>(data(out),data(in),out.count,out.rows*out.cols,nt,t);check(cudaGetLastError(),"time gather");return out;
}
void thc_gpu_resident::synchronize(){check(cudaDeviceSynchronize(),"stage completion");}
void thc_gpu_resident::finish_stage(){synchronize();if(_impl->low)_impl->pool->trim();}
size_t thc_gpu_resident::gemm_calls()const{return _impl->gemms;}
size_t thc_gpu_resident::solve_calls()const{return _impl->solves;}
size_t thc_gpu_resident::peak_bytes()const{return _impl->pool->peak;}
size_t thc_gpu_resident::host_transfer_calls()const{return _impl->copies;}
size_t thc_gpu_resident::host_transfer_bytes()const{return _impl->bytes;}
}

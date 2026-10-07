#ifndef GREEN_GPU_THC_GPU_OPS_H
#define GREEN_GPU_THC_GPU_OPS_H
#include <Eigen/Dense>
#include <complex>
#include <memory>

namespace green::gpu {
  /** Synchronous CUDA matrix tiles. Return only after D2H completion; callers
   * may then reuse their host factors. No asynchronous borrowed pair buffers.
   */
  class thc_gpu_ops {
  public:
    using matrix=Eigen::Matrix<std::complex<double>,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>;
    explicit thc_gpu_ops(bool low_device_memory);
    ~thc_gpu_ops();
    thc_gpu_ops(const thc_gpu_ops&)=delete;
    thc_gpu_ops& operator=(const thc_gpu_ops&)=delete;
    matrix gemm(const matrix& A,const matrix& B);
    matrix hadamard(const matrix& A,const matrix& B);
    matrix solve(const matrix& A,const matrix& B);
    matrix project(const matrix& X,const matrix& G) {return gemm(gemm(X,G),X.adjoint());}
    matrix backproject(const matrix& X,const matrix& G) {return gemm(gemm(X.adjoint(),G),X);}
    size_t gemm_calls() const;
    size_t solve_calls() const;
    size_t peak_workspace_bytes() const;
  private:
    struct implementation;
    std::unique_ptr<implementation> _impl;
  };
}
#endif

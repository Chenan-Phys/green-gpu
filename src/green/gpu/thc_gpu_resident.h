#ifndef GREEN_GPU_THC_GPU_RESIDENT_H
#define GREEN_GPU_THC_GPU_RESIDENT_H
#include <Eigen/Dense>
#include <complex>
#include <memory>
#include <vector>
#include <string>
#include <map>

namespace green::gpu {
  /** Stage-resident complex-double fields. Matrices are column-major on device;
   * batch index is outermost. Views retain ownership until queued work completes.
   * Low-memory mode trims idle allocations at stage boundaries, never per GEMM.
   */
  class thc_gpu_resident {
  public:
    using matrix=Eigen::Matrix<std::complex<double>,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>;
    struct storage;
    struct field {
      std::shared_ptr<storage> owner;
      size_t rows=0,cols=0,count=0,offset=0;
      field slice(size_t first,size_t number=1) const;
      field reshape(size_t r,size_t c,size_t batches=1) const;
    };
    // Gauss GEMM3M is an explicit, complex-double experiment limited to the
    // auxiliary compression/expansion helpers. All other GEMMs stay standard.
    thc_gpu_resident(bool low_memory,size_t budget,bool auxiliary_gemm3m=false,bool profile=false);
    ~thc_gpu_resident();
    thc_gpu_resident(const thc_gpu_resident&)=delete;
    thc_gpu_resident& operator=(const thc_gpu_resident&)=delete;
    field allocate(size_t rows,size_t cols,size_t count=1,bool zero=false);
    field upload(const matrix& value);
    field upload(const std::vector<matrix>& values);
    std::vector<matrix> download(const field& value);
    field multiply(const field& a,const field& b,char trans_a='N',char trans_b='N',double scale=1);
    field project(const field& x,const field& g);
    field project(const field& x,const field& g,const field& packed_adjoint);
    field backproject(const field& x,const field& g,double scale=1);
    field backproject(const field& x,const field& g,const field& packed_adjoint,double scale=1);
    field adjoint(const field& value);
    field hadamard(const field& a,const field& b);
    field diagonal_sum(const field& a,double scale);
    field diagonal(const field& a);
    void add(const field& destination,const field& source,double scale=1);
    void copy(const field& destination,const field& source);
    void zero(const field& destination);
    field compress(const field& m,const field& response);
    field compress(const field& m,const field& response,const field& packed_adjoint);
    field expand(const field& m,const field& core);
    field expand(const field& m,const field& core,const field& packed_adjoint);
    field screen_core(const field& polarization);
    field screen(const field& m,const field& response,bool auxiliary);
    void configure_momentum(size_t nk,size_t nq,size_t rank,const std::vector<size_t>& transfer,
                            const std::vector<double>& k,const std::vector<double>& q,bool fft);
    field correlate(const field& a,const field& b,bool right_is_q,bool transpose_left=false,
                    size_t first_q=0,size_t number_q=0);
    void accumulate_time(const field& destination,const field& source,size_t t,size_t nt,double scale);
    void symmetrize_time(const field& destination,size_t t,size_t nt);
    void symmetrize(const field& destination);
    void mirror_time(const field& destination,size_t t,size_t nt);
    field time_slice(const field& source,size_t t,size_t nt);
    field repeat(const field& source,size_t repetitions);
    field orbital_vertices(const field& x,const field& m,size_t q);
    field orbital_sigma(const field& vertices,const field& weighted,const field& g,size_t q,double scale=1);
    field prepare_sigma_right(const field& screened);
    field correlate_sigma_prepared(const field& projected,const field& prepared,size_t first_q=0,size_t number_q=0);
    void finish_stage();
    void synchronize();
    size_t gemm_calls()const;
    size_t solve_calls()const;
    size_t peak_bytes()const;
    size_t host_transfer_calls()const;
    size_t host_transfer_bytes()const;
    size_t fft_calls()const;
    std::map<std::string,double> component_seconds();
  private:
    field multiply_impl(const field& a,const field& b,char trans_a,char trans_b,double scale,bool gemm3m);
    field solve_response(field polarization,field rhs);
    struct implementation;
    std::unique_ptr<implementation> _impl;
  };
}
#endif

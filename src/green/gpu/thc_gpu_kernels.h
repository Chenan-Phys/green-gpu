#ifndef GREEN_GPU_THC_GPU_KERNELS_H
#define GREEN_GPU_THC_GPU_KERNELS_H

#include "gpu_kernel.h"
#include "thc_gpu_ops.h"
#include <green/grids/transformer_t.h>
#include <green/integrals/thc_factor_data.h>

namespace green::gpu {
  // Full-BZ, scalar, double-precision v1. Each node leader owns a CUDA
  // context. q/k tasks are distributed over nodes, never over Gaussian Q.
  inline integrals::thc_reader_options native_gpu_options(const params::params& p) {
    auto options=integrals::thc_options(p);
    options.preload_all_cores=!p["cuda_low_cpu_memory"].as<bool>();
    return options;
  }
  inline void report_thc_cuda(const thc_gpu_ops& ops) {
    std::cout<<"Native THC CUDA GEMMs="<<ops.gemm_calls()<<", LU solves="<<ops.solve_calls()
             <<", owned peak tile bytes="<<ops.peak_workspace_bytes()<<std::endl;
  }
  class thc_hf_gpu_kernel {
    size_t _ns,_nk; double _madelung;
    const ztensor<4>& _S;
    std::shared_ptr<integrals::thc_factor_data> _factors;
    std::unique_ptr<thc_gpu_ops> _ops;
  public:
    thc_hf_gpu_kernel(const params::params& p,size_t nao,size_t ns,size_t NQ,double madelung,
                      const symmetry::brillouin_zone_utils& bz,const ztensor<4>& S):
      _ns(ns),_nk(bz.nk()),_madelung(madelung),_S(S) {
      _factors=std::make_shared<integrals::thc_factor_data>(p["dfintegral_hf_file"],_nk,nao,NQ,native_gpu_options(p));
      if(_factors->set_kind()!="hf") throw std::runtime_error("native GPU HF requires HF core");
      if(!utils::context().node_rank) _ops=std::make_unique<thc_gpu_ops>(p["cuda_low_gpu_memory"].as<bool>());
    }
    ztensor<4> solve(const ztensor<4>& dm) {
      ztensor<4> result(_ns,_nk,dm.shape()[2],dm.shape()[3]); result.set_zero();
      auto& ctx=utils::context();
      if(!ctx.node_rank) {
        auto& ops=*_ops;
        const size_t r=_factors->rank();
        MatrixXcd m0=_factors->M(_factors->transfer(0,0));
        MatrixXcd hartree=ops.gemm(m0,m0.transpose());
        Eigen::VectorXcd density=Eigen::VectorXcd::Zero(r);
        for(size_t s=0;s<_ns;++s) for(size_t k=0;k<_nk;++k)
          density+=ops.project(_factors->X(k),matrix(dm(s,k))).diagonal()/double(_nk);
        MatrixXcd potential=ops.gemm(hartree,density);
        const double prefactor=_ns==2?1.0:0.5;
        for(size_t sk=ctx.internode_rank;sk<_ns*_nk;sk+=ctx.internode_size) {
          const size_t s=sk/_nk,k=sk%_nk;
          auto X=_factors->X(k);
          MatrixXcd diagonal=potential.col(0).asDiagonal();
          matrix(result(s,k))=ops.backproject(X,diagonal);
          for(size_t kp=0;kp<_nk;++kp) {
            MatrixXcd projected=ops.project(_factors->X(kp),matrix(dm(s,kp)));
            MatrixXcd m=_factors->M(_factors->transfer(k,kp));
            MatrixXcd Z=ops.gemm(m,m.adjoint());
            matrix(result(s,k))-=prefactor/double(_nk)*ops.backproject(X,ops.hadamard(projected,Z));
          }
          matrix(result(s,k))-=prefactor*_madelung*ops.gemm(ops.gemm(matrix(_S(s,k)),matrix(dm(s,k))),matrix(_S(s,k)));
        }
        if(!ctx.global_rank) report_thc_cuda(ops);
      }
      utils::allreduce(MPI_IN_PLACE,result.data(),result.size(),MPI_C_DOUBLE_COMPLEX,MPI_SUM,ctx.global);
      return result;
    }
  };
  class thc_gw_gpu_kernel {
    size_t _ns,_nk,_nt,_nw;
    const grids::transformer_t& _ft;
    std::shared_ptr<integrals::thc_factor_data> _factors;
    std::unique_ptr<thc_gpu_ops> _ops;
  public:
    using G_type=utils::shared_object<ztensor<5>>;
    thc_gw_gpu_kernel(const params::params& p,size_t nao,size_t ns,size_t NQ,
                      const grids::transformer_t& ft,const symmetry::brillouin_zone_utils& bz):
      _ns(ns),_nk(bz.nk()),_nt(ft.sd().repn_fermi().nts()),_nw(ft.sd().repn_bose().nw()),_ft(ft) {
      _factors=std::make_shared<integrals::thc_factor_data>(p["dfintegral_file"],_nk,nao,NQ,native_gpu_options(p));
      if(_factors->set_kind()!="correlation") throw std::runtime_error("native GPU GW requires correlation core");
      const size_t r=_factors->rank();
      if(_nt%2 || double(_nt+_nw)*r*r*16>double(p["thc_workspace_mb"].as<size_t>())*1024*1024)
        throw std::runtime_error("native THC GPU tau/frequency shape or workspace budget unsupported");
      if(!utils::context().node_rank) _ops=std::make_unique<thc_gpu_ops>(p["cuda_low_gpu_memory"].as<bool>());
    }
    void solve(G_type& g,G_type& sigma) {
      auto ctx=g.cntx();
      sigma.fence(); if(!ctx.node_rank)sigma.object().set_zero(); sigma.fence();
      const size_t r=_factors->rank();
      if(!ctx.node_rank) {
        auto& ops=*_ops;
        for(size_t q=ctx.internode_rank;q<_factors->nq();q+=ctx.internode_size) {
          ztensor<4> chi(_nt,1,r,r),wc_w(_nw,1,r,r); chi.set_zero();wc_w.set_zero();
          MatrixXcd m=_factors->M(q),Z=ops.gemm(m,m.adjoint());
          for(size_t t=0;t<_nt/2;++t) {
            MatrixXcd bubble=MatrixXcd::Zero(r,r);
            for(size_t i=0;i<_nk;++i) for(size_t j=0;j<_nk;++j) {
              if(_factors->transfer(j,i)!=q)continue;
              for(size_t s=0;s<_ns;++s) {
                MatrixXcd left=ops.project(_factors->X(i),matrix(g.object()(_nt-t-1,s,i)));
                MatrixXcd right=ops.project(_factors->X(j),matrix(g.object()(t,s,j)));
                bubble-=(_ns==2?1.0:2.0)/double(_nk)*ops.hadamard(left.transpose(),right);
              }
            }
            matrix(chi(t,0))=0.5*(bubble+bubble.adjoint()).eval();
            matrix(chi(_nt-t-1,0))=matrix(chi(t,0));
          }
          _ft.tau_f_to_w_b(chi,wc_w,0,_nw,true);
          MatrixXcd identity=MatrixXcd::Identity(r,r);
          for(size_t w=0;w<_nw;++w) {
            MatrixXcd response=matrix(wc_w(w,0)),zr=ops.gemm(Z,response);
            MatrixXcd A=identity-zr,rhs=ops.gemm(zr,Z),wc=ops.solve(A,rhs);
            const double residual=(ops.gemm(A,wc)-rhs).norm()/std::max(1.0,rhs.norm());
            if(!wc.allFinite() || residual>1e-9)throw std::runtime_error("native THC CUDA screening residual failed");
            matrix(wc_w(w,0))=wc;
          }
          _ft.w_b_to_tau_f(wc_w,chi,0,_nt,true);
          for(size_t k=0;k<_nk;++k)for(size_t kp=0;kp<_nk;++kp) {
            if(_factors->transfer(k,kp)!=q)continue;
            auto X=_factors->X(k),Xp=_factors->X(kp);
            for(size_t t=0;t<_nt;++t)for(size_t s=0;s<_ns;++s) {
              MatrixXcd projected=ops.project(Xp,matrix(g.object()(t,s,kp)));
              matrix(sigma.object()(t,s,k))-=ops.backproject(X,ops.hadamard(projected,matrix(chi(t,0))))/double(_nk);
            }
          }
        }
        utils::allreduce(MPI_IN_PLACE,sigma.object().data(),sigma.object().size(),MPI_C_DOUBLE_COMPLEX,MPI_SUM,ctx.internode_comm);
        if(!ctx.global_rank)report_thc_cuda(ops);
      }
      sigma.fence();
    }
  };
}
#endif

#ifndef GREEN_GPU_THC_GPU_KERNELS_H
#define GREEN_GPU_THC_GPU_KERNELS_H

#include "gpu_kernel.h"
#include "thc_gpu_resident.h"
#include <green/grids/transformer_t.h>
#include <green/integrals/thc_factor_data.h>
#include <green/tensors/thc_gw_fft.h>
#include <chrono>

namespace green::gpu {
  // Full-BZ, scalar, double-precision v1. Each node leader owns a CUDA
  // context. q/k tasks are distributed over nodes, never over Gaussian Q.
  inline integrals::thc_reader_options native_gpu_options(const params::params& p) {
    auto options=integrals::thc_options(p);
    options.preload_all_cores=!p["cuda_low_cpu_memory"].as<bool>();
    return options;
  }
  class thc_hf_gpu_kernel {
    size_t _ns,_nk; double _madelung;
    const ztensor<4>& _S;
    std::shared_ptr<integrals::thc_factor_data> _factors;
    std::unique_ptr<thc_gpu_resident> _ops;
  public:
    thc_hf_gpu_kernel(const params::params& p,size_t nao,size_t ns,size_t NQ,double madelung,
                      const symmetry::brillouin_zone_utils& bz,const ztensor<4>& S):
      _ns(ns),_nk(bz.nk()),_madelung(madelung),_S(S) {
      _factors=std::make_shared<integrals::thc_factor_data>(p["dfintegral_hf_file"],_nk,nao,NQ,native_gpu_options(p));
      if(_factors->set_kind()!="hf") throw std::runtime_error("native GPU HF requires HF core");
      if(double(_ns*_nk+8)*_factors->rank()*_factors->rank()*16>double(p["thc_workspace_mb"].as<size_t>())*1024*1024)
        throw std::runtime_error("native THC GPU HF matrix workspace exceeds declared budget");
      if(!utils::context().node_rank) _ops=std::make_unique<thc_gpu_resident>(p["cuda_low_gpu_memory"].as<bool>(),p["thc_workspace_mb"].as<size_t>()*1024*1024);
      if(!utils::context().global_rank)std::cout<<"Native THC CUDA double: host "<<(p["cuda_low_cpu_memory"].as<bool>()?"one q cached":"all cores preloaded")
        <<", device "<<(p["cuda_low_gpu_memory"].as<bool>()?"stage tiles trimmed":"stage tiles reused")<<std::endl;
    }
    ztensor<4> solve(const ztensor<4>& dm) {
      ztensor<4> result(_ns,_nk,dm.shape()[2],dm.shape()[3]); result.set_zero();
      auto& ctx=utils::context();
      if(!ctx.node_rank) {
        auto& ops=*_ops;
        const size_t r=_factors->rank(),n=_factors->nao();
        using field=thc_gpu_resident::field;
        std::vector<thc_gpu_resident::matrix> host_x,host_dm,host_S;
        for(size_t k=0;k<_nk;++k)host_x.emplace_back(_factors->X(k));
        for(size_t s=0;s<_ns;++s)for(size_t k=0;k<_nk;++k){host_dm.emplace_back(matrix(dm(s,k)));host_S.emplace_back(matrix(_S(s,k)));}
        auto x=ops.upload(host_x),d=ops.upload(host_dm),overlap=ops.upload(host_S);
        auto m0=ops.upload(thc_gpu_resident::matrix(_factors->M(_factors->transfer(0,0))));
        auto hartree=ops.multiply(m0,m0,'N','T');
        auto density=ops.allocate(r,1,1,true);
        std::vector<field> projected;
        for(size_t s=0;s<_ns;++s){
          projected.push_back(ops.project(x,d.slice(s*_nk,_nk)));
          auto sum=ops.diagonal_sum(projected.back(),1./_nk);ops.add(density,sum);
        }
        auto potential=ops.multiply(hartree,density);
        auto diagonal=ops.diagonal(potential);
        auto coulomb=ops.backproject(x,diagonal);
        auto device_result=ops.allocate(n,n,_ns*_nk,true);
        const double prefactor=_ns==2?1.0:0.5;
        for(size_t sk=ctx.internode_rank;sk<_ns*_nk;sk+=ctx.internode_size) {
          const size_t s=sk/_nk,k=sk%_nk;
          ops.copy(device_result.slice(sk),coulomb.slice(k));
          auto first=ops.multiply(overlap.slice(sk),d.slice(sk));
          auto correction=ops.multiply(first,overlap.slice(sk),'N','N',-prefactor*_madelung);
          ops.add(device_result.slice(sk),correction);
        }
        for(size_t q=0;q<_factors->nq();++q){
          auto m=ops.upload(thc_gpu_resident::matrix(_factors->M(q)));
          auto Z=ops.multiply(m,m,'N','C');
          for(size_t sk=ctx.internode_rank;sk<_ns*_nk;sk+=ctx.internode_size){
            size_t s=sk/_nk,k=sk%_nk;
            for(size_t kp=0;kp<_nk;++kp)if(_factors->transfer(k,kp)==q){
              auto product=ops.hadamard(projected[s].slice(kp),Z);
              auto exchange=ops.backproject(x.slice(k),product,-prefactor/double(_nk));
              ops.add(device_result.slice(sk),exchange);
            }
          }
          ops.finish_stage();
        }
        auto values=ops.download(device_result);
        for(size_t s=0;s<_ns;++s)for(size_t k=0;k<_nk;++k)matrix(result(s,k))=values[s*_nk+k];
        if(!ctx.global_rank)std::cout<<"Native THC resident CUDA HF GEMMs="<<ops.gemm_calls()<<", host transfer calls="<<ops.host_transfer_calls()
          <<", host transfer bytes="<<ops.host_transfer_bytes()<<", owned peak bytes="<<ops.peak_bytes()<<std::endl;
      }
      utils::allreduce(MPI_IN_PLACE,result.data(),result.size(),MPI_C_DOUBLE_COMPLEX,MPI_SUM,ctx.global);
      return result;
    }
  };
  class thc_gw_gpu_kernel {
    size_t _ns,_nk,_nt,_nw,_tile_q=0;
    bool _fft=false,_auxiliary=false;
    size_t _workspace_bytes=0;
    std::string _screening;
    const grids::transformer_t& _ft;
    std::shared_ptr<integrals::thc_factor_data> _factors;
    std::unique_ptr<thc_gpu_resident> _ops;
  public:
    using G_type=utils::shared_object<ztensor<5>>;
    thc_gw_gpu_kernel(const params::params& p,size_t nao,size_t ns,size_t NQ,
                      const grids::transformer_t& ft,const symmetry::brillouin_zone_utils& bz):
      _ns(ns),_nk(bz.nk()),_nt(ft.sd().repn_fermi().nts()),_nw(ft.sd().repn_bose().nw()),_ft(ft) {
      _factors=std::make_shared<integrals::thc_factor_data>(p["dfintegral_file"],_nk,nao,NQ,native_gpu_options(p));
      if(_factors->set_kind()!="correlation") throw std::runtime_error("native GPU GW requires correlation core");
      const size_t r=_factors->rank(),nq=_factors->nq();
      _fft=p["thc_gw_k_contraction"].as<std::string>()=="fft";
      _screening=p["thc_gw_screening"].as<std::string>();
      _auxiliary=tensors::thc_auxiliary_screening(_screening,r,NQ);
      _workspace_bytes=p["thc_workspace_mb"].as<size_t>()*1024*1024;
      if(_fft){tensors::thc_momentum_fft layout(*_factors);}
      const double physical=2.*_ns*_nk*_nt*nao*nao+double(_nk)*r*nao+2.*_nt*_nw+double(r)*NQ;
      const double projection=(10.*_nk+8)*r*r+2.*_nk*r*nao;
      const double screening=_auxiliary?(7.*_nw+2.*_nt)*NQ*NQ:(7.*_nw+_nt)*r*r;
      const double fixed=physical+std::max(projection,screening)+2.*_nk*r*r+_nk*_nk+nq*_nk;
      const double available=double(_workspace_bytes)/16-fixed;
      const double per_q=_auxiliary?double(_nt)*NQ*NQ+3.*r*r+3.*r*NQ:double(_nt)*r*r;
      if(_nt%2 || available<per_q)throw std::runtime_error("native THC GPU workspace budget cannot hold one q stage");
      _tile_q=std::min(nq,size_t(available/per_q));
      if(_fft && _tile_q<nq)throw std::runtime_error("THC FFT all-q workspace exceeds declared budget; use direct or increase budget");
      if(!utils::context().global_rank)std::cout<<"Native THC resident CUDA GW: "<<(_fft?"batched cuFFT":"direct device sums")
        <<", screening "<<(_auxiliary?"auxiliary":"point")<<" dimension "<<(_auxiliary?NQ:r)<<", q tile "<<_tile_q<<std::endl;
      if(!utils::context().node_rank) {
        _ops=std::make_unique<thc_gpu_resident>(p["cuda_low_gpu_memory"].as<bool>(),_workspace_bytes);
        std::vector<size_t> transfer(_nk*_nk);
        for(size_t k=0;k<_nk;++k)for(size_t kp=0;kp<_nk;++kp)transfer[k*_nk+kp]=_factors->transfer(k,kp);
        _ops->configure_momentum(_nk,nq,r,transfer,_factors->kmesh_scaled(),_factors->qmesh_scaled(),_fft);
      }
    }
    void solve(G_type& g,G_type& sigma) {
      auto ctx=g.cntx();
      sigma.fence(); if(!ctx.node_rank)sigma.object().set_zero(); sigma.fence();
      const size_t r=_factors->rank(),n=_factors->nao(),nq=_factors->nq();
      if(!ctx.node_rank && (!_fft || !ctx.internode_rank)) {
        auto& ops=*_ops;
        using field=thc_gpu_resident::field;
        std::vector<thc_gpu_resident::matrix> host_x;
        for(size_t k=0;k<_nk;++k)host_x.emplace_back(_factors->X(k));
        auto x=ops.upload(host_x);field green;
        {
          std::vector<thc_gpu_resident::matrix> host_g;host_g.reserve(_nt*_ns*_nk);
          for(size_t t=0;t<_nt;++t)for(size_t s=0;s<_ns;++s)for(size_t k=0;k<_nk;++k)host_g.emplace_back(matrix(g.object()(t,s,k)));
          green=ops.upload(host_g);
        }
        auto result=ops.allocate(n,n,_nt*_ns*_nk,true);
        // The original transform excludes the endpoint rows of fermionic tau.
        thc_gpu_resident::matrix forward=thc_gpu_resident::matrix::Zero(_nt,_nw);
        forward.block(1,0,_ft.Tnt_BF().cols(),_nw)=_ft.Tnt_BF().transpose();
        auto forward_device=ops.upload(forward);
        auto backward_device=ops.upload(thc_gpu_resident::matrix(_ft.Ttn_FB().transpose()));
        auto now=[](){return std::chrono::steady_clock::now();};
        auto elapsed=[&](auto begin){return std::chrono::duration<double>(now()-begin).count();};
        double bubble_seconds=0,screen_seconds=0,sigma_seconds=0;
        for(size_t first=0;first<nq;first+=_tile_q) {
          size_t count=std::min(_tile_q,nq-first);
          const size_t d=_auxiliary?_factors->naux():r;
          auto chi=ops.allocate(d,d,count*_nt,true);
          field cores;
          if(_auxiliary){
            std::vector<thc_gpu_resident::matrix> host_cores;
            for(size_t q=first;q<first+count;++q)host_cores.emplace_back(_factors->M(q));
            cores=ops.upload(host_cores);
          }
          auto begin=now();
          for(size_t t=0;t<_nt/2;++t) {
            field response;
            if(_auxiliary)response=ops.allocate(r,r,count,true);
            for(size_t s=0;s<_ns;++s) {
              auto left=ops.project(x,green.slice(((_nt-t-1)*_ns+s)*_nk,_nk));
              auto right=ops.project(x,green.slice((t*_ns+s)*_nk,_nk));
              auto bubble=ops.correlate(left,right,false,true,first,count);
              if(_auxiliary)ops.add(response,bubble,-(_ns==2?1.:2.));
              else ops.accumulate_time(chi,bubble,t,_nt,-(_ns==2?1.:2.));
            }
            if(_auxiliary){
              // M is time independent: compress once per mirrored half-tau pair.
              ops.symmetrize(response);
              auto projected_response=ops.compress(cores,response);
              ops.accumulate_time(chi,projected_response,t,_nt,1.);
              ops.mirror_time(chi,t,_nt);
            }else ops.symmetrize_time(chi,t,_nt);
          }
          ops.finish_stage();bubble_seconds+=elapsed(begin);begin=now();
          for(size_t iq=0;iq<count;++iq) {
            size_t q=first+iq;auto tau=chi.slice(iq*_nt,_nt);
            if(!_fft && q%ctx.internode_size!=size_t(ctx.internode_rank)){ops.zero(tau);continue;}
            auto frequency=ops.multiply(tau.reshape(d*d,_nt),forward_device).reshape(d,d,_nw);
            field wc;
            if(_auxiliary)wc=ops.screen_core(frequency);
            else {auto m=ops.upload(thc_gpu_resident::matrix(_factors->M(q)));wc=ops.screen(m,frequency,false);}
            auto back=ops.multiply(wc.reshape(d*d,_nw),backward_device).reshape(d,d,_nt);
            ops.copy(tau,back);
          }
          ops.finish_stage();screen_seconds+=elapsed(begin);begin=now();
          for(size_t t=0;t<_nt;++t) {
            auto core=ops.time_slice(chi,t,_nt);
            auto wc=_auxiliary?ops.expand(cores,core):core;
            for(size_t s=0;s<_ns;++s) {
              auto projected=ops.project(x,green.slice((t*_ns+s)*_nk,_nk));
              auto point_sigma=ops.correlate(projected,wc,true,false,first,count);
              auto orbital=ops.backproject(x,point_sigma,-1.);
              ops.add(result.slice((t*_ns+s)*_nk,_nk),orbital);
            }
          }
          ops.finish_stage();sigma_seconds+=elapsed(begin);
        }
        auto values=ops.download(result);
        for(size_t t=0;t<_nt;++t)for(size_t s=0;s<_ns;++s)for(size_t k=0;k<_nk;++k)
          matrix(sigma.object()(t,s,k))=values[(t*_ns+s)*_nk+k];
        if(!ctx.global_rank)std::cout<<"Native THC resident CUDA GEMMs="<<ops.gemm_calls()<<", LU solves="<<ops.solve_calls()
          <<", owned peak bytes="<<ops.peak_bytes()<<", host transfer calls="<<ops.host_transfer_calls()
          <<", host transfer bytes="<<ops.host_transfer_bytes()<<", bubble_seconds="<<bubble_seconds
          <<", screen_seconds="<<screen_seconds<<", sigma_seconds="<<sigma_seconds<<std::endl;
      }
      if(!ctx.node_rank)utils::allreduce(MPI_IN_PLACE,sigma.object().data(),sigma.object().size(),MPI_C_DOUBLE_COMPLEX,MPI_SUM,ctx.internode_comm);
      sigma.fence();
    }
  };
}
#endif

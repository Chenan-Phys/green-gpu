# GPU THC modes

Use the coordinated `feature/thc-contraction-order` revisions of green-symmetry,
green-gpu and green-mbpt. DF remains the default; select it explicitly with
`--interaction_representation df`. Select THC with
`--interaction_representation thc --thc_mode reconstruct|native`.

Reconstruction supplies synchronous original-Q host tiles to existing GPU
kernels. `cuda_low_cpu_memory` chooses a cached q core or all cores preloaded.
It never preloads a full reconstructed DF archive. Existing reconstruction
mixed precision and both device memory flags are supported independently.

Native scalar full-BZ double HF/GW keeps projections and intermediates on the
GPU. HF uploads X, density and overlap once per solve, reuses projected density,
uploads each q core once, and downloads the final self-energy. GW uploads X,
physical G and IR transform matrices once per solve. Batched cuBLAS projections,
device momentum contractions, tau/frequency transforms, screening and
backprojection share owned buffers; only the final Sigma and small solver status
arrays return to the host. Direct GW uses budgeted q tiles, with a one-q streaming
fallback. The projection cache is reused across every q in a tile.

`--thc_gw_screening auto|point|auxiliary` defaults to auto: solve in original
auxiliary space when Q < interpolation rank, otherwise point space. With
`Z=M M^H`, the auxiliary calculation is exactly
`P=M^H chi M; (I-P) C=P; Wc=M C M^H`. It introduces no fit or rank truncation.
Auxiliary mode compresses the summed-spin bubble in the mirrored half of tau,
then applies the original IR transforms to Q-by-Q matrices. It solves for C in
frequency and retains C in tau; Wc is expanded once per current tau and reused
across spins. M is independent of tau, so moving it across these linear
transforms preserves the same fitted interaction. The q-dependent M stays
outside momentum transforms. Point mode retains its original ordering.
Systems up to dimension 32 use batched cuBLAS LU. Dimensions up to 256 also
use batching when there are at least 32 frequencies, based on workstation
microbenchmarks. Other systems reuse one cuSOLVER workspace across frequencies.
Both paths check factorization status, solve status, and residuals.

`--thc_cuda_aux_gemm3m true` explicitly selects complex-double Gauss GEMM3M
for auxiliary compression and expansion. The default is false. It changes the
floating-point operation order and uses individual cuBLAS calls per batch;
performance depends on shape and GPU. HF, projections, IR transforms, point
screening, LU and residual checks retain the standard backend. Explicit enable
requires compute capability at least 5.0. Compare full GW time and numerical
results when choosing this backend.

`--thc_gw_k_contraction direct|fft` defaults to direct. FFT uses cached batched
cuFFT plans on complete Cartesian commensurate meshes. The shared mesh mapper
preserves shifted cosets and shuffled k/q ordering. Complex correlations use
the negative Fourier index without conjugating the second field. Length-one
momentum transforms reduce to identity. FFT retains all-q histories in the
selected screening space (Q-by-Q auxiliary cores or point-space matrices);
insufficient declared workspace rejects with a direct-mode alternative.

`thc_workspace_mb` caps owned device buffers, including explicit cuFFT workspace.
Allocation checks also retain 512 MiB free device headroom. CUDA contexts and
library-owned internal allocations are additional; process-tree GPU sampling
is reported separately. Low device memory trims idle buffers at stage boundaries;
whole device mode keeps them for reuse. Both modes reuse intermediates within
a stage. `thc_factor_memory_mb` separately bounds host X/core storage.

Native execution uses device 0 on each node leader. Direct q tasks are distributed
over node leaders; FFT executes on the first node leader and reduces Sigma.
Same-host MPI tests establish correctness, not multi-node/multi-GPU scaling.
Native single precision, GF2, IBZ/TR/spatial reduction, spinors, Cholesky screening,
nt_batch overrides and AqQ/extrapolation remain unsupported. GPU GF2 selection
with reconstruction executes GPU HF plus CPU GF2 correlation.

Printed counters report owned peak bytes, host transfer calls/bytes, GEMMs,
LU solves and synchronized GW stage times. Performance depends on rank, Q,
basis, mesh and requested accuracy; retain the DF choice when it is faster.


## Sigma contraction and execution controls

The `feature/thc-sigma-optimization` branch adds exact complex-double execution
alternatives for native scalar, full-BZ GW. DF remains the default interaction
representation. These controls do not change the fitted factors or fit tolerances.

- `--thc_gw_sigma point|orbital|auto` defaults to `point`. `orbital` requires
  auxiliary screening and direct momentum sums. It constructs fixed vertices
  `V[k,kp,A,a,i] = sum_p conj(X[k,p,a])*X[kp,p,i]*M[q,p,A]`, then contracts
  `Sigma[k] -= sum_(kp,A,B) V_A G[kp] V_B^H C[q,A,B]/Nk`. The screened core `C`
  stays in Q space; Sigma never expands it into a point matrix. The CPU and GPU
  layouts differ internally and are tested against independent point expressions.
  The cache scales as `Nk*Nq*n*n*Q` complex values and can be reused across solver
  iterations. Cold-run timings include its construction. `auto` uses a conservative
  cached-point operation estimate with a 10% margin, not a timing guarantee.
  In unsupported momentum/screening modes it selects `point`. GPU auto also
  selects point when vertex workspace does not fit or a tau batch is requested.
- `--thc_fft_reuse_screening true|false` defaults to `true`. In momentum FFT mode
  it transforms the screened interaction once per tau and shares it across spins.
  The original non-conjugating correlation, negative-frequency mapping and shifted
  mesh values are preserved. Two-spin Sigma uses five FFT calls per tau instead
  of six. The complete Nt=110 example uses 880 rather than 990 calls, including
  the unchanged bubble stage. A one-point mesh needs no transforms.
- `--thc_cpu_threads N` defaults to 1, allows 1..64, and bounds parallel Sigma
  tau workers by the declared workspace. It applies to auxiliary/direct CPU GW
  with retained owned-q histories. Streaming falls back to one worker. Each worker
  writes disjoint tau slices; q accumulation order is preserved. Use one BLAS
  thread when testing these workers to avoid nested CPU oversubscription.
- `--thc_cuda_sigma_batch N` defaults to 0. Values 1..32 batch point-space
  projection/backprojection over spins and up to N adjacent tau slices. A final
  partial tile is supported. Extra projected/point fields consume workspace;
  this option is incompatible with explicit orbital Sigma. Zero preserves the
  original per-spin point path. Benchmark the size for the intended dimensions.

MBPT uses standard C++ threads scoped to the THC Sigma kernel. It does not
activate legacy OpenMP tensor loops. Worker code uses borrowed Eigen maps of
buffers whose owners stay alive through every join; it avoids ndarray slice
creation because GREEN v1.0.0 storage reference counting is non-atomic.
Rebuild the coordinated
symmetry, GPU and MBPT revisions together after changing resident APIs.

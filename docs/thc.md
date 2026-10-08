# GPU THC modes

Use the coordinated `feature/thc-performance` revisions of green-symmetry,
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
Small systems use batched cuBLAS LU; larger systems reuse one cuSOLVER workspace
across frequencies. Both paths check solver status and residuals.

`--thc_gw_k_contraction direct|fft` defaults to direct. FFT uses cached batched
cuFFT plans on complete Cartesian commensurate meshes. The shared mesh mapper
preserves shifted cosets and shuffled k/q ordering. Complex correlations use
the negative Fourier index without conjugating the second field. Length-one
momentum transforms reduce to identity. FFT retains all-q response/Wc arrays;
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

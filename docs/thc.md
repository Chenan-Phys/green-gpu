# GPU THC modes

Use coordinated green-symmetry and green-mbpt THC feature revisions.
Reconstruction remains a synchronous host pair provider followed by existing
GPU kernels. It supports both `cuda_low_cpu_memory` modes: one cached q core
or all cores preloaded. It never preloads a full reconstructed DF archive.
Both device memory modes and existing reconstruction mixed precision are
covered independently.

Native full-BZ scalar double HF/GW dispatches cuBLAS complex-double GEMMs,
CUDA Hadamard tiles and cuSOLVER pivoted LU screening solves. Host/device
copies are synchronous before caller buffers can be reused. Low device memory
releases buffers after each operation; whole device mode reuses owned tile
capacities. Allocation checks retain 512 MiB free device headroom. Printed
peak tile bytes exclude CUDA contexts/library allocations; process-tree GPU
sampling is separate. Native execution uses device 0 on each node leader;
multi-GPU scheduling is not established by the one-GPU tests.

Native P_sp/Sigma_sp, Cholesky screening, nt_batch overrides, spinors, IBZ
reductions, native GF2 and AqQ/extrapolation are unsupported. The tau/frequency
workspace is one q at a time in direct mode with an explicit host budget. Interpolation I is
not Gaussian Q and does not enter the original-Q MPI schedule. GF2 with GPU
selection still executes GPU HF plus CPU GF2 correlation.

The native implementation prioritizes validated algebra and bounded ownership.
It transfers many small tiles. A small fixture timing is not a large-system
speedup or compression guarantee. Optional native GW
`--thc_gw_k_contraction fft` runs full-mesh complex momentum transforms on the
host while CUDA executes projection, screening and backprojection. It uses a
separate conservative all-q workspace check; all four host/device memory flag
combinations pass the complex fixture. Point-space symmetry, distributed/device
FFTs and multi-device performance require independent work.

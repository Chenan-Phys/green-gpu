[![GitHub license](https://img.shields.io/github/license/Green-Phys/green-mbpt?cacheSeconds=3600&color=informational&label=License)](./LICENSE)
[![GitHub license](https://img.shields.io/badge/C%2B%2B-17-blue)](https://en.cppreference.com/w/cpp/compiler_support/17)

![gpu](https://github.com/Green-Phys/green-gpu/actions/workflows/test.yaml/badge.svg)
[![codecov](https://codecov.io/gh/Green-Phys/green-gpu/graph/badge.svg?token=GZGIYJ52PW)](https://codecov.io/gh/Green-Phys/green-gpu)

# green-gpu
Implementation of HF/GW kernels for GPU

Set the following cmake variables for custom kernel extension in green-mpt project
   - GREEN_KERNEL_URL="https://github.com/Green-Phys/green-gpu" 
   - GREEN_CUSTOM_KERNEL_LIB="GREEN::GPU"
   - GREEN_CUSTOM_KERNEL_ENUM=GPU 
   - GREEN_CUSTOM_KERNEL_HEADER=\<green/gpu/gpu_factory.h\>

## Space-group DF archives on this feature branch

The HF/GW readers accept `green.df.space_group.v1` through the shared
`green-symmetry` host reconstruction implementation. Its default dependency is
`Chenan-Phys/green-symmetry` commit
`82b5c67a5c2f6d53dd0fbf80178eb1a234879041`, configurable with
`GREEN_SYMMETRY_GIT_REPOSITORY` and `GREEN_SYMMETRY_GIT_TAG`.
Use a coordinated SG-aware `green-mbpt` consumer and add
`--integral_symmetry space_group`. Keep both GPU memory flags explicit.
All four host/device memory combinations use representative storage;
whole-host preload checks `--integral_symmetry_preload_bytes` (default 1 GiB
per node) before allocation. Reconstruction owns its output and completes
in complex double before the existing precision conversion and CUDA kernels.
The dense metric/transform cache is bounded by
`--integral_symmetry_cache_bytes` (64 MiB per reader by default).

CPU/GPU HF and GW comparisons, both precision flags and two local MPI ranks
were exercised on one GPU. The existing code uses single precision when both
`P_sp` and `Sigma_sp` are true; other combinations retain existing behavior.
`GF2 --kernel GPU` dispatches GPU HF with CPU GF2 correlation in `green-mbpt`.
There is no GPU GF2 correlation kernel here. Special finite-size sidecars,
truncated metric sectors and unsupported schema/profile combinations are
rejected by the coordinated consumer. Multi-GPU/multi-node tests were not run.

# Acknowledgements

This work is supported by National Science Foundation under the award CSSI-2310582

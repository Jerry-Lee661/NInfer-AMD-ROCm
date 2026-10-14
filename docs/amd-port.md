# AMD gfx1100 port notes

How the CUDA engine was brought to RDNA 3 without touching CUDA sources.

## Architecture

- `compat/cuda/`: HIP shim headers, first on the include path. About 67 runtime symbols are mapped (`cudaGraph*` to `hipGraph*`, `cudaLaunchKernelEx` to `hipLaunchKernelGgl`, cooperative groups via `<hip/hip_cooperative_groups.h>`, bf16 via `hip_bfloat16.h` aliases). All 173 `#include <cuda_runtime.h>` sites compile unchanged.
- `src/ops/common/{memory,math,mma}.cuh`: the only places inline PTX survives. `cp_async` becomes a synchronous copy (zfill semantics kept); `mma_bf16/f16/s8` are emulated with `__shfl_sync` + FMA honoring the exact PTX fragment layouts. Later work replaced hot paths with real RDNA 3 WMMA (VOP3P `v_wmma_*` instructions; there is no HIP WMMA API) and `__builtin_amdgcn_sudot4` for int8 dot products.
- CMake: `NINFER_BACKEND=HIP` compiles `.cu` as HIP (`-x hip`); NVFP4/TMA targets are excluded from the HIP build.

## Platform facts (gfx1100, measured)

- LDS limit 64 KB per workgroup (sm_86: 163 KB): prompt kernels needed Br/Bc halving to fit.
- No `ldmatrix`, no `cp.async`.
- RDNA 3 WMMA consists of 6 VOP3P instructions (16x16x16) with replicated data in the A/B fragments; the layout contract had to be reverse-engineered from ISA documents and probes (official manual, WMMA instructions section).
- The int8 dot-product entry point is `sudot4` (`sdot4` requires dot1-insts, gfx1150 only).
- `multiProcessorCount` reports 48 WGPs, not 96 CUs: occupancy and grid math must convert.
- hipGraph is available (note the 5-argument `hipGraphInstantiate` signature); graph exec update failures are handled by re-instantiation.

## Build pitfalls

- `CMAKE_CXX_SCAN_FOR_MODULES=OFF` is mandatory; module scanning dyndep otherwise causes full-rebuild loops.
- Single-file probe builds on a distro ROCm layout (no `/opt/rocm`) need explicit `--rocm-path`, device-lib and `-L` flags passed to hipcc.
- The heaviest translation unit (`w8_small_t.cu`) takes hours; incremental builds are otherwise fast.

## Measurement notes

See [BENCHMARKS.md](../BENCHMARKS.md). In short: greedy decoding only, INT8 KV enabled, medians on an idle GPU, and the lossy approximate-attention fast path excluded from baselines.

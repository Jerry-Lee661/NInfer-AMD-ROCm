# NOTICE

This product includes software developed by:

- The NInfer upstream project and contributors: https://github.com/Neroued/ninfer
- Don-Chad/ninfer-3090, the sm_86 community retarget this tree is based on: https://github.com/Don-Chad/ninfer-3090
- The third-party components under `third_party/` (utf8proc, cpp-httplib, and others; see each subdirectory for its own license)

The AMD gfx1100 / ROCm port work in this tree comprises: the `compat/cuda` HIP compatibility shim, the PTX-primitive replacement layer in `src/ops/common/{memory,math,mma}.cuh`, the RDNA3 WMMA and `sudot4` kernel paths, INT8 KV cache routing, and the decode-graph node-fusion series. It was developed between 2026-09 and 2026-10 on Radeon RX 7900 XTX hardware.

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE).

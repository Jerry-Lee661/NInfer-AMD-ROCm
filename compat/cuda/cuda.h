#pragma once
// HIP backend shim: driver API header. The only CUDA consumer (nvfp4_w4a4_tma.cuh) is
// excluded from HIP builds; kept as a forwarder so stray includes resolve.
#include "cuda_runtime.h"

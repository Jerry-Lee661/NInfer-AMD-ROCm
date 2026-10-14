#pragma once
// HIP backend shim: fp16 is provided natively by HIP with CUDA-compatible names.
// The rename header must come first: amd_hip_fp16.h pulls amd_hip_bf16.h in.
#include "hip_bf16_rename.h"
#include <hip/hip_fp16.h>

// Ride the bf16 API along: the rename macros are already active here, and any TU
// that includes fp16 alone still needs __float2bfloat16 & co. defined.
#include "cuda_bf16.h"

#pragma once
// HIP backend shim: CUDA math_constants.h subset actually referenced by NInfer.
// CUDA names these without a suffix; CUDART_* specials are float bit patterns so they
// stay usable from both host and device translation units.
#include <cstdint>

#include <cuda_runtime.h>

namespace ninfer_hip_mc_detail {
__host__ __device__ inline float bits_to_float(unsigned int u) {
    float f;
    __builtin_memcpy(&f, &u, sizeof(f));
    return f;
}
}  // namespace ninfer_hip_mc_detail

#define CUDART_INF_F     ninfer_hip_mc_detail::bits_to_float(0x7f800000u)
#define CUDART_NEG_INF_F ninfer_hip_mc_detail::bits_to_float(0xff800000u)
#define CUDART_NAN_F     ninfer_hip_mc_detail::bits_to_float(0x7fc00000u)

#define kE       2.718281828459045090796f
#define kLog2E   1.44269504088896340736f
#define kLog10E  0.434294481903251827651f
#define kLn2     0.693147180559945309417f
#define kLn10    2.30258509299404568402f
#define kPi      3.141592653589793115998f
#define kPiOver2 1.570796326794896557999f
#define kPiOver4 0.785398163397448278999f
#define k1OverPi 0.318309886183790671538f
#define k2OverPi 0.636619772367581343076f
#define k2OverSqrtPi 1.12837916709551257390f
#define kSqrt2   1.414213562373095145475f
#define kSqrt1Over2 0.707106781186547572737f

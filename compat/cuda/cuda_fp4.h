#pragma once
// HIP backend shim: minimal FP4 E2M1 pair codec with CUDA-compatible names.
// Only the decode path is exercised (NVFP4 weights ride the A16 dequantizing route).
#include <cstdint>

#include <hip/hip_runtime.h>

namespace ninfer_hip_fp4_detail {

// E2M1 nibble: 1 sign, 2 exponent (bias 1), 1 mantissa -> +-{0, .5, 1, 1.5, 2, 3, 4, 6}.
__host__ __device__ inline float e2m1_nibble_to_float(unsigned char n) {
    const unsigned int sign = static_cast<unsigned int>(n & 0x8u) << 28;
    const unsigned int e = (n >> 1) & 3u;
    const unsigned int m = n & 1u;
    if (e == 0u && m == 0u) return __int_as_float(sign);
    unsigned int f32bits;
    if (e == 0u) {
        // Subnormal: m/2 * 2^-1 = m * 2^-2; f32(2^-2) = 0x3D000000.
        f32bits = 0x3D000000u + (m << 23);
    } else {
        f32bits = ((e - 1u + 127u) << 23) | (m << 22);
    }
    return __int_as_float(f32bits | sign);
}

}  // namespace ninfer_hip_fp4_detail

struct __nv_fp4x2_e2m1 {
    unsigned char __x;
    __host__ __device__ __nv_fp4x2_e2m1() = default;
    __host__ __device__ operator float2() const {
        float2 out;
        out.x = ninfer_hip_fp4_detail::e2m1_nibble_to_float(__x & 0x0Fu);
        out.y = ninfer_hip_fp4_detail::e2m1_nibble_to_float(__x >> 4);
        return out;
    }
};

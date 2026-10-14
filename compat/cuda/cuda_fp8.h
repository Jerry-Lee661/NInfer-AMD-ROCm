#pragma once
// HIP backend shim: minimal FP8 E4M3 codec with CUDA-compatible names.
// NInfer only consumes E4M3 (satfinite) through the interfaces below.
#include <cstdint>

#include "hip_bf16_rename.h"
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

typedef unsigned char __nv_fp8_storage_t;
typedef unsigned short __nv_fp8x2_storage_t;

typedef enum __nv_saturation_t { __NV_NOSAT = 0, __NV_SATFINITE = 1 } __nv_saturation_t;
typedef enum __nv_fp8_interpretation_t { __NV_E4M3 = 0, __NV_E5M2 = 1 } __nv_fp8_interpretation_t;

namespace ninfer_hip_fp8_detail {

// E4M3 (1 sign, 4 exponent bias 7, 3 mantissa; no inf; NaN=0x7F/0xFF) -> float.
__host__ __device__ inline float e4m3_to_float(unsigned char v) {
    const unsigned int sign = static_cast<unsigned int>(v & 0x80u) << 24;
    const unsigned int abs = v & 0x7Fu;
    if (abs >= 0x7Fu) return __int_as_float(0x7FC00000u | sign);  // NaN
    unsigned int f32bits;
    if (abs == 0u) {
        f32bits = 0u;
    } else if (abs < 8u) {
        // Subnormal: value = abs * 2^-9; f32(2^-9) = 0x3B000000, adding (abs<<23) is exact.
        f32bits = 0x3B000000u + (abs << 23);
    } else {
        const unsigned int e = abs >> 3;  // 1..15
        const unsigned int m = abs & 7u;
        f32bits = ((e + 120u) << 23) | (m << 20);  // e - 7 + 127
    }
    return __int_as_float(f32bits | sign);
}

// float -> E4M3 satfinite (round-to-nearest-even, clamps to +-448 = 0x7E).
__host__ __device__ inline unsigned char float_to_e4m3_satfinite(float f) {
    const unsigned int bits = __float_as_uint(f);
    const unsigned int sign = (bits >> 24) & 0x80u;
    const unsigned int a = bits & 0x7FFFFFFFu;
    if (a > 0x7F800000u) return static_cast<unsigned char>(sign | 0x7Fu);  // NaN
    if (a > 0x43600000u) return static_cast<unsigned char>(sign | 0x7Eu);  // > 448
    if (a < 0x3C800000u) {
        // Subnormal domain (< 2^-6): round to nearest multiple of 2^-9, ties to even.
        const float t = __int_as_float(a) * 512.0f;  // in [0, 8)
        unsigned int u = static_cast<unsigned int>(t);
        const float frac = t - static_cast<float>(u);
        if (frac > 0.5f || (frac == 0.5f && (u & 1u) != 0u)) ++u;
        if (u == 8u) return static_cast<unsigned char>(sign | 0x38u);  // carry to 2^-6
        return static_cast<unsigned char>(sign | u);
    }
    // Normal: RNE from 23-bit to 3-bit mantissa (drop 20 bits, half-ulp 0x80000).
    const unsigned int rounded = a + 0x7FFFFu + ((a >> 20) & 1u);
    const unsigned int e32 = (rounded >> 23) & 0xFFu;
    const unsigned int m3 = (rounded >> 20) & 7u;
    int e4 = static_cast<int>(e32) - 127 + 7;
    if (e4 < 1) return static_cast<unsigned char>(sign | 0x08u);  // exactly 2^-6
    return static_cast<unsigned char>(sign | static_cast<unsigned int>((e4 << 3) | static_cast<int>(m3)));
}

}  // namespace ninfer_hip_fp8_detail

struct __nv_fp8_e4m3 {
    __nv_fp8_storage_t __x;
    __host__ __device__ __nv_fp8_e4m3() = default;
    __host__ __device__ explicit __nv_fp8_e4m3(float f)
        : __x(ninfer_hip_fp8_detail::float_to_e4m3_satfinite(f)) {}
    __host__ __device__ operator float() const {
        return ninfer_hip_fp8_detail::e4m3_to_float(__x);
    }
};

struct __nv_fp8x2_e4m3 {
    __nv_fp8x2_storage_t __x;
    __host__ __device__ __nv_fp8x2_e4m3() = default;
    __host__ __device__ operator float2() const {
        float2 out;
        out.x = ninfer_hip_fp8_detail::e4m3_to_float(static_cast<unsigned char>(__x & 0xFFu));
        out.y = ninfer_hip_fp8_detail::e4m3_to_float(static_cast<unsigned char>(__x >> 8));
        return out;
    }
    __host__ __device__ operator __half2() const {
        return __half2(__float2half_rn(ninfer_hip_fp8_detail::e4m3_to_float(static_cast<unsigned char>(__x & 0xFFu))),
                       __float2half_rn(ninfer_hip_fp8_detail::e4m3_to_float(static_cast<unsigned char>(__x >> 8))));
    }
};

__host__ __device__ inline __nv_fp8_storage_t
__nv_cvt_float_to_fp8(const float x, const __nv_saturation_t saturate,
                      const __nv_fp8_interpretation_t fp8_interpretation) {
    (void)saturate;
    (void)fp8_interpretation;  // E4M3 satfinite is the only path NInfer exercises.
    return ninfer_hip_fp8_detail::float_to_e4m3_satfinite(x);
}

__host__ __device__ inline __nv_fp8x2_storage_t
__nv_cvt_float2_to_fp8x2(const float2 x, const __nv_saturation_t saturate,
                         const __nv_fp8_interpretation_t fp8_interpretation) {
    return static_cast<__nv_fp8x2_storage_t>(
               __nv_cvt_float_to_fp8(x.x, saturate, fp8_interpretation)) |
           (static_cast<__nv_fp8x2_storage_t>(
                __nv_cvt_float_to_fp8(x.y, saturate, fp8_interpretation))
            << 8);
}

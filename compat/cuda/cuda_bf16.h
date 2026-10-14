#pragma once
// HIP backend shim for bf16: full CUDA-compatible implementation on a public-bits type.
// (The HIP native bf16 API is not referenced from this header at all.)
// the CUDA names here are unambiguous and NInfer can touch __x directly.
// Self-contained: pulls in nothing beyond our runtime shim, so pure host translation
// units including this header never instantiate the HIP device-only bf16 builtins.
#include "cuda_runtime.h"


namespace ninfer_hip_bf16_detail {
__host__ __device__ inline unsigned int bits_of(float f) {
    unsigned int u;
    __builtin_memcpy(&u, &f, sizeof(u));
    return u;
}
__host__ __device__ inline float float_of(unsigned int u) {
    float f;
    __builtin_memcpy(&f, &u, sizeof(f));
    return f;
}
}  // namespace ninfer_hip_bf16_detail

// Reclaim the CUDA names for the definitions below (the renamed HIP versions stay
// untouched under their ninfer_hipref_ identifiers).
#undef __float2bfloat16
#undef __float2bfloat16_rn
#undef __bfloat162float
#undef __bfloat1622float2
#undef __bfloat16_as_ushort
#undef __ushort_as_bfloat16
#undef __low2bfloat16
#undef __high2bfloat16
#undef __low2float
#undef __high2float
#undef __halves2bfloat162
#undef __bfloat162bfloat162
#undef __floats2bfloat162_rn
#undef __hadd2
#undef __hsub2
#undef __hmul2
#undef __hsub2_rn
#undef __habs2

struct __nv_bfloat16 {
    unsigned short __x;

    __host__ __device__ __nv_bfloat16() = default;

    __host__ __device__ __nv_bfloat16(float f) {
        const unsigned int bits = ninfer_hip_bf16_detail::bits_of(f);
        if ((bits & 0x7FFFFFFFu) > 0x7F800000u) {
            __x = static_cast<unsigned short>((bits >> 16) | 0x0040u);
        } else {
            __x = static_cast<unsigned short>((bits + 0x7FFFu + ((bits >> 16) & 1u)) >> 16);
        }
    }


    __host__ __device__ operator float() const {
        return ninfer_hip_bf16_detail::float_of(static_cast<unsigned int>(__x) << 16);
    }

    __host__ __device__ __nv_bfloat16& operator=(float f) {
        *this = __nv_bfloat16(f);
        return *this;
    }
};

struct __nv_bfloat162 {
    __nv_bfloat16 x;
    __nv_bfloat16 y;

    __host__ __device__ __nv_bfloat162() = default;
    __host__ __device__ __nv_bfloat162(const __nv_bfloat16& a, const __nv_bfloat16& b) : x(a), y(b) {}
};

__host__ __device__ inline __nv_bfloat16 __float2bfloat16(float f) { return __nv_bfloat16(f); }

__host__ __device__ inline __nv_bfloat16 __float2bfloat16_rn(float f) { return __nv_bfloat16(f); }

__host__ __device__ inline float __bfloat162float(__nv_bfloat16 v) { return static_cast<float>(v); }

__host__ __device__ inline unsigned short __bfloat16_as_ushort(__nv_bfloat16 v) { return v.__x; }

__host__ __device__ inline __nv_bfloat16 __ushort_as_bfloat16(unsigned short v) {
    __nv_bfloat16 out;
    out.__x = v;
    return out;
}

__host__ __device__ inline __nv_bfloat16 __low2bfloat16(__nv_bfloat162 v) { return v.x; }

__host__ __device__ inline __nv_bfloat16 __high2bfloat16(__nv_bfloat162 v) { return v.y; }

__host__ __device__ inline float __low2float(__nv_bfloat162 v) { return static_cast<float>(v.x); }

__host__ __device__ inline float __high2float(__nv_bfloat162 v) { return static_cast<float>(v.y); }

__host__ __device__ inline __nv_bfloat162 __halves2bfloat162(__nv_bfloat16 lo, __nv_bfloat16 hi) {
    __nv_bfloat162 out;
    out.x = lo;
    out.y = hi;
    return out;
}

__host__ __device__ inline __nv_bfloat162 __bfloat162bfloat162(__nv_bfloat16 v) {
    return __halves2bfloat162(v, v);
}

__host__ __device__ inline __nv_bfloat162 __floats2bfloat162_rn(float lo, float hi) {
    return __halves2bfloat162(__nv_bfloat16(lo), __nv_bfloat16(hi));
}

__host__ __device__ inline float2 __bfloat1622float2(__nv_bfloat162 v) {
    float2 out;
    out.x = static_cast<float>(v.x);
    out.y = static_cast<float>(v.y);
    return out;
}

__host__ __device__ inline __nv_bfloat16 operator+(__nv_bfloat16 a, __nv_bfloat16 b) {
    return __nv_bfloat16(static_cast<float>(a) + static_cast<float>(b));
}
__host__ __device__ inline __nv_bfloat16 operator-(__nv_bfloat16 a, __nv_bfloat16 b) {
    return __nv_bfloat16(static_cast<float>(a) - static_cast<float>(b));
}
__host__ __device__ inline __nv_bfloat16 operator*(__nv_bfloat16 a, __nv_bfloat16 b) {
    return __nv_bfloat16(static_cast<float>(a) * static_cast<float>(b));
}
__host__ __device__ inline __nv_bfloat16 operator/(__nv_bfloat16 a, __nv_bfloat16 b) {
    return __nv_bfloat16(static_cast<float>(a) / static_cast<float>(b));
}
__host__ __device__ inline __nv_bfloat16& operator+=(__nv_bfloat16& a, __nv_bfloat16 b) { a = a + b; return a; }
__host__ __device__ inline __nv_bfloat16& operator-=(__nv_bfloat16& a, __nv_bfloat16 b) { a = a - b; return a; }
__host__ __device__ inline __nv_bfloat16& operator*=(__nv_bfloat16& a, __nv_bfloat16 b) { a = a * b; return a; }
__host__ __device__ inline __nv_bfloat16& operator/=(__nv_bfloat16& a, __nv_bfloat16 b) { a = a / b; return a; }
__host__ __device__ inline bool operator==(__nv_bfloat16 a, __nv_bfloat16 b) {
    return static_cast<float>(a) == static_cast<float>(b);
}
__host__ __device__ inline bool operator!=(__nv_bfloat16 a, __nv_bfloat16 b) { return !(a == b); }
__host__ __device__ inline bool operator<(__nv_bfloat16 a, __nv_bfloat16 b) {
    return static_cast<float>(a) < static_cast<float>(b);
}
__host__ __device__ inline bool operator>(__nv_bfloat16 a, __nv_bfloat16 b) { return b < a; }
__host__ __device__ inline bool operator<=(__nv_bfloat16 a, __nv_bfloat16 b) { return !(a > b); }
__host__ __device__ inline bool operator>=(__nv_bfloat16 a, __nv_bfloat16 b) { return !(a < b); }

__host__ __device__ inline __nv_bfloat162 operator+(__nv_bfloat162 a, __nv_bfloat162 b) {
    return __floats2bfloat162_rn(static_cast<float>(a.x) + static_cast<float>(b.x),
                                 static_cast<float>(a.y) + static_cast<float>(b.y));
}
__host__ __device__ inline __nv_bfloat162 operator-(__nv_bfloat162 a, __nv_bfloat162 b) {
    return __floats2bfloat162_rn(static_cast<float>(a.x) - static_cast<float>(b.x),
                                 static_cast<float>(a.y) - static_cast<float>(b.y));
}
__host__ __device__ inline __nv_bfloat162 operator*(__nv_bfloat162 a, __nv_bfloat162 b) {
    return __floats2bfloat162_rn(static_cast<float>(a.x) * static_cast<float>(b.x),
                                 static_cast<float>(a.y) * static_cast<float>(b.y));
}
__host__ __device__ inline __nv_bfloat162& operator+=(__nv_bfloat162& a, __nv_bfloat162 b) { a = a + b; return a; }
__host__ __device__ inline __nv_bfloat162& operator*=(__nv_bfloat162& a, __nv_bfloat162 b) { a = a * b; return a; }

__host__ __device__ inline __nv_bfloat162 __hadd2(__nv_bfloat162 a, __nv_bfloat162 b) { return a + b; }
__host__ __device__ inline __nv_bfloat162 __hsub2(__nv_bfloat162 a, __nv_bfloat162 b) { return a - b; }
__host__ __device__ inline __nv_bfloat162 __hmul2(__nv_bfloat162 a, __nv_bfloat162 b) { return a * b; }
__host__ __device__ inline __nv_bfloat162 __hsub2_rn(__nv_bfloat162 a, __nv_bfloat162 b) { return a - b; }
__host__ __device__ inline __nv_bfloat162 __habs2(__nv_bfloat162 a) {
    __nv_bfloat162 out;
    out.x = __ushort_as_bfloat16(static_cast<unsigned short>(__bfloat16_as_ushort(a.x) & 0x7FFFu));
    out.y = __ushort_as_bfloat16(static_cast<unsigned short>(__bfloat16_as_ushort(a.y) & 0x7FFFu));
    return out;
}

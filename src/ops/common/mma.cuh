#pragma once

#include "ops/common/memory.cuh"

#if defined(NINFER_HIP)

#include <cstdint>

#include <cuda_fp16.h>

namespace ninfer::ops {

namespace hip_mma_detail {

// Flat-address reconstruction for the 32-bit LDS offsets produced by smem_addr();
// verified constant on gfx1100 / HIP 7.1.52801 (see memory.cuh kLdsFlatBase).
__device__ __forceinline__ const char* lds_char(unsigned addr) {
    return reinterpret_cast<const char*>(kLdsFlatBase | static_cast<unsigned long long>(addr));
}

__device__ __forceinline__ float bf16_pair_get(unsigned pair, int idx) {
    const unsigned half = (idx != 0) ? (pair >> 16) : (pair & 0xFFFFu);
    return __uint_as_float(half << 16);
}

__device__ __forceinline__ float f16_pair_get(unsigned pair, int idx) {
    const unsigned short half =
        static_cast<unsigned short>((idx != 0) ? (pair >> 16) : (pair & 0xFFFFu));
    return __half2float(__ushort_as_half(half));
}

__device__ __forceinline__ int s8_byte_get(unsigned quad, int idx) {
    return static_cast<signed char>((quad >> (8 * idx)) & 0xFFu);
}

// ldmatrix.m8n8.x1: lane l receives row (l>>2)'s b16 pair [2*(l&3), 2*(l&3)+1] of matrix m.
// The row address of matrix m lives in lane 8*m + (l>>2) — the caller-supplied per-lane
// address follows the same lane->(matrix,row) mapping as the PTX instruction.
__device__ __forceinline__ void ldmatrix_x1_core(unsigned& r, unsigned addr, int m) {
    const int lane = static_cast<int>(threadIdx.x & 31u);
    const int row = lane >> 2;
    const int col = lane & 3;
    const unsigned row_addr = __shfl_sync(0xffffffffffffffffULL, addr, 8 * m + row);
    r = *reinterpret_cast<const unsigned*>(lds_char(row_addr) + 4 * col);
}

__device__ __forceinline__ void ldmatrix_x1_t_core(unsigned& r, unsigned addr, int m) {
    // Transposed: lane l receives M^T[l>>2][2*(l&3)..+1] = M[2*(l&3)][l>>2], M[2*(l&3)+1][l>>2].
    const int lane = static_cast<int>(threadIdx.x & 31u);
    const int row = lane >> 2;
    const int col = lane & 3;
    const unsigned addr_a = __shfl_sync(0xffffffffffffffffULL, addr, 8 * m + 2 * col);
    const unsigned addr_b = __shfl_sync(0xffffffffffffffffULL, addr, 8 * m + 2 * col + 1);
    const unsigned short* row_a = reinterpret_cast<const unsigned short*>(lds_char(addr_a));
    const unsigned short* row_b = reinterpret_cast<const unsigned short*>(lds_char(addr_b));
    r = static_cast<unsigned>(row_a[row]) | (static_cast<unsigned>(row_b[row]) << 16);
}

} // namespace hip_mma_detail

__device__ __forceinline__ void ldmatrix_x2(unsigned& r0, unsigned& r1, unsigned addr) {
    hip_mma_detail::ldmatrix_x1_core(r0, addr, 0);
    hip_mma_detail::ldmatrix_x1_core(r1, addr, 1);
}

__device__ __forceinline__ void ldmatrix_x4(unsigned& r0, unsigned& r1, unsigned& r2, unsigned& r3,
                                            unsigned addr) {
    hip_mma_detail::ldmatrix_x1_core(r0, addr, 0);
    hip_mma_detail::ldmatrix_x1_core(r1, addr, 1);
    hip_mma_detail::ldmatrix_x1_core(r2, addr, 2);
    hip_mma_detail::ldmatrix_x1_core(r3, addr, 3);
}

__device__ __forceinline__ void ldmatrix_x2_t(unsigned& r0, unsigned& r1, unsigned addr) {
    hip_mma_detail::ldmatrix_x1_t_core(r0, addr, 0);
    hip_mma_detail::ldmatrix_x1_t_core(r1, addr, 1);
}

__device__ __forceinline__ void ldmatrix_x4_t(unsigned& r0, unsigned& r1, unsigned& r2,
                                              unsigned& r3, unsigned addr) {
    hip_mma_detail::ldmatrix_x1_t_core(r0, addr, 0);
    hip_mma_detail::ldmatrix_x1_t_core(r1, addr, 1);
    hip_mma_detail::ldmatrix_x1_t_core(r2, addr, 2);
    hip_mma_detail::ldmatrix_x1_t_core(r3, addr, 3);
}

// m16n8k16 fragment emulation (bf16/f16 share the layout; the element decode differs):
//   lane l: g = l>>2, i = l&3
//   A: a0={A[g][2i],A[g][2i+1]} a1={A[g+8][2i],A[g+8][2i+1]}
//      a2={A[g][2i+8],A[g][2i+9]} a3={A[g+8][2i+8],A[g+8][2i+9]}
//   B: b0={B[2i][g],B[2i+1][g]} b1={B[2i+8][g],B[2i+9][g]}
//   C: c0=C[g][2i] c1=C[g][2i+1] c2=C[g+8][2i] c3=C[g+8][2i+1]
// Element A[g][k] lives in lane 4g+((k>>1)&3) — the +8 column half stays inside the same
// row group and moves to register a2, not to the next group. Same rule for B[k][n] in lane
// 4n+((k>>1)&3). The lane register pair is selected by (k&8) and the half by (k&1).
__device__ __forceinline__ void mma_bf16(float& c0, float& c1, float& c2, float& c3, unsigned a0,
                                         unsigned a1, unsigned a2, unsigned a3, unsigned b0,
                                         unsigned b1) {
    const int lane = static_cast<int>(threadIdx.x & 31u);
    const int g = lane >> 2;
    const int i = lane & 3;
    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
#pragma unroll
    for (int k = 0; k < 16; ++k) {
        const unsigned hi_sel = (k & 8) ? 1u : 0u;
        const int holder = static_cast<int>((k >> 1) & 3);
        const float a_lo = hip_mma_detail::bf16_pair_get(
            __shfl_sync(0xffffffffffffffffULL, hi_sel ? a2 : a0, 4 * g + holder), k & 1);
        const float a_hi = hip_mma_detail::bf16_pair_get(
            __shfl_sync(0xffffffffffffffffULL, hi_sel ? a3 : a1, 4 * g + holder), k & 1);
        const float b_n0 = hip_mma_detail::bf16_pair_get(
            __shfl_sync(0xffffffffffffffffULL, hi_sel ? b1 : b0, 4 * (2 * i) + holder), k & 1);
        const float b_n1 = hip_mma_detail::bf16_pair_get(
            __shfl_sync(0xffffffffffffffffULL, hi_sel ? b1 : b0, 4 * (2 * i + 1) + holder), k & 1);
        acc0 += a_lo * b_n0;
        acc1 += a_lo * b_n1;
        acc2 += a_hi * b_n0;
        acc3 += a_hi * b_n1;
    }
    c0 += acc0;
    c1 += acc1;
    c2 += acc2;
    c3 += acc3;
}

__device__ __forceinline__ void mma_f16(float& c0, float& c1, float& c2, float& c3, unsigned a0,
                                        unsigned a1, unsigned a2, unsigned a3, unsigned b0,
                                        unsigned b1) {
    const int lane = static_cast<int>(threadIdx.x & 31u);
    const int g = lane >> 2;
    const int i = lane & 3;
    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
#pragma unroll
    for (int k = 0; k < 16; ++k) {
        const unsigned hi_sel = (k & 8) ? 1u : 0u;
        const int holder = static_cast<int>((k >> 1) & 3);
        const float a_lo = hip_mma_detail::f16_pair_get(
            __shfl_sync(0xffffffffffffffffULL, hi_sel ? a2 : a0, 4 * g + holder), k & 1);
        const float a_hi = hip_mma_detail::f16_pair_get(
            __shfl_sync(0xffffffffffffffffULL, hi_sel ? a3 : a1, 4 * g + holder), k & 1);
        const float b_n0 = hip_mma_detail::f16_pair_get(
            __shfl_sync(0xffffffffffffffffULL, hi_sel ? b1 : b0, 4 * (2 * i) + holder), k & 1);
        const float b_n1 = hip_mma_detail::f16_pair_get(
            __shfl_sync(0xffffffffffffffffULL, hi_sel ? b1 : b0, 4 * (2 * i + 1) + holder), k & 1);
        acc0 += a_lo * b_n0;
        acc1 += a_lo * b_n1;
        acc2 += a_hi * b_n0;
        acc3 += a_hi * b_n1;
    }
    c0 += acc0;
    c1 += acc1;
    c2 += acc2;
    c3 += acc3;
}

// m16n8k32 s8: four int8 per b32. A: a0={A[g][4i..4i+3]} a1={A[g+8][4i..]}
// a2={A[g][16+4i..]} a3={A[g+8][16+4i..]}; B: b0={B[4i..4i+3][g]} b1={B[16+4i..19][g]}.
// A[g][k] -> lane 4g+((k>>2)&3), a0/a2 (k<16 / k>=16), byte (k&3); B[k][n] -> lane
// 4n+((k>>2)&3), b0/b1, byte (k&3) (byte index is k&3 in both halves since 16+4j has byte
// index j).
__device__ __forceinline__ void mma_s8(int& c0, int& c1, int& c2, int& c3, unsigned a0, unsigned a1,
                                       unsigned a2, unsigned a3, unsigned b0, unsigned b1) {
    const int lane = static_cast<int>(threadIdx.x & 31u);
    const int g = lane >> 2;
    const int i = lane & 3;
    int acc0 = 0, acc1 = 0, acc2 = 0, acc3 = 0;
#pragma unroll
    for (int k = 0; k < 32; ++k) {
        const unsigned hi_sel = (k & 16) ? 1u : 0u;
        const int holder = static_cast<int>((k >> 2) & 3);
        const int a_lo = hip_mma_detail::s8_byte_get(
            __shfl_sync(0xffffffffffffffffULL, hi_sel ? a2 : a0, 4 * g + holder), k & 3);
        const int a_hi = hip_mma_detail::s8_byte_get(
            __shfl_sync(0xffffffffffffffffULL, hi_sel ? a3 : a1, 4 * g + holder), k & 3);
        const int b_n0 = hip_mma_detail::s8_byte_get(
            __shfl_sync(0xffffffffffffffffULL, hi_sel ? b1 : b0, 4 * (2 * i) + holder), k & 3);
        const int b_n1 = hip_mma_detail::s8_byte_get(
            __shfl_sync(0xffffffffffffffffULL, hi_sel ? b1 : b0, 4 * (2 * i + 1) + holder), k & 3);
        acc0 += a_lo * b_n0;
        acc1 += a_lo * b_n1;
        acc2 += a_hi * b_n0;
        acc3 += a_hi * b_n1;
    }
    c0 += acc0;
    c1 += acc1;
    c2 += acc2;
    c3 += acc3;
}

// Blackwell-only fp8/fp4 mma variants: their owning translation units are excluded from
// HIP builds (A8/W4A4 isolation in src/CMakeLists.txt), and the A16 routes never call
// them. Any unexpected instantiation traps loudly instead of computing garbage.
__device__ __forceinline__ void mma_fp8_e4m3(float& c0, float& c1, float& c2, float& c3,
                                             unsigned a0, unsigned a1, unsigned a2, unsigned a3,
                                             unsigned b0, unsigned b1) {
    (void)c0; (void)c1; (void)c2; (void)c3;
    (void)a0; (void)a1; (void)a2; (void)a3; (void)b0; (void)b1;
    __builtin_trap();
}

__device__ __forceinline__ void mma_tf32_bits(float& c0, float& c1, float& c2, float& c3,
                                              unsigned a0, unsigned a1, unsigned a2, unsigned a3,
                                              unsigned b0, unsigned b1) {
    // m16n8k8: A: a0=A[g][2i] a1=A[g+8][2i] a2=A[g][2i+1] a3=A[g+8][2i+1];
    // B: b0=B[2i][g] b1=B[2i+1][g]. A[g][k] -> lane 4g+(k>>1), (k&1)?a2:a0;
    // B[k][n] -> lane 4n+(k>>1), (k&1)?b1:b0. tf32 rides the upper 19 bits of f32.
    const int lane = static_cast<int>(threadIdx.x & 31u);
    const int g = lane >> 2;
    const int i = lane & 3;
    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
#pragma unroll
    for (int k = 0; k < 8; ++k) {
        const unsigned sel = static_cast<unsigned>(k & 1);
        const float a_lo = __uint_as_float(
            __shfl_sync(0xffffffffffffffffULL, sel ? a2 : a0, 4 * g + (k >> 1)) & 0xFFFFE000u);
        const float a_hi = __uint_as_float(
            __shfl_sync(0xffffffffffffffffULL, sel ? a3 : a1, 4 * g + (k >> 1)) & 0xFFFFE000u);
        const float b_n0 = __uint_as_float(
            __shfl_sync(0xffffffffffffffffULL, sel ? b1 : b0, 4 * (2 * i) + (k >> 1)) & 0xFFFFE000u);
        const float b_n1 = __uint_as_float(
            __shfl_sync(0xffffffffffffffffULL, sel ? b1 : b0, 4 * (2 * i + 1) + (k >> 1)) & 0xFFFFE000u);
        acc0 += a_lo * b_n0;
        acc1 += a_lo * b_n1;
        acc2 += a_hi * b_n0;
        acc3 += a_hi * b_n1;
    }
    c0 += acc0;
    c1 += acc1;
    c2 += acc2;
    c3 += acc3;
}

__device__ __forceinline__ void mma_tf32(float& c0, float& c1, float& c2, float& c3, float a0,
                                         float a1, float a2, float a3, float b0, float b1) {
    mma_tf32_bits(c0, c1, c2, c3, __float_as_uint(a0), __float_as_uint(a1), __float_as_uint(a2),
                  __float_as_uint(a3), __float_as_uint(b0), __float_as_uint(b1));
}

__device__ __forceinline__ void mma_nvfp4_e4m3(float& c0, float& c1, float& c2, float& c3,
                                               unsigned a0, unsigned a1, unsigned a2, unsigned a3,
                                               unsigned b0, unsigned b1, unsigned sfa,
                                               unsigned sfb) {
    (void)c0; (void)c1; (void)c2; (void)c3;
    (void)a0; (void)a1; (void)a2; (void)a3; (void)b0; (void)b1; (void)sfa; (void)sfb;
    __builtin_trap();
}

} // namespace ninfer::ops

#else // CUDA backend (upstream implementation)

namespace ninfer::ops {

__device__ __forceinline__ void ldmatrix_x2(unsigned& r0, unsigned& r1, unsigned addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
                 : "=r"(r0), "=r"(r1)
                 : "r"(addr));
}

__device__ __forceinline__ void ldmatrix_x4(unsigned& r0, unsigned& r1, unsigned& r2, unsigned& r3,
                                            unsigned addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
                 : "r"(addr));
}

__device__ __forceinline__ void ldmatrix_x2_t(unsigned& r0, unsigned& r1, unsigned addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
                 : "=r"(r0), "=r"(r1)
                 : "r"(addr));
}

__device__ __forceinline__ void ldmatrix_x4_t(unsigned& r0, unsigned& r1, unsigned& r2,
                                              unsigned& r3, unsigned addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
                 : "r"(addr));
}

__device__ __forceinline__ void mma_bf16(float& c0, float& c1, float& c2, float& c3, unsigned a0,
                                         unsigned a1, unsigned a2, unsigned a3, unsigned b0,
                                         unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void mma_f16(float& c0, float& c1, float& c2, float& c3, unsigned a0,
                                        unsigned a1, unsigned a2, unsigned a3, unsigned b0,
                                        unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void mma_s8(int& c0, int& c1, int& c2, int& c3, unsigned a0, unsigned a1,
                                       unsigned a2, unsigned a3, unsigned b0, unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+r"(c0), "+r"(c1), "+r"(c2), "+r"(c3)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void mma_fp8_e4m3(float& c0, float& c1, float& c2, float& c3,
                                             unsigned a0, unsigned a1, unsigned a2, unsigned a3,
                                             unsigned b0, unsigned b1) {
    asm volatile("mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void mma_tf32_bits(float& c0, float& c1, float& c2, float& c3,
                                              unsigned a0, unsigned a1, unsigned a2, unsigned a3,
                                              unsigned b0, unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void mma_tf32(float& c0, float& c1, float& c2, float& c3, float a0,
                                         float a1, float a2, float a3, float b0, float b1) {
    mma_tf32_bits(c0, c1, c2, c3, __float_as_uint(a0), __float_as_uint(a1), __float_as_uint(a2),
                  __float_as_uint(a3), __float_as_uint(b0), __float_as_uint(b1));
}

__device__ __forceinline__ void mma_nvfp4_e4m3(float& c0, float& c1, float& c2, float& c3,
                                               unsigned a0, unsigned a1, unsigned a2, unsigned a3,
                                               unsigned b0, unsigned b1, unsigned sfa,
                                               unsigned sfb) {
    constexpr unsigned short kScaleBlockId  = 0;
    constexpr unsigned short kScaleThreadId = 0;
    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X."
                 "m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
                 "{%0,%1,%2,%3}, "
                 "{%4,%5,%6,%7}, "
                 "{%8,%9}, "
                 "{%0,%1,%2,%3}, "
                 "{%10}, "
                 "{%11,%12}, "
                 "{%13}, "
                 "{%14,%15};\n"
                 : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(sfa),
                   "h"(kScaleBlockId), "h"(kScaleThreadId), "r"(sfb), "h"(kScaleBlockId),
                   "h"(kScaleThreadId));
}

} // namespace ninfer::ops

#endif // NINFER_HIP / CUDA backend select

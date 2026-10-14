#pragma once

// Q5G64 RowSplit x BF16 Tensor Core GEMM, real WMMA variant.
//
// Same tiling, staging and epilogue as q5_rowsplit_gemm_mma_kernel; only the inner loop changes:
// the m16n8k16 emulation is replaced by the hardware v_wmma_f32_16x16x16_bf16_w32 builtin, whose
// operands are read straight out of LDS. Numerics follow the rule that the q4 swiglu kernel had to
// learn the hard way (amd-research.md §3.7.4): the tensor core sees the *unscaled* integer codes
// (|q| <= 16 is exact in bf16) and the group scale is folded into the fp32 accumulator once per K
// tile, which is exactly one quant group here.
//
// Fragment contract (measured on gfx1100 by tools/hip_wmma_selftest.cu): builtin = (2nd arg) x
// (1st arg); A slot (activations, [k][token]) lane word j = {X[2j][col], X[2j+1][col]},
// col = lane & 15; B slot (weights, [row][k]) lane word j = W[row][2j..2j+1], row = lane & 15;
// C lane element j = C[row][2j + (lane >> 4)], row = lane & 15. Both operands are 32-byte
// contiguous runs, so the shared tiles are row-major with an 8-element pad per row (stride 72
// elements = 144 B = 36 banks) to keep the 32-byte per-lane reads on distinct banks.

#include "ops/common/math.cuh"
#include "ops/common/memory.cuh"
#include "ops/linear/q5/q5_rowsplit_storage.cuh"

#include <cuda_bf16.h>

#include <cstdint>

#if defined(NINFER_HIP)

namespace ninfer::ops::detail {

namespace q5_wmma_cfg {
constexpr int kBM      = 64;
constexpr int kBN      = 128;
constexpr int kBK      = 64;
constexpr int kStride  = kBK + 8;
constexpr int kKSub    = kBK / 16;
constexpr int kThreads = 256;
} // namespace q5_wmma_cfg

// Defined in q5_rowsplit_gemm_mma.cu: shared by the plain GEMM and the residual-add route.
bool q5_wmma_enabled();

using Q5WmmaA8  = unsigned __attribute__((ext_vector_type(8)));
using Q5WmmaB16 = std::uint16_t __attribute__((ext_vector_type(16)));
using Q5WmmaC8F = float __attribute__((ext_vector_type(8)));

__device__ __forceinline__ Q5WmmaA8 q5_wmma_load_a(const __nv_bfloat16* row) {
    const uint4 lo = *reinterpret_cast<const uint4*>(row);
    const uint4 hi = *reinterpret_cast<const uint4*>(row + 8);
    Q5WmmaA8 va;
    va[0] = lo.x;
    va[1] = lo.y;
    va[2] = lo.z;
    va[3] = lo.w;
    va[4] = hi.x;
    va[5] = hi.y;
    va[6] = hi.z;
    va[7] = hi.w;
    return va;
}

__device__ __forceinline__ Q5WmmaB16 q5_wmma_load_b(const __nv_bfloat16* row) {
    const uint4 lo = *reinterpret_cast<const uint4*>(row);
    const uint4 hi = *reinterpret_cast<const uint4*>(row + 8);
    const unsigned words[8] = {lo.x, lo.y, lo.z, lo.w, hi.x, hi.y, hi.z, hi.w};
    Q5WmmaB16 vb;
#pragma unroll
    for (int w = 0; w < 8; ++w) {
        vb[2 * w]     = static_cast<std::uint16_t>(words[w] & 0xFFFFu);
        vb[2 * w + 1] = static_cast<std::uint16_t>(words[w] >> 16);
    }
    return vb;
}

// Four code bytes + their high-bit byte -> 4 bf16 pairs holding the *unscaled* 5-bit integers.
// The storage convention is value = (x ^ 16) - 16 for a 5-bit code x, which is the plain
// sign extension of x; 128 + (x ^ 16) minus 144 lands exactly on it (|q| <= 16 is exact in bf16).
__device__ __forceinline__ void q5_wmma_decode_eight(unsigned word, unsigned high_byte,
                                                     unsigned (&out)[4]) {
    constexpr unsigned kBase = 0x4300u; // bf16 128.0 in the low byte
    constexpr unsigned kBias = 0x4310u; // bf16 144.0
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const unsigned byte = (word >> (8 * i)) & 0xFFu;
        const unsigned x0   = ((byte & 0x0Fu) | (((high_byte >> (2 * i)) & 1u) << 4)) ^ 0x10u;
        const unsigned x1   = ((byte >> 4) | (((high_byte >> (2 * i + 1)) & 1u) << 4)) ^ 0x10u;
        const unsigned pair = (kBase | x0) | ((kBase | x1) << 16);
        const __nv_bfloat162 biased = *reinterpret_cast<const __nv_bfloat162*>(&pair);
        const unsigned bias         = kBias | (kBias << 16);
        const __nv_bfloat162 value  = __hsub2(biased, *reinterpret_cast<const __nv_bfloat162*>(&bias));
        out[i]                      = *reinterpret_cast<const unsigned*>(&value);
    }
}

template <int Stages, class Epilogue>
__global__ __launch_bounds__(q5_wmma_cfg::kThreads, 1) void q5_rowsplit_gemm_wmma_kernel(
    const __nv_bfloat16* __restrict__ x, const std::uint8_t* __restrict__ codes,
    const std::uint8_t* __restrict__ high, const std::uint8_t* __restrict__ scales,
    const __nv_bfloat16* __restrict__ residual, __nv_bfloat16* __restrict__ out,
    std::int32_t rows, std::int32_t k, std::int32_t cols, std::int32_t padded_k) {
    using ninfer::ops::cp_async;
    using ninfer::ops::cp_commit;
    using ninfer::ops::cp_wait;
    using ninfer::ops::store_vec;
    constexpr int BM      = q5_wmma_cfg::kBM;
    constexpr int BN      = q5_wmma_cfg::kBN;
    constexpr int BK      = q5_wmma_cfg::kBK;
    constexpr int ST      = q5_wmma_cfg::kStride;
    constexpr int KSUB    = q5_wmma_cfg::kKSub;
    constexpr int THREADS = q5_wmma_cfg::kThreads;

    __shared__ __align__(16) __nv_bfloat16 s_x[Stages][BN * ST];
    __shared__ __align__(16) __nv_bfloat16 s_w[BM * ST];
    __shared__ __align__(16) std::uint8_t s_c[Stages][BM * Q5RowSplitStorage::kCodeBytesPerGroup];
    __shared__ __align__(16) std::uint8_t s_h[Stages][BM * Q5RowSplitStorage::kHighBytesPerGroup];
    __shared__ __align__(16) std::uint16_t s_s[Stages][BM];

    const int tid   = static_cast<int>(threadIdx.x);
    const int warp  = tid >> 5;
    const int lane  = tid & 31;
    const int wm    = warp & 1;
    const int wn    = warp >> 1;
    const int row0  = static_cast<int>(blockIdx.x) * BM;
    const int col0  = static_cast<int>(blockIdx.y) * BN;
    const int groups_per_row = padded_k / Q5RowSplitStorage::kGroupK;
    const int nkt            = padded_k / BK;

    Q5WmmaC8F acc[2][2];
#pragma unroll
    for (int mi = 0; mi < 2; ++mi) {
#pragma unroll
        for (int ni = 0; ni < 2; ++ni) {
#pragma unroll
            for (int e = 0; e < 8; ++e) { acc[mi][ni][e] = 0.0f; }
        }
    }

    auto stage_load = [&](int stage, int kt) {
        const int k0  = kt * BK;
        const int grp = k0 / Q5RowSplitStorage::kGroupK;
#pragma unroll 1
        for (int item = tid; item < BN * (BK / 8); item += THREADS) {
            const int tl       = item >> 3;
            const int kc       = item & 7;
            __nv_bfloat16* dst = &s_x[stage][tl * ST + kc * 8];
            if (col0 + tl < cols && k0 + kc * 8 + 8 <= k) {
                cp_async<16>(dst, &x[static_cast<std::int64_t>(col0 + tl) * k + k0 + kc * 8]);
            } else {
                store_vec(dst, make_int4(0, 0, 0, 0));
            }
        }
        if (tid < BM * 2) {
            const int row  = tid >> 1;
            const int half = tid & 1;
            auto* dst      = &s_c[stage][row * 32 + half * 16];
            if (row0 + row < rows) {
                const std::int64_t gi = static_cast<std::int64_t>(row0 + row) * groups_per_row + grp;
                cp_async<16>(dst, &codes[gi * Q5RowSplitStorage::kCodeBytesPerGroup + half * 16]);
            } else {
                store_vec(dst, make_int4(0, 0, 0, 0));
            }
        }
        if (tid >= THREADS - BM && tid < THREADS - BM + BM) {
            const int row = tid - (THREADS - BM);
            if (row0 + row < rows) {
                const std::int64_t gi = static_cast<std::int64_t>(row0 + row) * groups_per_row + grp;
                cp_async<8>(&s_h[stage][row * Q5RowSplitStorage::kHighBytesPerGroup],
                            &high[gi * Q5RowSplitStorage::kHighBytesPerGroup]);
            } else {
                store_vec(&s_h[stage][row * Q5RowSplitStorage::kHighBytesPerGroup],
                          static_cast<std::uint64_t>(0));
            }
        }
        if (tid < BM) {
            const int row       = tid;
            std::uint16_t value = 0;
            if (row0 + row < rows) {
                const std::int64_t gi = static_cast<std::int64_t>(row0 + row) * groups_per_row + grp;
                value = *reinterpret_cast<const std::uint16_t*>(
                    &scales[gi * Q5RowSplitStorage::kScaleBytesPerGroup]);
            }
            s_s[stage][row] = value;
        }
    };

    // Four threads per row, 16 weights each: two 4-byte code words with their high-bit bytes.
    auto decode_weights = [&](int stage) {
        const int row           = tid >> 2;
        const int sub           = tid & 3;
        const std::uint8_t* src = &s_c[stage][row * 32 + sub * 8];
        const unsigned low      = *reinterpret_cast<const unsigned*>(src);
        const unsigned highw    = *reinterpret_cast<const unsigned*>(src + 4);
        const std::uint8_t* hb  = &s_h[stage][row * Q5RowSplitStorage::kHighBytesPerGroup];
        unsigned lo[4];
        unsigned hi[4];
        q5_wmma_decode_eight(low, hb[sub * 2], lo);
        q5_wmma_decode_eight(highw, hb[sub * 2 + 1], hi);
        __nv_bfloat162* dst = reinterpret_cast<__nv_bfloat162*>(&s_w[row * ST + sub * 16]);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            dst[i]     = *reinterpret_cast<const __nv_bfloat162*>(&lo[i]);
            dst[4 + i] = *reinterpret_cast<const __nv_bfloat162*>(&hi[i]);
        }
    };

#pragma unroll
    for (int s = 0; s < Stages; ++s) {
        if (s < nkt) { stage_load(s, s); }
        cp_commit();
    }

    for (int it = 0; it < nkt; ++it) {
        const int stage = it % Stages;
        cp_wait<Stages - 1>();
        __syncthreads();
        decode_weights(stage);
        __syncthreads();

        const __nv_bfloat16* arow[2];
        arow[0] = &s_x[stage][(wn * 32 + (lane & 15)) * ST];
        arow[1] = arow[0] + 16 * ST;
        const __nv_bfloat16* brow[2];
        brow[0] = &s_w[(wm * 16 + (lane & 15)) * ST];
        brow[1] = &s_w[(32 + wm * 16 + (lane & 15)) * ST];

        Q5WmmaC8F part[2][2];
#pragma unroll
        for (int mi = 0; mi < 2; ++mi) {
#pragma unroll
            for (int ni = 0; ni < 2; ++ni) {
#pragma unroll
                for (int e = 0; e < 8; ++e) { part[mi][ni][e] = 0.0f; }
            }
        }

#pragma unroll
        for (int ks = 0; ks < KSUB; ++ks) {
            Q5WmmaA8 va[2];
            Q5WmmaB16 vb[2];
#pragma unroll
            for (int ni = 0; ni < 2; ++ni) { va[ni] = q5_wmma_load_a(arow[ni] + ks * 16); }
#pragma unroll
            for (int mi = 0; mi < 2; ++mi) { vb[mi] = q5_wmma_load_b(brow[mi] + ks * 16); }
#pragma unroll
            for (int mi = 0; mi < 2; ++mi) {
#pragma unroll
                for (int ni = 0; ni < 2; ++ni) {
                    part[mi][ni] =
                        __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(va[ni], vb[mi], part[mi][ni]);
                }
            }
        }

        const float scale[2] = {__half2float(__ushort_as_half(s_s[stage][wm * 16 + (lane & 15)])),
                                __half2float(
                                    __ushort_as_half(s_s[stage][32 + wm * 16 + (lane & 15)]))};
#pragma unroll
        for (int mi = 0; mi < 2; ++mi) {
#pragma unroll
            for (int ni = 0; ni < 2; ++ni) {
#pragma unroll
                for (int e = 0; e < 8; ++e) {
                    acc[mi][ni][e] = fmaf(scale[mi], part[mi][ni][e], acc[mi][ni][e]);
                }
            }
        }

        __syncthreads();
        const int next = it + Stages;
        if (next < nkt) { stage_load(stage, next); }
        cp_commit();
    }

#pragma unroll
    for (int mi = 0; mi < 2; ++mi) {
        const int row = row0 + mi * 32 + wm * 16 + (lane & 15);
        if (row < rows) {
#pragma unroll
            for (int ni = 0; ni < 2; ++ni) {
                const int col_base = col0 + wn * 32 + ni * 16 + (lane >> 4);
#pragma unroll
                for (int e = 0; e < 8; ++e) {
                    const int col = col_base + 2 * e;
                    if (col < cols) {
                        const std::int64_t index =
                            static_cast<std::int64_t>(col) * rows + row;
                        const float value = acc[mi][ni][e];
                        out[index]        = __float2bfloat16_rn(
                            Epilogue::apply(value, residual == nullptr ? 0.0f
                                                                       : __bfloat162float(residual[index])));
                    }
                }
            }
        }
    }
}

// Q5MmaEpilogue is shared with the emulated kernel; these adapters keep the same tag names.
struct Q5WmmaStore {
    static __device__ __forceinline__ float apply(float value, float) { return value; }
};

struct Q5WmmaAddResidual {
    static __device__ __forceinline__ float apply(float value, float residual) {
        return value + residual;
    }
};

} // namespace ninfer::ops::detail

#endif // NINFER_HIP

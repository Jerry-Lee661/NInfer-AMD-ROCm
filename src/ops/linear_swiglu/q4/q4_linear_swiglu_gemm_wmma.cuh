#pragma once

// WMMA variant of the folded gate/up Q4 kernel. Same tiling, staging and epilogue as
// q4_linear_swiglu_mma_split_half_pair_kernel; only the inner loop changes: the mma_bf16
// fragment emulation is replaced by the hardware v_wmma_f32_16x16x16_bf16_w32 builtin, whose
// operands are read straight out of LDS.
//
// Fragment contract (measured on gfx1100 by tools/hip_wmma_selftest.cu, re-validated end to end
// against a host reference by tools/q4_swiglu_wmma_probe.cu):
//   builtin = (second argument) x (first argument);
//   A slot (activations, X^T in [k][token] order): lane's word j holds {X[2j][col], X[2j+1][col]}
//     with col = lane & 15, i.e. the lane's own token row over 16 consecutive k;
//   B slot (weights, [row][k] order): lane's word j holds W[row][2j..2j+1] with row = lane & 15;
//   C: lane's element j holds C[row][2j + (lane >> 4)] with row = lane & 15.
// Both operands are therefore 32-byte contiguous runs of one row, which is why the shared tiles
// are plain row-major with a single 8-element pad per row (stride 72 elements = 144 B = 36 banks):
// the 36-word stride walks the bank space by 4, so a 16-row column of 32-byte reads lands on
// distinct banks, where an unpadded 128-byte stride would put all of them on one bank.

#include "ops/common/math.cuh"
#include "ops/common/memory.cuh"
#include "ops/linear/q4/q4_rowsplit_storage.cuh"

#include <cuda_bf16.h>

#include <cstdint>

#if defined(NINFER_HIP)

namespace ninfer::ops::detail {

// Folded gate/up geometry: BM = 64 rows (32 gate rows followed by their 32 paired up rows),
// BN = 128 tokens, BK = 64 = one q4 group. 8 warps: 2 in M (each owning a gate 16-row and the
// matching up 16-row tile, so the SiLU epilogue never crosses warps), 4 in N.
namespace swiglu_wmma_cfg {
constexpr int kBM      = 64;
constexpr int kPM      = kBM / 2;
constexpr int kBN      = 128;
constexpr int kBK      = 64;
constexpr int kStride  = kBK + 8;
constexpr int kKSub    = kBK / 16;
constexpr int kThreads = 256;
} // namespace swiglu_wmma_cfg

using WmmaA8  = unsigned __attribute__((ext_vector_type(8)));
using WmmaB16 = std::uint16_t __attribute__((ext_vector_type(16)));
using WmmaC8F = float __attribute__((ext_vector_type(8)));

__device__ __forceinline__ WmmaA8 wmma_load_a(const __nv_bfloat16* row) {
    const uint4 lo = *reinterpret_cast<const uint4*>(row);
    const uint4 hi = *reinterpret_cast<const uint4*>(row + 8);
    WmmaA8 va;
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

__device__ __forceinline__ WmmaB16 wmma_load_b(const __nv_bfloat16* row) {
    const uint4 lo = *reinterpret_cast<const uint4*>(row);
    const uint4 hi = *reinterpret_cast<const uint4*>(row + 8);
    const unsigned words[8] = {lo.x, lo.y, lo.z, lo.w, hi.x, hi.y, hi.z, hi.w};
    WmmaB16 vb;
#pragma unroll
    for (int w = 0; w < 8; ++w) {
        vb[2 * w]     = static_cast<std::uint16_t>(words[w] & 0xFFFFu);
        vb[2 * w + 1] = static_cast<std::uint16_t>(words[w] >> 16);
    }
    return vb;
}

template <int Stages>
__global__ __launch_bounds__(swiglu_wmma_cfg::kThreads, 1) void q4_linear_swiglu_wmma_kernel(
    const __nv_bfloat16* __restrict__ x, const std::uint8_t* __restrict__ codes,
    const std::uint8_t* __restrict__ scales, __nv_bfloat16* __restrict__ out,
    std::int32_t intermediate, std::int32_t k, std::int32_t t, std::int32_t padded_k) {
    using ninfer::ops::cp_async;
    using ninfer::ops::cp_commit;
    using ninfer::ops::cp_wait;
    using ninfer::ops::store_vec;
    constexpr int BM      = swiglu_wmma_cfg::kBM;
    constexpr int PM      = swiglu_wmma_cfg::kPM;
    constexpr int BN      = swiglu_wmma_cfg::kBN;
    constexpr int BK      = swiglu_wmma_cfg::kBK;
    constexpr int ST      = swiglu_wmma_cfg::kStride;
    constexpr int KSUB    = swiglu_wmma_cfg::kKSub;
    constexpr int THREADS = swiglu_wmma_cfg::kThreads;

    __shared__ __align__(16) __nv_bfloat16 s_x[Stages][BN * ST];
    __shared__ __align__(16) __nv_bfloat16 s_w[BM * ST];
    __shared__ __align__(16) std::uint8_t s_c[Stages][BM * 32];
    __shared__ __align__(16) std::uint16_t s_s[Stages][BM];

    const int tid  = static_cast<int>(threadIdx.x);
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int wm   = warp & 1; // folded rows: gate wm*16.. and up PM + wm*16..
    const int wn   = warp >> 1; // columns wn*32..
    const int m0   = static_cast<int>(blockIdx.x) * PM;
    const int kg   = padded_k >> 6;
    const int nkt  = padded_k / BK;

    WmmaC8F acc[2][2];
#pragma unroll
    for (int mi = 0; mi < 2; ++mi) {
#pragma unroll
        for (int ni = 0; ni < 2; ++ni) {
#pragma unroll
            for (int e = 0; e < 8; ++e) { acc[mi][ni][e] = 0.0f; }
        }
    }

    auto global_row = [&](int row) { return m0 + (row & (PM - 1)) + (row / PM) * intermediate; };

    auto stage_load = [&](int stage, int kt) {
        const int k0 = kt * BK;
        const int g  = kt;
#pragma unroll 1
        for (int item = tid; item < BN * (BK / 8); item += THREADS) {
            const int tl       = item >> 3;
            const int kc       = item & 7;
            __nv_bfloat16* dst = &s_x[stage][tl * ST + kc * 8];
            if (tl < t && k0 + kc * 8 + 8 <= k) {
                cp_async<16>(dst, &x[static_cast<std::int64_t>(tl) * k + k0 + kc * 8]);
            } else {
                store_vec(dst, make_int4(0, 0, 0, 0));
            }
        }
        if (tid < BM * 2) {
            const int row  = tid >> 1;
            const int half = tid & 1;
            const int gidx = global_row(row);
            auto* dst      = &s_c[stage][row * 32 + half * 16];
            if (m0 + (row & (PM - 1)) < intermediate) {
                cp_async<16>(dst, &codes[(static_cast<std::int64_t>(gidx) * kg + g) * 32 + half * 16]);
            } else {
                store_vec(dst, make_int4(0, 0, 0, 0));
            }
        }
        if (tid >= THREADS - BM) {
            const int row       = tid - (THREADS - BM);
            const int gidx      = global_row(row);
            std::uint16_t value = 0;
            if (m0 + (row & (PM - 1)) < intermediate) {
                value = *reinterpret_cast<const std::uint16_t*>(
                    &scales[(static_cast<std::int64_t>(gidx) * kg + g) * 2]);
            }
            s_s[stage][row] = value;
        }
    };

    // The whole tile's raw codes are in place: decode 64 rows x 64 weights. decode_eight yields the
    // *unscaled* codes, which bf16 represents exactly ((n ^ 8) - 8 is in [-8, 7]), so the tensor
    // core sees exact integers and the group scale is folded into the fp32 accumulator once per K
    // tile instead. Rounding the scale into bf16 here instead would cost ~0.4% on every weight of
    // the group and, measured against the emulated path, shifted up to 8% of the outputs.
    auto decode_weights = [&](int stage) {
        const int row           = tid >> 2;
        const int sub           = tid & 3;
        const std::uint8_t* src = &s_c[stage][row * 32 + sub * 8];
        const unsigned low      = *reinterpret_cast<const unsigned*>(src);
        const unsigned high     = *reinterpret_cast<const unsigned*>(src + 4);
        unsigned lo[4];
        unsigned hi[4];
        Q4MmaDecodeAtom::decode_eight(low, lo);
        Q4MmaDecodeAtom::decode_eight(high, hi);
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
        brow[1] = &s_w[(PM + wm * 16 + (lane & 15)) * ST];

        // One K tile is exactly one quant group, so the tensor core accumulates only unscaled
        // integer products here; the group scale is applied to the fp32 accumulator afterwards.
        WmmaC8F part[2][2];
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
            WmmaA8 va[2];
            WmmaB16 vb[2];
#pragma unroll
            for (int ni = 0; ni < 2; ++ni) { va[ni] = wmma_load_a(arow[ni] + ks * 16); }
#pragma unroll
            for (int mi = 0; mi < 2; ++mi) { vb[mi] = wmma_load_b(brow[mi] + ks * 16); }
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
                                    __ushort_as_half(s_s[stage][PM + wm * 16 + (lane & 15)]))};
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

    // SiLU epilogue: acc[0] holds the gate rows, acc[1] the up rows they pair with.
    const int grow = m0 + wm * 16 + (lane & 15);
    if (grow < intermediate) {
#pragma unroll
        for (int ni = 0; ni < 2; ++ni) {
            const int col0 = wn * 32 + ni * 16 + (lane >> 4);
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const int col = col0 + 2 * j;
                if (col < t) {
                    out[static_cast<std::int64_t>(col) * intermediate + grow] =
                        __float2bfloat16_rn(silu(acc[0][ni][j]) * acc[1][ni][j]);
                }
            }
        }
    }
}

} // namespace ninfer::ops::detail

#endif // NINFER_HIP

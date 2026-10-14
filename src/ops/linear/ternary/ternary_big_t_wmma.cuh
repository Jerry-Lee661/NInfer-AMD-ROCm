#pragma once

// Ternary (PQ2_0) big-T WMMA kernel for prefill, v1. q4_q5_grouped_wmma.cuh's validated
// tiling (BM=64 weight rows x BN=128 tokens x BK=64 slab, cp_async double buffer, activations
// staged in LDS, unscaled codes on the tensor core, group scale folded per K tile) with the
// PQ2 byte_perm LUT decode swapped in (constants from bench/pq2_wmma_lut_proto.py, proven in
// ternary_small_t_wmma.cuh). One 128-weight group = two consecutive slabs; the group scale
// is duplicated into both slabs' staging so the accumulate stays per-slab uniform.
//
// v1 scope: PQ2 only, k % 128 == 0, t >= 17 (t 2..16 stays on the small-T kernel). Replaces
// the s8/wide_t rungs for prefill: those sit on the FMA-emulated mma chain (engine shapes
// measured 14.1 tok/s prefill, 2026-09-30 -- bench/decode-hip.md Phase 3).

#include "ops/common/math.cuh"
#include "ops/common/memory.cuh"
#include "ops/linear/q5/q5_rowsplit_gemm_wmma.cuh"

#include <cuda_bf16.h>

#include <cstdint>

#if defined(NINFER_HIP)

namespace ninfer::ops::detail {

namespace ternary_big_t_cfg {
constexpr int kBM      = 64;
constexpr int kBN      = 128;
constexpr int kBK      = 64;
constexpr int kStride  = kBK + 8;
constexpr int kKSub    = kBK / 16;
constexpr int kThreads = 256;
constexpr int kStages  = 2;
} // namespace ternary_big_t_cfg

struct TernaryBigTJob {
    const __nv_bfloat16* __restrict__ x;  // token-major [t, k] (ternary86 lineage convention)
    const std::uint8_t* __restrict__ codes;
    const std::uint8_t* __restrict__ scales;
    __nv_bfloat16* __restrict__ out;  // written at [col * out_stride + row]
    std::int32_t n;
    std::int32_t k;
    std::int32_t padded_k;
    std::int32_t out_stride;
};

template <int BN_, int THREADS_>
__global__ __launch_bounds__(THREADS_, 1) void ternary_big_t_wmma_kernel(
    TernaryBigTJob job, std::int32_t t) {
    using ninfer::ops::cp_async;
    using ninfer::ops::cp_commit;
    using ninfer::ops::cp_wait;
    using ninfer::ops::store_vec;
    constexpr int BM      = ternary_big_t_cfg::kBM;
    constexpr int BN      = BN_;  // C1: 64/128 变体(小 BN 换每 SIMD 2 CTA 驻留)
    constexpr int BK      = ternary_big_t_cfg::kBK;
    constexpr int ST      = ternary_big_t_cfg::kStride;
    constexpr int KSUB    = ternary_big_t_cfg::kKSub;
    constexpr int THREADS = THREADS_;
    constexpr int STAGES  = ternary_big_t_cfg::kStages;

    __shared__ __align__(16) __nv_bfloat16 s_x[STAGES][BN * ST];
    __shared__ __align__(16) std::uint8_t s_c[STAGES][BM * 16];  // PQ2: 16 B codes / row slab
    __shared__ __align__(16) std::uint16_t s_s[STAGES][BM];      // group scale, duplicated
    __shared__ __align__(16) __nv_bfloat16 s_w[BM * ST];

    const int tid  = static_cast<int>(threadIdx.x);
    const int lane = tid & 31;
    const int wm   = (tid >> 5) & 1;  // weight-row half (B operand)
    const int wn   = tid >> 6;        // token quarter (A operand)
    const int row0 = static_cast<int>(blockIdx.x) * BM;
    const int col0 = static_cast<int>(blockIdx.y) * BN;
    const int kg128 = job.padded_k >> 7;
    const int nkt   = job.padded_k / BK;

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
        const int k0    = kt * BK;
#pragma unroll 1
        for (int item = tid; item < BN * (BK / 8); item += THREADS) {
            const int tl       = item >> 3;
            const int kc       = item & 7;
            __nv_bfloat16* dst = &s_x[stage][tl * ST + kc * 8];
            if (col0 + tl < t && k0 + kc * 8 + 8 <= job.k) {
                cp_async<16>(dst, &job.x[static_cast<std::int64_t>(col0 + tl) * job.k + k0 + kc * 8]);
            } else {
                store_vec(dst, make_int4(0, 0, 0, 0));
            }
        }
        if (tid < BM) {
            const int row   = tid;
            const int srow  = row0 + row;
            const int group = kt >> 1;
            if (srow < job.n && group < kg128) {
                // Engine layout: SEPARATE codes and scales planes (per ternary86 SIMT kernels);
                // this slab takes its half of the 32-byte group codes.
                const std::int64_t gi = static_cast<std::int64_t>(srow) * kg128 + group;
                cp_async<16>(&s_c[stage][row * 16], job.codes + gi * 32 + (kt & 1) * 16);
                s_s[stage][row] = *reinterpret_cast<const std::uint16_t*>(job.scales + gi * 2);
            } else {
                store_vec(&s_c[stage][row * 16], make_int4(0, 0, 0, 0));
                s_s[stage][row] = 0;
            }
        }
    };

    // Decode:每线程解 4 码字节 → 16 bf16(四分之一 slab 行)。线程数泛化:64 行 × 4
    // 块 = 256 个单元,按 tid 步进(256T 各解 1 块,128T 各解 2 块;bn=64 变体首版
    // 写死 tid>>2 只覆盖 32 行,留下 25% 象限未写的档案 bug,2026-10-01 修)。
    auto decode_weights = [&](int stage) {
#pragma unroll
        for (int u = 0; u < (BM * 4) / THREADS; ++u) {
            const int idx = tid + u * THREADS;
            const int row = idx >> 2;
            const int sub = idx & 3;
            const std::uint8_t* src = &s_c[stage][row * 16 + sub * 4];
            __nv_bfloat16* dst      = &s_w[row * ST + sub * 16];
#pragma unroll
            for (int byte = 0; byte < 4; ++byte) {
                const std::uint32_t b   = src[byte];
                const std::uint32_t u_  = (b & 3u) | ((b << 4) & 0x300u);
                const std::uint32_t b_o = b >> 2;
                const std::uint32_t u_o = (b_o & 3u) | ((b_o << 4) & 0x300u);
                const std::uint32_t sel_e = (u_ | (u_ << 4)) + 0x4040u;
                const std::uint32_t sel_o = (u_o | (u_o << 4)) + 0x4040u;
                const std::uint32_t even  = __byte_perm(0x00800080u, 0x003F00BFu, sel_e);
                const std::uint32_t odd   = __byte_perm(0x00800080u, 0x003F00BFu, sel_o);
                const std::uint32_t r0    = __byte_perm(even, odd, 0x5410u);
                const std::uint32_t r1    = __byte_perm(even, odd, 0x7632u);
                std::uint16_t* d16 = reinterpret_cast<std::uint16_t*>(dst + byte * 4);
                d16[0] = static_cast<std::uint16_t>(r0 & 0xFFFFu);
                d16[1] = static_cast<std::uint16_t>(r0 >> 16);
                d16[2] = static_cast<std::uint16_t>(r1 & 0xFFFFu);
                d16[3] = static_cast<std::uint16_t>(r1 >> 16);
            }
        }
    };

#pragma unroll
    for (int s = 0; s < STAGES; ++s) {
        if (s < nkt) { stage_load(s, s); }
        cp_commit();
    }

    for (int it = 0; it < nkt; ++it) {
        const int stage = it % STAGES;
        cp_wait<STAGES - 1>();
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
        const int next = it + STAGES;
        if (next < nkt) { stage_load(stage, next); }
        cp_commit();
    }

#pragma unroll
    for (int mi = 0; mi < 2; ++mi) {
        const int row = row0 + mi * 32 + wm * 16 + (lane & 15);
        if (row < job.n) {
#pragma unroll
            for (int ni = 0; ni < 2; ++ni) {
                const int col_base = col0 + wn * 32 + ni * 16 + (lane >> 4);
#pragma unroll
                for (int e = 0; e < 8; ++e) {
                    const int col = col_base + 2 * e;
                    if (col < t) {
                        job.out[static_cast<std::int64_t>(col) * job.out_stride + row] =
                            __float2bfloat16_rn(acc[mi][ni][e]);
                    }
                }
            }
        }
    }
}

} // namespace ninfer::ops::detail

#endif // NINFER_HIP

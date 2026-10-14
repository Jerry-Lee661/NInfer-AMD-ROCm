#pragma once

// Grouped q4/q5 GEMM over RowSplitGroupedMmaJob lists, real WMMA variant of
// rowsplit_grouped_mma_kernel (RowSplitGroupedMmaCodec::Mixed). Shared by the two ops whose prefill
// route is that kernel: the GDN input projection and the attention input projection
// (amd-research.md §3.7.5b).
//
// One launch covers several (weight, output) jobs whose row tiles are flattened into blockIdx.x,
// with columns on blockIdx.y; each CTA resolves its job, stages that job's code plane (plus the q5
// high plane when the job is q5), and writes through the job's output row offset and stride.
//
// Numerics follow the two rules the earlier kernels paid for (amd-research.md §3.7.4/§3.7.5):
// the tensor core sees unscaled integer codes and the group scale folds into the fp32 accumulator
// once per K tile; the q5 code convention is (x ^ 0x10) - 0x10.

#include "ops/common/math.cuh"
#include "ops/common/memory.cuh"
#include "ops/common/rowsplit_grouped_mma.cuh"
#include "ops/linear/q4/q4_rowsplit_storage.cuh"
#include "ops/linear/q5/q5_rowsplit_gemm_wmma.cuh"

#include <cuda_bf16.h>

#include <cstdint>

#if defined(NINFER_HIP)

namespace ninfer::ops::detail {

namespace grouped_wmma_cfg {
constexpr int kBM      = 64;
constexpr int kBN      = 128;
constexpr int kBK      = 64;
constexpr int kStride  = kBK + 8;
constexpr int kKSub    = kBK / 16;
constexpr int kThreads = 256;
} // namespace grouped_wmma_cfg

template <int Stages>
__global__ __launch_bounds__(grouped_wmma_cfg::kThreads, 1) void q4_q5_grouped_wmma_kernel(
    const __nv_bfloat16* __restrict__ x, RowSplitGroupedMmaJob job0, RowSplitGroupedMmaJob job1,
    RowSplitGroupedMmaJob job2, RowSplitGroupedMmaJob job3, std::int32_t k, std::int32_t t,
    std::int32_t padded_k) {
    using ninfer::ops::cp_async;
    using ninfer::ops::cp_commit;
    using ninfer::ops::cp_wait;
    using ninfer::ops::store_vec;
    constexpr int BM      = grouped_wmma_cfg::kBM;
    constexpr int BN      = grouped_wmma_cfg::kBN;
    constexpr int BK      = grouped_wmma_cfg::kBK;
    constexpr int ST      = grouped_wmma_cfg::kStride;
    constexpr int KSUB    = grouped_wmma_cfg::kKSub;
    constexpr int THREADS = grouped_wmma_cfg::kThreads;

    // Row tiles of the jobs are flattened into blockIdx.x, in the order the launcher counted them.
    const int tiles0 = (job0.n + BM - 1) / BM;
    const int tiles1 = (job1.n + BM - 1) / BM;
    const int tiles2 = (job2.n + BM - 1) / BM;
    int tile         = static_cast<int>(blockIdx.x);
    RowSplitGroupedMmaJob job;
    if (tile < tiles0) {
        job = job0;
    } else if ((tile -= tiles0) < tiles1) {
        job = job1;
    } else if ((tile -= tiles1) < tiles2) {
        job = job2;
    } else {
        tile -= tiles2;
        job = job3;
    }

    __shared__ __align__(16) __nv_bfloat16 s_x[Stages][BN * ST];
    __shared__ __align__(16) __nv_bfloat16 s_w[BM * ST];
    __shared__ __align__(16) std::uint8_t s_c[Stages][BM * 32];
    __shared__ __align__(16) std::uint8_t s_h[Stages][BM * 8];
    __shared__ __align__(16) std::uint16_t s_s[Stages][BM];

    const int tid  = static_cast<int>(threadIdx.x);
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int wm   = warp & 1;
    const int wn   = warp >> 1;
    const int row0 = tile * BM;
    const int col0 = static_cast<int>(blockIdx.y) * BN;
    const int kg   = padded_k >> 6;
    const int nkt  = padded_k / BK;

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
        const int grp = k0 / 64;
#pragma unroll 1
        for (int item = tid; item < BN * (BK / 8); item += THREADS) {
            const int tl       = item >> 3;
            const int kc       = item & 7;
            __nv_bfloat16* dst = &s_x[stage][tl * ST + kc * 8];
            if (col0 + tl < t && k0 + kc * 8 + 8 <= k) {
                cp_async<16>(dst, &x[static_cast<std::int64_t>(col0 + tl) * k + k0 + kc * 8]);
            } else {
                store_vec(dst, make_int4(0, 0, 0, 0));
            }
        }
        if (tid < BM * 2) {
            const int row  = tid >> 1;
            const int half = tid & 1;
            auto* dst      = &s_c[stage][row * 32 + half * 16];
            if (row0 + row < job.n) {
                const std::int64_t gi = static_cast<std::int64_t>(row0 + row) * kg + grp;
                cp_async<16>(dst, &job.codes[gi * 32 + half * 16]);
            } else {
                store_vec(dst, make_int4(0, 0, 0, 0));
            }
        }
        if (job.q5 && tid >= THREADS - BM) {
            const int row = tid - (THREADS - BM);
            if (row0 + row < job.n) {
                const std::int64_t gi = static_cast<std::int64_t>(row0 + row) * kg + grp;
                cp_async<8>(&s_h[stage][row * 8], &job.high[gi * 8]);
            } else {
                store_vec(&s_h[stage][row * 8], static_cast<std::uint64_t>(0));
            }
        }
        if (tid < BM) {
            const int row       = tid;
            std::uint16_t value = 0;
            if (row0 + row < job.n) {
                const std::int64_t gi = static_cast<std::int64_t>(row0 + row) * kg + grp;
                value = *reinterpret_cast<const std::uint16_t*>(&job.scales[gi * 2]);
            }
            s_s[stage][row] = value;
        }
    };

    auto decode_weights = [&](int stage) {
        const int row           = tid >> 2;
        const int sub           = tid & 3;
        const std::uint8_t* src = &s_c[stage][row * 32 + sub * 8];
        const unsigned low      = *reinterpret_cast<const unsigned*>(src);
        const unsigned highw    = *reinterpret_cast<const unsigned*>(src + 4);
        unsigned lo[4];
        unsigned hi[4];
        if (job.q5) {
            const std::uint8_t* hb = &s_h[stage][row * 8];
            q5_wmma_decode_eight(low, hb[sub * 2], lo);
            q5_wmma_decode_eight(highw, hb[sub * 2 + 1], hi);
        } else {
            Q4MmaDecodeAtom::decode_eight(low, lo);
            Q4MmaDecodeAtom::decode_eight(highw, hi);
        }
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
        if (row < job.n) {
#pragma unroll
            for (int ni = 0; ni < 2; ++ni) {
                const int col_base = col0 + wn * 32 + ni * 16 + (lane >> 4);
#pragma unroll
                for (int e = 0; e < 8; ++e) {
                    const int col = col_base + 2 * e;
                    if (col < t) {
                        job.out[static_cast<std::int64_t>(col) * job.out_ld + job.out_row_offset +
                                row] = __float2bfloat16_rn(acc[mi][ni][e]);
                    }
                }
            }
        }
    }
}

} // namespace ninfer::ops::detail

#endif // NINFER_HIP

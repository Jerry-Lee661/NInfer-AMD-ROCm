#pragma once

// Narrow small-T WMMA variant of the grouped q4/q5 GEMM, v4. Lineage: v1/v2-early died to a
// fold-array overflow (sized by KSPLIT, indexed by warp); v2 (grouped construction verbatim,
// BM=32, k-split-4) went green and default-on but only -12% on gdn_proj: 80 serialized 64-k
// tiles, each paying two CTA barriers plus the full DRAM latency of a single-buffered stage,
// and staging the same 16 token rows into LDS in every CTA (~3.8 GB/s effective).
//
// v4 applies the ninfer-all t2_small_t_v2 lesson (ecosystem-survey.md §7): activations are NOT
// staged at all -- va loads 32-B slices straight from global (the token working set is <= 80 KB,
// permanently L2-resident and shared by every CTA), while only the weight codes (the real
// traffic, ~50 MB/step) stay double-buffered in LDS. The B side keeps the proven grouped
// decode-to-s_w + q5_wmma_load_b path; the fold is unchanged (indexed by warp, sized for 8).
//
// Layout contract (validated): builtin = (2nd arg) x (1st arg); word j of A pairs with word j of
// B half-wise (lo<->lo, hi<->hi); C register j of lane l is column 2j + (l>>4), row = l & 15.
// Numerics: unscaled integer codes into the tensor core (q4: (n^8)-8; q5: (x^16)-16 via the high
// plane), fp16 group scale folded into the fp32 accumulator once per 64-k tile.

#include "ops/common/math.cuh"
#include "ops/common/memory.cuh"
#include "ops/common/rowsplit_grouped_mma.cuh"
#include "ops/linear/q4/q4_rowsplit_storage.cuh"
#include "ops/linear/q5/q5_rowsplit_gemm_wmma.cuh"

#include <cuda_bf16.h>

#include <cstdint>

#if defined(NINFER_HIP)

namespace ninfer::ops::detail {

namespace q4_q5_small_t_cfg {
constexpr int kThreads = 256;      // 8 warps: 2 M-halves x 4 k-quarters
constexpr int kRows    = 32;       // rows per CTA tile (2 M-halves x 16)
#ifndef Q4Q5_ST_KSPLIT
#define Q4Q5_ST_KSPLIT 4
#endif
constexpr int kKSplit  = Q4Q5_ST_KSPLIT;  // k-quarters (warp >> 1); 1 = no split (bisection)
constexpr int kSlots   = 2;        // double-buffered code stages
} // namespace q4_q5_small_t_cfg

__global__ __launch_bounds__(q4_q5_small_t_cfg::kThreads, 1) void q4_q5_small_t_wmma_kernel(
    const __nv_bfloat16* __restrict__ x, RowSplitGroupedMmaJob job0, RowSplitGroupedMmaJob job1,
    RowSplitGroupedMmaJob job2, RowSplitGroupedMmaJob job3, std::int32_t k, std::int32_t t,
    std::int32_t padded_k) {
    using ninfer::ops::cp_async;
    using ninfer::ops::cp_commit;
    using ninfer::ops::cp_wait;
    constexpr int ROWS   = q4_q5_small_t_cfg::kRows;
    constexpr int KSPLIT = q4_q5_small_t_cfg::kKSplit;
    constexpr int SLOTS  = q4_q5_small_t_cfg::kSlots;
    constexpr int BK     = 64;
    constexpr int ST     = BK + 8;
    constexpr int KSUB   = BK / 16;

    __shared__ __align__(16) __nv_bfloat16 s_x[KSPLIT][SLOTS][16 * 72];   // coalesced cp_async staging
    __shared__ __align__(16) __nv_bfloat16 s_w[KSPLIT][ROWS][ST];
    __shared__ __align__(16) std::uint8_t s_c[KSPLIT][SLOTS][ROWS * 32];
    __shared__ __align__(16) std::uint8_t s_h[KSPLIT][SLOTS][ROWS * 8];
    __shared__ __align__(16) std::uint16_t s_s[KSPLIT][SLOTS][ROWS];
    __shared__ __align__(16) float partial[8 * 32 * 8];  // indexed by warp, not quarter

    // Row tiles of the jobs are flattened into blockIdx.x, in launcher-counted order.
    const int tiles0 = (job0.n + ROWS - 1) / ROWS;
    const int tiles1 = (job1.n + ROWS - 1) / ROWS;
    const int tiles2 = (job2.n + ROWS - 1) / ROWS;
    int tile = static_cast<int>(blockIdx.x);
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

    const int tid  = static_cast<int>(threadIdx.x);
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int wm   = warp & 1;                    // M half: rows 0..15 / 16..31 of the tile
    const int wks  = (KSPLIT == 1) ? 0 : (warp >> 1);  // k quarter (0..3)
    const int l64m = (KSPLIT == 1) ? (tid & 63) : (tid - 64 * wks);
    const int row0 = tile * ROWS + wm * 16;
    const int kg   = padded_k >> 6;               // 64-k scale groups per row
    const int tiles_per_wks = padded_k / BK / KSPLIT;

    // Per-k-quarter staging of the tile's 32 weight rows into slot (chunk & 1).
    auto stage = [&](int kt) {
        const int slot = kt & (SLOTS - 1);
        const int l64  = l64m;
        const int row  = l64 >> 1;            // 0..31, both M halves of the tile
        const int half = l64 & 1;
        const int srow = tile * ROWS + row;
        if (srow < job.n) {
            const std::int64_t gi = static_cast<std::int64_t>(srow) * kg + kt;
            cp_async<16>(&s_c[wks][slot][row * 32 + half * 16], job.codes + gi * 32 + half * 16);
            if (job.q5) { cp_async<8>(&s_h[wks][slot][row * 8], job.high + gi * 8); }
            s_s[wks][slot][row] = *reinterpret_cast<const std::uint16_t*>(job.scales + gi * 2);
        } else {
            s_s[wks][slot][row] = 0;
        }
    };

    // The grouped kernel's decode, verbatim math, from a slot into the tile's s_w.
    auto decode = [&](int slot) {
        const int l64 = l64m;
        const int row  = l64 >> 1;
        const int c16  = l64 & 1;
        const std::uint8_t* src = &s_c[wks][slot][row * 32 + c16 * 16];
        const unsigned w[4] = {*reinterpret_cast<const unsigned*>(src),
                               *reinterpret_cast<const unsigned*>(src + 4),
                               *reinterpret_cast<const unsigned*>(src + 8),
                               *reinterpret_cast<const unsigned*>(src + 12)};
        unsigned lo[4];
        unsigned hi[4];
        unsigned lo2[4];
        unsigned hi2[4];
        if (job.q5) {
            const std::uint8_t* hb = &s_h[wks][slot][row * 8];
            q5_wmma_decode_eight(w[0], hb[c16 * 4 + 0], lo);
            q5_wmma_decode_eight(w[1], hb[c16 * 4 + 1], hi);
            q5_wmma_decode_eight(w[2], hb[c16 * 4 + 2], lo2);
            q5_wmma_decode_eight(w[3], hb[c16 * 4 + 3], hi2);
        } else {
            Q4MmaDecodeAtom::decode_eight(w[0], lo);
            Q4MmaDecodeAtom::decode_eight(w[1], hi);
            Q4MmaDecodeAtom::decode_eight(w[2], lo2);
            Q4MmaDecodeAtom::decode_eight(w[3], hi2);
        }
        __nv_bfloat16* dst = &s_w[wks][row][c16 * 32];
        __nv_bfloat162* d2 = reinterpret_cast<__nv_bfloat162*>(dst);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            d2[i]     = *reinterpret_cast<const __nv_bfloat162*>(&lo[i]);
            d2[4 + i] = *reinterpret_cast<const __nv_bfloat162*>(&hi[i]);
        }
        __nv_bfloat162* d4 = reinterpret_cast<__nv_bfloat162*>(dst + 16);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            d4[i]     = *reinterpret_cast<const __nv_bfloat162*>(&lo2[i]);
            d4[4 + i] = *reinterpret_cast<const __nv_bfloat162*>(&hi2[i]);
        }
    };

    Q5WmmaC8F acc{};
    const int arow_row = lane & 15;               // token this lane reads (A operand row)
    const int brow_row = wm * 16 + (lane & 15);

    stage(wks * tiles_per_wks + 0);
    stage(wks * tiles_per_wks + 1);
    cp_commit();
    cp_commit();

    for (int it = 0; it < tiles_per_wks; ++it) {
        const int slot = it & (SLOTS - 1);
        cp_wait<1>();
        __syncthreads();
        decode(slot);
        __syncthreads();

        const __nv_bfloat16* brow = s_w[wks][brow_row];
        const float scale = __half2float(__ushort_as_half(s_s[wks][slot][brow_row]));
        // Two independent accumulator chains: the ISA guide requires a NOP (or an independent
        // op) between back-to-back dependent WMMA whose D feeds the next C -- one chain stalls
        // on its own latency (s_delay_alu walls in the ISA dump), two chains interleave.
        // va comes from the coalesced double-buffered s_x (v4's per-lane global direct read
        // issued 32 scattered 16-B requests per ks per warp, 8x duplicated across warps).
        Q5WmmaC8F part0{};
        Q5WmmaC8F part1{};
#pragma unroll
        for (int ksp = 0; ksp < KSUB / 2; ++ksp) {
            // Activations straight from global (L2-resident, shared across CTAs; t2_small_t_v2's
            // move): tokens >= t never read, their fragment stays zero.
            Q5WmmaA8 va0{};
            Q5WmmaA8 va1{};
            if (arow_row < t) {
                const std::int64_t kbase = static_cast<std::int64_t>(wks * tiles_per_wks + it) * BK;
                va0 = q5_wmma_load_a(&x[static_cast<std::int64_t>(arow_row) * k + kbase +
                                        (2 * ksp) * 16]);
                va1 = q5_wmma_load_a(&x[static_cast<std::int64_t>(arow_row) * k + kbase +
                                        (2 * ksp + 1) * 16]);
            }
            Q5WmmaB16 vb0 = q5_wmma_load_b(brow + (2 * ksp) * 16);
            Q5WmmaB16 vb1 = q5_wmma_load_b(brow + (2 * ksp + 1) * 16);
            part0 = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(va0, vb0, part0);
            part1 = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(va1, vb1, part1);
        }
#pragma unroll
        for (int e = 0; e < 8; ++e) { acc[e] = fmaf(scale, part0[e] + part1[e], acc[e]); }

        __syncthreads();
        if (it + 2 < tiles_per_wks) { stage((wks * tiles_per_wks + it) + 2); }
        cp_commit();
    }

    // k-quarter fold: warps 0..3 hold wm=0 quarters, warps 4..7 hold wm=1.
    __syncthreads();
    if (wks != 0) {
#pragma unroll
        for (int j = 0; j < 8; ++j) { partial[(warp * 32 + lane) * 8 + j] = acc[j]; }
    }
    __syncthreads();
    if (wks == 0) {
#pragma unroll
        for (int q = 1; q < KSPLIT; ++q) {
#pragma unroll
            for (int j = 0; j < 8; ++j) { acc[j] += partial[((warp + 2 * q) * 32 + lane) * 8 + j]; }
        }
        const int row   = row0 + arow_row;
        const int cpar  = lane >> 4;
        if (row < job.n) {
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const int col = 2 * j + cpar;
                if (col < t) {
                    job.out[static_cast<std::int64_t>(col) * job.out_ld + job.out_row_offset + row] =
                        __float2bfloat16_rn(acc[j]);
                }
            }
        }
    }
}

} // namespace ninfer::ops::detail

#endif // NINFER_HIP

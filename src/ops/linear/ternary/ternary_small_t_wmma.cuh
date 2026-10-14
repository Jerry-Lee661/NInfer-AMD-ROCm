#pragma once

// Ternary (PQ2_0) small-T WMMA kernel, v1. Mirror of q4_q5_small_t_wmma.cuh's validated
// skeleton (BM=32, 8 warps = 2 row-halves x 4 k-quarters, codes double-buffered, activations
// global-direct, dual accumulators, per-warp fold) with the ternary decode swapped in:
//
//   PQ2_0: 2-bit slots, 4/byte, code-1 = value in {-1,0,+1}; one fp16 scale per 128-weight
//   group. s_w holds UNSCALED bf16 via the byte_perm LUT (constants derived by
//   bench/pq2_wmma_lut_proto.py, all-green: kLutLow 0x00800080, kLutHigh 0x003F00BF,
//   spread +0x4040, interleave 0x5410/0x7632); the group scale folds per slab at
//   accumulate time exactly like q4 (each slab reads its group's duplicated scale).
//
// v1 scope: PQ2 only, k % 128 == 0, cols 2..16. PTQ1 (base-3 trit walk) stays on the SIMT path.

#include "ops/common/math.cuh"
#include "ops/common/memory.cuh"
#include "ops/linear/q5/q5_rowsplit_gemm_wmma.cuh"
#include "ops/linear/ternary/ternary_rowsplit_storage.cuh"

#include <cuda_bf16.h>

#include <cstdint>

#if defined(NINFER_HIP)

namespace ninfer::ops::detail {

namespace ternary_small_t_cfg {
constexpr int kThreads = 256;  // 8 warps: 2 M-halves x 4 k-quarters
constexpr int kRows    = 32;   // rows per CTA tile
constexpr int kSlots   = 2;    // double-buffered code stages
}  // namespace ternary_small_t_cfg

struct TernarySmallTJob {
    const __nv_bfloat16* __restrict__ x;  // token-major [t, k] (ternary86 lineage convention)
    const std::uint8_t* __restrict__ codes;
    const std::uint8_t* __restrict__ scales;
    __nv_bfloat16* __restrict__ out;  // [t, out_stride] written at [col * out_stride + row]
    std::int32_t n;
    std::int32_t k;
    std::int32_t padded_k;
    std::int32_t out_stride;  // verify-fused qkv buffers write with stride != n (10240 layouts)
};

__global__ __launch_bounds__(ternary_small_t_cfg::kThreads, 1) void ternary_small_t_wmma_kernel(
    TernarySmallTJob job, std::int32_t t, float* diag) {
    using ninfer::ops::cp_async;
    using ninfer::ops::cp_commit;
    using ninfer::ops::cp_wait;
    constexpr int ROWS   = ternary_small_t_cfg::kRows;
    constexpr int SLOTS  = ternary_small_t_cfg::kSlots;
    constexpr int BK     = 64;  // 64-k slab; one 128-weight group = two consecutive slabs
    constexpr int ST     = BK + 8;
    constexpr int KSUB   = BK / 16;

    // Per-k-quarter planes: each wks warp-pair stages ITS OWN slab data (q4 skeleton's
    // [KSPLIT] dimension — flattening it lets quarters clobber each other).
    __shared__ __align__(16) __nv_bfloat16 s_w[4][SLOTS][ROWS][ST];
    __shared__ __align__(16) std::uint8_t s_c[4][SLOTS][ROWS * 16];   // PQ2: 16 B codes / row slab
    __shared__ __align__(16) std::uint16_t s_s[4][SLOTS][ROWS];       // group scale, duplicated

    const int tid  = static_cast<int>(threadIdx.x);
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int wm   = warp & 1;   // M half: rows 0..15 / 16..31 of the tile
    const int wks  = warp >> 1;  // k quarter (0..3)
    const int l64  = tid & 63;   // 64 threads per k-quarter group
    const int row0 = static_cast<int>(blockIdx.x) * ROWS + wm * 16;
    const int kg128 = job.padded_k >> 7;             // 128-k groups per row
    const int tiles_per_wks = job.padded_k / BK / 4; // 64-k slabs per warp quarter

    // Per-slab staging: 64 threads stage 32 rows (thread t -> row t>>1, half t&1).
    // PQ2 codes for one row slab = 16 B = one cp_async<16>; the group scale (2 B) is
    // duplicated into both slabs of the group so accumulate stays per-slab uniform.
    auto stage = [&](int kt) {
        const int slot  = kt & (SLOTS - 1);
        const int row   = l64 >> 1;
        const int half  = l64 & 1;
        const int srow  = static_cast<int>(blockIdx.x) * ROWS + row;
        const int group = kt >> 1;
        if (half == 0 && (srow >= job.n || group >= kg128)) { s_s[wks][slot][row] = 0; }
        if (srow < job.n && group < kg128) {
            // Engine layout: SEPARATE codes and scales planes (per ternary86 SIMT kernels).
            const std::int64_t gi = static_cast<std::int64_t>(srow) * kg128 + group;
            cp_async<8>(&s_c[wks][slot][row * 16 + half * 8],
                        job.codes + gi * 32 + (kt & 1) * 16 + half * 8);
            if (half == 0) {
                s_s[wks][slot][row] =
                    *reinterpret_cast<const std::uint16_t*>(job.scales + gi * 2);
            }
        }
    };

    // Decode: one thread decodes its 8 code bytes -> 32 bf16 into s_w (half row).
    // LUT math from bench/pq2_wmma_lut_proto.py (all-green).
    auto decode = [&](int slot) {
        const int row  = l64 >> 1;
        const int half = l64 & 1;
        const std::uint8_t* src = &s_c[wks][slot][row * 16 + half * 8];
        __nv_bfloat16* dst = &s_w[wks][slot][row][half * 32];
#pragma unroll
        for (int byte = 0; byte < 8; ++byte) {
            const std::uint32_t b = src[byte];
            const std::uint32_t u   = (b & 3u) | ((b << 4) & 0x300u);
            const std::uint32_t b_o = b >> 2;
            const std::uint32_t u_o = (b_o & 3u) | ((b_o << 4) & 0x300u);
            const std::uint32_t sel_e = (u | (u << 4)) + 0x4040u;
            const std::uint32_t sel_o = (u_o | (u_o << 4)) + 0x4040u;
            const std::uint32_t even = __byte_perm(0x00800080u, 0x003F00BFu, sel_e);
            const std::uint32_t odd  = __byte_perm(0x00800080u, 0x003F00BFu, sel_o);
            const std::uint32_t r0   = __byte_perm(even, odd, 0x5410u);
            const std::uint32_t r1   = __byte_perm(even, odd, 0x7632u);
            std::uint16_t* d16 = reinterpret_cast<std::uint16_t*>(dst + byte * 4);
            d16[0] = static_cast<std::uint16_t>(r0 & 0xFFFFu);
            d16[1] = static_cast<std::uint16_t>(r0 >> 16);
            d16[2] = static_cast<std::uint16_t>(r1 & 0xFFFFu);
            d16[3] = static_cast<std::uint16_t>(r1 >> 16);
        }
    };

    using C8F = float[8];
    Q5WmmaC8F acc{};
    const int arow_row = lane & 15;              // token this lane reads (A operand row)
    const int brow_row = wm * 16 + (lane & 15);  // weight row this lane's B fragment covers

    stage(wks * tiles_per_wks + 0);
    stage(wks * tiles_per_wks + 1);
    cp_commit();
    cp_commit();

#pragma unroll 2
    for (int it = 0; it < tiles_per_wks; ++it) {
        const int slot = it & (SLOTS - 1);
        cp_wait<1>();
        __syncthreads();
        decode(slot);
        __syncthreads();

        const __nv_bfloat16* brow = s_w[wks][slot][brow_row];
        const float scale = __half2float(__ushort_as_half(s_s[wks][slot][brow_row]));
        // Dual accumulator chains (ISA 7.9: independent ops between dependent WMMA).
        Q5WmmaC8F part0{};
        Q5WmmaC8F part1{};
#pragma unroll
        for (int ksp = 0; ksp < KSUB / 2; ++ksp) {
            Q5WmmaA8 va0{};
            Q5WmmaA8 va1{};
            if (arow_row < t) {
                const std::int64_t kbase = static_cast<std::int64_t>(wks * tiles_per_wks + it) * BK;
                va0 = q5_wmma_load_a(&job.x[static_cast<std::int64_t>(arow_row) * job.k + kbase +
                                             (2 * ksp) * 16]);
                va1 = q5_wmma_load_a(&job.x[static_cast<std::int64_t>(arow_row) * job.k + kbase +
                                             (2 * ksp + 1) * 16]);
            }
            Q5WmmaB16 vb0 = q5_wmma_load_b(brow + (2 * ksp) * 16);
            Q5WmmaB16 vb1 = q5_wmma_load_b(brow + (2 * ksp + 1) * 16);
            part0 = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(va0, vb0, part0);
            part1 = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(va1, vb1, part1);
        }
#pragma unroll
        for (int e = 0; e < 8; ++e) { acc[e] = fmaf(scale, part0[e] + part1[e], acc[e]); }
        (void)diag;

        __syncthreads();
        if (it + 2 < tiles_per_wks) { stage((wks * tiles_per_wks + it) + 2); }
        cp_commit();
    }

    // k-quarter fold: warps 0..3 hold wm=0 quarters, warps 4..7 hold wm=1.
    __shared__ __align__(16) float partial[8 * 32 * 8];  // indexed by warp
    __syncthreads();
    if (wks != 0) {
#pragma unroll
        for (int j = 0; j < 8; ++j) { partial[(warp * 32 + lane) * 8 + j] = acc[j]; }
    }
    __syncthreads();
    if (wks == 0) {
#pragma unroll
        for (int q = 1; q < 4; ++q) {
#pragma unroll
            for (int j = 0; j < 8; ++j) { acc[j] += partial[((warp + 2 * q) * 32 + lane) * 8 + j]; }
        }
        const int row  = row0 + arow_row;
        const int cpar = lane >> 4;
        if (row < job.n) {
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const int col = 2 * j + cpar;
                if (col < t) {
                    job.out[static_cast<std::int64_t>(col) * job.out_stride + row] =
                        __float2bfloat16_rn(acc[j]);
                }
            }
        }
    }
}

}  // namespace ninfer::ops::detail

#endif  // NINFER_HIP

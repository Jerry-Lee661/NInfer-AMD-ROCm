#pragma once

// SwiGLU small-T narrow WMMA kernel (T <= 16): the gdn v5-final pattern applied to the q4
// swiglu op. One CTA tile = 32 output rows; its 64 weight rows are 32 gate rows (weight rows
// [tile*32, tile*32+32)) plus their 32 matching up rows ([out_rows + tile*32, +32); the up half
// lives at weight rows [out_rows, 2*out_rows)). Eight warps split as 2 row-halves x 4
// k-quarters; each warp accumulates the gate chain and the up chain of its own 16 rows -- two
// independent WMMA chains per k16 step, which satisfies the ISA guide's NOP rule for free --
// and the silu(gate)*up epilogue fuses into the store after the k-quarter fold.
//
// Numerics: unscaled q4 codes into the tensor core, fp16 group scale folded into fp32 once
// per 64-k tile (the two rules the prefill swiglu kernel paid for).

#include "ops/common/math.cuh"
#include "ops/common/memory.cuh"
#include "ops/linear/q4/q4_rowsplit_storage.cuh"
#include "ops/linear/q5/q5_rowsplit_gemm_wmma.cuh"

#include <cuda_bf16.h>

#include <cstdint>

#if defined(NINFER_HIP)

namespace ninfer::ops::detail {

namespace q4_swiglu_small_t_cfg {
constexpr int kThreads = 256;      // 8 warps: 2 row-halves x 4 k-quarters
constexpr int kOutRows = 32;       // output rows per CTA tile
constexpr int kKSplit  = 4;        // k-quarters (warp >> 1)
constexpr int kSlots   = 2;        // double-buffered code stages
} // namespace q4_swiglu_small_t_cfg

// Engine-side trace flag (set by the launcher from NINFER_SWIGLU_NARROW_TRACE): tile 0 /
// lane 2 prints its per-warp pre-fold accumulators (the C register of (row 2, col 0)).
__device__ int q4_swiglu_trace = 0;

__global__ __launch_bounds__(q4_swiglu_small_t_cfg::kThreads, 1) void q4_swiglu_small_t_wmma_kernel(
    const __nv_bfloat16* __restrict__ x, const std::uint8_t* __restrict__ codes,
    const std::uint8_t* __restrict__ scales, __nv_bfloat16* __restrict__ out, std::int32_t k,
    std::int32_t t, std::int32_t padded_k, std::int32_t out_rows) {
    using ninfer::ops::cp_async;
    using ninfer::ops::cp_commit;
    using ninfer::ops::cp_wait;
    constexpr int OROWS  = q4_swiglu_small_t_cfg::kOutRows;
    constexpr int KSPLIT = q4_swiglu_small_t_cfg::kKSplit;
    constexpr int SLOTS  = q4_swiglu_small_t_cfg::kSlots;
    constexpr int BK     = 64;
    constexpr int KSUB   = BK / 16;
    constexpr int WROWS  = OROWS * 2;   // staged weight rows per k-quarter (gate 32 + up 32)

    // The fold publishes 8 warps x 32 lanes x (gate + up) x 8 floats = 16 KB, which on top of
    // the other staging would break the 64 KB LDS budget; it runs after the main loop where the
    // code staging is dead, so it borrows its storage. (A standalone partial[KSPLIT*2*32*8] was
    // 32 B short: its tail lane wrote past LDS, probe-invisible in a lone workgroup but
    // garbage under engine co-residency.)
    __shared__ __align__(16) union {
        std::uint8_t c[KSPLIT][SLOTS][WROWS * 32];
        float partial[8 * 32 * 16];
    } s_stage;
    __shared__ __align__(16) __nv_bfloat16 s_w[KSPLIT][WROWS][BK + 8];
    __shared__ __align__(16) std::uint16_t s_s[KSPLIT][SLOTS][WROWS];

    const int tid  = static_cast<int>(threadIdx.x);
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int wm   = warp & 1;                    // row half: 0..15 / 16..31 of the tile
    const int wks  = warp >> 1;                   // k quarter (0..3)
    const int l64  = tid - 64 * wks;              // 0..63 within this quarter's two warps
    const int tile = static_cast<int>(blockIdx.x);
    const int row0 = tile * OROWS + wm * 16;
    const int kg   = padded_k >> 6;
    const int tiles_per_wks = padded_k / BK / KSPLIT;

    // Stage one 64-k tile of this quarter's 64 weight rows (32 gate + 32 up). The slot is the
    // CALLER's iteration parity, not kt & 1: with odd tiles_per_wks the kt parity decouples
    // from the iteration parity and the loop would decode a never-staged slot.
    auto stage = [&](int kt, int slot) {
        const int r    = l64;                 // 0..63: 0..31 gate, 32..63 up
        const int r32  = r & 31;
        const int wrow = (r < 32 ? tile * OROWS + r : out_rows + tile * OROWS + r32);
        const std::int64_t gi = static_cast<std::int64_t>(wrow) * kg + kt;
        cp_async<16>(&s_stage.c[wks][slot][r * 32 + 0], codes + gi * 32 + 0);
        cp_async<16>(&s_stage.c[wks][slot][r * 32 + 16], codes + gi * 32 + 16);
        s_s[wks][slot][r] = *reinterpret_cast<const std::uint16_t*>(scales + gi * 2);
    };

    // The grouped kernel's decode, verbatim math, from a slot into this quarter's s_w. 64 rows
    // with 64 threads = one thread per row (the gdn kernel has two threads per 32 rows, so the
    // lane-parity half-split does NOT carry over: each thread here must decode BOTH 16-byte
    // halves or half of every s_w row is stale LDS -- the probe's random-u16 scales hid this
    // behind NaN/Inf references and 0.4% relative tolerance; REALSCALE data exposed it).
    auto decode = [&](int slot) {
        const int r = l64;
#pragma unroll
        for (int c16 = 0; c16 < 2; ++c16) {
            const std::uint8_t* src = &s_stage.c[wks][slot][r * 32 + c16 * 16];
            const unsigned w[4] = {*reinterpret_cast<const unsigned*>(src),
                                   *reinterpret_cast<const unsigned*>(src + 4),
                                   *reinterpret_cast<const unsigned*>(src + 8),
                                   *reinterpret_cast<const unsigned*>(src + 12)};
            unsigned lo[4];
            unsigned hi[4];
            unsigned lo2[4];
            unsigned hi2[4];
            Q4MmaDecodeAtom::decode_eight(w[0], lo);
            Q4MmaDecodeAtom::decode_eight(w[1], hi);
            Q4MmaDecodeAtom::decode_eight(w[2], lo2);
            Q4MmaDecodeAtom::decode_eight(w[3], hi2);
            __nv_bfloat16* dst = &s_w[wks][r][c16 * 32];
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
        }
    };

    Q5WmmaC8F acc_g{};
    Q5WmmaC8F acc_u{};
    const int arow_row = lane & 15;               // token this lane reads (A operand row)
    const int vb_g     = wm * 16 + (lane & 15);              // gate row within the tile
    const int vb_u     = OROWS + wm * 16 + (lane & 15); // up row within the tile (gate rows 0..31, up 32..63)

    stage(wks * tiles_per_wks + 0, 0);
    if (tiles_per_wks >= 2) { stage(wks * tiles_per_wks + 1, 1); }
    cp_commit();
    cp_commit();

    for (int it = 0; it < tiles_per_wks; ++it) {
        const int slot = it & (SLOTS - 1);
        cp_wait<1>();
        __syncthreads();
        decode(slot);
        __syncthreads();

        const __nv_bfloat16* bg = s_w[wks][vb_g];
        const __nv_bfloat16* bu = s_w[wks][vb_u];
        const float sg = __half2float(__ushort_as_half(s_s[wks][slot][vb_g]));
        const float su = __half2float(__ushort_as_half(s_s[wks][slot][vb_u]));
        Q5WmmaC8F part_g{};
        Q5WmmaC8F part_u{};
#pragma unroll
        for (int ks = 0; ks < KSUB; ++ks) {
            // Activations straight from global (L2-resident, shared across CTAs): tokens >= t
            // never read, their fragment stays zero.
            Q5WmmaA8 va{};
            if (arow_row < t) {
                va = q5_wmma_load_a(
                    &x[static_cast<std::int64_t>(arow_row) * k +
                       static_cast<std::int64_t>(wks * tiles_per_wks + it) * BK + ks * 16]);
            }
            Q5WmmaB16 vbg = q5_wmma_load_b(bg + ks * 16);
            Q5WmmaB16 vbu = q5_wmma_load_b(bu + ks * 16);
            part_g = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(va, vbg, part_g);
            part_u = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(va, vbu, part_u);
        }
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            acc_g[e] = fmaf(sg, part_g[e], acc_g[e]);
            acc_u[e] = fmaf(su, part_u[e], acc_u[e]);
        }

        __syncthreads();
        if (it + 2 < tiles_per_wks) { stage((wks * tiles_per_wks + it) + 2, slot); }
        cp_commit();
    }

    if (q4_swiglu_trace && tile == 0 && lane < 4) {
        printf("swt warp=%d wks=%d wm=%d lane=%d accg=%+.6e %+.6e accu=%+.6e %+.6e\n", warp, wks,
               wm, lane, static_cast<double>(acc_g[0]), static_cast<double>(acc_g[1]),
               static_cast<double>(acc_u[0]), static_cast<double>(acc_u[1]));
    }

    // k-quarter fold: warps 0..3 hold wm=0 quarters, warps 4..7 hold wm=1; gate/up separate.
    // Slot stride is 16 floats per (warp, lane) -- gate at +0, up at +8. A packed +8 layout
    // collides warp W's up slots with warp W+1's gate slots (lane 31 vs lane 0) and the fold
    // sums garbage; 8 warps x 32 lanes x 16 floats = exactly the borrowed staging's 4096.
    __syncthreads();
    if (wks != 0) {
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            s_stage.partial[(warp * 32 + lane) * 16 + j] = acc_g[j];
            s_stage.partial[(warp * 32 + lane) * 16 + 8 + j] = acc_u[j];
        }
    }
    __syncthreads();
    if (wks == 0) {
#pragma unroll
        for (int q = 1; q < KSPLIT; ++q) {
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                acc_g[j] += s_stage.partial[((warp + 2 * q) * 32 + lane) * 16 + j];
                acc_u[j] += s_stage.partial[((warp + 2 * q) * 32 + lane) * 16 + 8 + j];
            }
        }
        if (q4_swiglu_trace && tile == 0 && lane < 4) {
            printf("swt final lane=%d accg=%+.6e %+.6e accu=%+.6e %+.6e\n", lane,
                   static_cast<double>(acc_g[0]), static_cast<double>(acc_g[1]),
                   static_cast<double>(acc_u[0]), static_cast<double>(acc_u[1]));
        }
        const int row   = tile * OROWS + wm * 16 + arow_row;
        const int cpar  = lane >> 4;
        if (row < out_rows) {
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const int col = 2 * j + cpar;
                if (col < t) {
                    const float g = acc_g[j];
                    const float v = g / (1.0f + __expf(-g)) * acc_u[j];
                    out[static_cast<std::int64_t>(col) * out_rows + row] =
                        __float2bfloat16_rn(v);
                }
            }
        }
    }
}

} // namespace ninfer::ops::detail

#endif // NINFER_HIP

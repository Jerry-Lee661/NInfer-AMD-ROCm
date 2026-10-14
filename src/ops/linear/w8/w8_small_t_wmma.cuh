#pragma once

// WMMA variant of the W8 small-T kernel, in its own header so that edits to it do not
// ripple through the eight other TUs that include w8_small_t_mma.cuh.

#include "ops/linear/w8/w8_small_t_mma.cuh"

namespace ninfer::ops::detail {

// WMMA variant of the same schedule: the mma_bf16 emulation inner loop is replaced by
// the hardware v_wmma_f32_16x16x16_bf16_w32 builtin (layout contract validated on
// gfx1100 by tools/w8_small_t_wmma_probe.cu: builtin = (2nd arg) x (1st arg), C register
// j of lane l is column 2j + (l>>4)). Same staging, scales, k-split reduction, and
// epilogue as the emulated kernel; only the fragment packing differs. Requires
// kTileCols == 16 (one WMMA N-tile) and kTileK a multiple of 32 (two scale groups per
// warp slice).
template <class Geometry, int ActiveCols, class Schedule, class Output, bool XDirect = false>
__global__ __launch_bounds__(Schedule::kThreads, Schedule::kMinBlocksPerSm) void
w8_small_t_wmma_kernel(const __nv_bfloat16* __restrict__ x,
                       const std::uint8_t* __restrict__ codes,
                       const std::uint8_t* __restrict__ scales, Output output) {
    static_assert(Schedule::kTileTokens == 16, "WMMA path packs one 16-column N-tile");
    using A8   = unsigned __attribute__((ext_vector_type(8)));
    using B16S = short __attribute__((ext_vector_type(16)));
    using C8F  = float __attribute__((ext_vector_type(8)));
    constexpr int kHidden     = Geometry::kInputRows;
    constexpr int kTileK      = Schedule::kTileKPerWarp;
    constexpr int kWarps      = Schedule::kKWarps;
    constexpr int kGroupK     = Schedule::kGroupK;
    constexpr int kGroups     = kHidden / kGroupK;
    constexpr int kRowsPerCta = Schedule::kRowsPerCta;
    constexpr int kTileCols   = Schedule::kTileTokens;
    constexpr int kNt         = 1;  // one WMMA 16-column tile per lane set
    constexpr unsigned long long kMask = 0xffffffffffffffffULL;

    union SharedStorage {
        struct {
            std::uint8_t codes[kRowsPerCta][kGroupK];
            __nv_bfloat16 activations[kWarps][kTileCols * kTileK];
        } staging;
        float partial[kWarps * 32 * 8];
    };
    __shared__ __align__(16) SharedStorage shared;
    auto& code_shared = shared.staging.codes;
    auto& b_shared    = shared.staging.activations;

    const int tid     = static_cast<int>(threadIdx.x);
    const int warp    = tid >> 5;
    const int lane    = tid & 31;
    const int k_split = warp;

    const int cta_row0 = static_cast<int>(blockIdx.x) * kRowsPerCta;

    const auto stage_x = [&](int group_k0) {
        for (int item = lane; item < ActiveCols * (kTileK / 8); item += 32) {
            const int col = item / (kTileK / 8);
            const int k8  = item - col * (kTileK / 8);
            cp_async<16, Schedule::kActivationCache>(
                &b_shared[warp][col * kTileK + w8_small_t_swizzle_64(col, k8 * 8)],
                &x[static_cast<std::int64_t>(col) * kHidden + group_k0 + warp * kTileK +
                   k8 * 8]);
        }
    };
    const auto stage_codes = [&](int group_k0) {
        for (int row_item = 0; row_item < kRowsPerCta / kWarps; ++row_item) {
            const int row = warp * (kRowsPerCta / kWarps) + row_item;
            for (int chunk = lane; chunk < kGroupK / 16; chunk += 32) {
                const int swizzled_chunk = chunk ^ (row & 7);
                cp_async<16, Schedule::kWeightCache>(
                    &code_shared[row][swizzled_chunk * 16],
                    codes + static_cast<std::int64_t>(cta_row0 + row) * kHidden + group_k0 +
                        chunk * 16);
            }
        }
    };

    const int warp_koff = k_split * kTileK;
    C8F acc{};
    stage_codes(0);
    if constexpr (!XDirect) { stage_x(0); }
    cp_commit();
    cp_wait<0>();
    __syncthreads();

#pragma unroll 1
    for (int group_index = 0; group_index < kGroups; ++group_index) {
        const int group_k0 = group_index * kGroupK;
#pragma unroll
        for (int half = 0; half < 2; ++half) {
            C8F half_acc{};
#pragma unroll
            for (int s = 0; s < 2; ++s) {
                const int k_local = half * 32 + s * 16;
                // B operand: this lane's weight row (m = lane & 15), 16 consecutive codes.
                const int row_b   = lane & 15;
                const int chunk   = (warp_koff + k_local) >> 4;
                const int offset  = (chunk ^ (row_b & 7)) * 16;
                const uint4 codes16 =
                    load_vec<uint4>(&code_shared[row_b][offset]);  // 16 codes, in-k order
                B16S vb;
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const unsigned pair =
                        w8_small_t_bf16_pair_from_s8((codes16[j / 2] >> (16 * (j % 2))) &
                                                     0xFFFFu);
                    vb[2 * j]     = static_cast<short>(pair & 0xFFFFu);
                    vb[2 * j + 1] = static_cast<short>(pair >> 16);
                }
                // A operand: activations X[k][n], k-major.
                A8 va;
                if constexpr (XDirect) {
                    // v4 配方（gdn 窄核同款）：x 工作集（ActiveCols×kHidden×2B ≤ 80KB）
                    // 常驻 L2 且全 CTA 共享，直读全局省掉每 CTA 的 LDS 暂存与一次同步。
                    // 同一 (k,k+1) 对落在一个 4-B 对齐的 u32 里。
                    const int n = lane & 15;
                    const std::int64_t k0 = static_cast<std::int64_t>(n) * kHidden +
                                            group_k0 + warp_koff + k_local;
#pragma unroll
                    for (int j = 0; j < 8; ++j) {
                        va[j] = *reinterpret_cast<const unsigned*>(x + k0 + 2 * j);
                    }
                } else {
#pragma unroll
                    for (int j = 0; j < 8; ++j) {
                        const int n = lane & 15;
                        const int base =
                            n * kTileK + w8_small_t_swizzle_64(n, k_local + 2 * j);
                        const unsigned short lo =
                            __bfloat16_as_ushort(b_shared[k_split][base]);
                        const unsigned short hi =
                            __bfloat16_as_ushort(b_shared[k_split][base + 1]);
                        va[j] = static_cast<unsigned>(lo) | (static_cast<unsigned>(hi) << 16);
                    }
                }
                half_acc = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(va, vb, half_acc);
            }
            // Scale: this lane's row, the scale group pair covering the warp k-slice.
            const int row_scale = lane & 15;
            const unsigned pair = *reinterpret_cast<const unsigned*>(
                scales + (static_cast<std::int64_t>(cta_row0 + row_scale) *
                              Geometry::kGroupsPerRow +
                          group_k0 / 32 + warp_koff / 32) * 2);
            const unsigned bits = half == 0 ? pair & 0xffffu : pair >> 16;
            const float scale   = __half2float(__ushort_as_half(bits));
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                acc[j] = fmaf(half_acc[j], scale, acc[j]);
            }
        }

        if (group_index + 1 < kGroups) {
            __syncthreads();
            stage_codes(group_k0 + kGroupK);
            if constexpr (!XDirect) { stage_x(group_k0 + kGroupK); }
            cp_commit();
            cp_wait<0>();
            __syncthreads();
        }
    }

    // k-split reduction: warp-pair exchange, then warp 0 folds the remaining halves.
    __syncthreads();
    float* partial = shared.partial;
    if ((k_split & 1) != 0) {
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            partial[(warp * 32 + lane) * 8 + j] = acc[j];
        }
    }
    __syncthreads();
    if ((k_split & 1) == 0) {
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            acc[j] += partial[((warp + 1) * 32 + lane) * 8 + j];
            if (k_split != 0) { partial[(warp * 32 + lane) * 8 + j] = acc[j]; }
        }
    }
    __syncthreads();
    if (k_split == 0) {
        const W8OutputTile output_tile = output.tile(cta_row0);
#pragma unroll
        for (int split = 2; split < kWarps; split += 2) {
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                acc[j] += partial[(split * 32 + lane) * 8 + j];
            }
        }
        const int row  = cta_row0 + (lane & 15);
        const int cpar = lane >> 4;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const int col = 2 * j + cpar;
            if (col < ActiveCols) {
                __nv_bfloat16* destination = output_tile.at(row, col);
                *destination = __float2bfloat16_rn(acc[j]);
            }
        }
    }
}

} // namespace ninfer::ops::detail

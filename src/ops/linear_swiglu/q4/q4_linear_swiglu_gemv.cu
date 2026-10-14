#include "ops/linear_swiglu/q4/q4_linear_swiglu_kernels.h"

#include "ops/common/math.cuh"
#include "ops/common/memory.cuh"
#include "ops/common/warp.cuh"
#include "core/device.h" // CUDA_CHECK
#include "ops/linear/q4/q4_rowsplit_storage.cuh"
#include "ops/linear/q4/q4_small_t_mma.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <array>
#include <utility>

namespace ninfer::ops::detail {
namespace {

constexpr int kN                 = 34816;
constexpr int kK                 = 5120;
constexpr int kIntermediate      = kN / 2;
constexpr int kGroupK            = 64;
constexpr int kGroups            = kK / kGroupK;
constexpr int kBytesPerGroup     = 32;
constexpr int kVecBytes          = 16;
constexpr int kGroupsPerWarpTile = 16;
constexpr int kVecsPerWarpTile   = kGroupsPerWarpTile * kBytesPerGroup / kVecBytes;
constexpr int kWarpsPerBlock     = 4;
constexpr int kBlockThreads      = kWarpsPerBlock * 32;
constexpr int kPairsPerBlock     = kWarpsPerBlock;
constexpr int kXVecs             = kK / 8; // x as uint4 (8 bf16 each)
constexpr int kTiles             = kGroups / kGroupsPerWarpTile;
static_assert(kIntermediate % kPairsPerBlock == 0);
static_assert(kBytesPerGroup == 2 * kVecBytes);
static_assert(kGroups % kGroupsPerWarpTile == 0);
static_assert(kVecsPerWarpTile == 32);

struct Q4SwiGluSmallTGeometry {
    static constexpr int kInputRows    = kK;
    static constexpr int kGroupsPerRow = kK / kGroupK;
};

struct Q4SwiGluSmallTRows {
    static constexpr int kOutputRowsPerCta = 8;

    __device__ __forceinline__ int weight_row(int output_row0, int local_row) const {
        return output_row0 + (local_row & 7) + (local_row >= 8 ? kIntermediate : 0);
    }
};

struct Q4SwiGluSmallTEpilogue {
    __nv_bfloat16* out;

    template <int ActiveCols>
    __device__ __forceinline__ void store(int row, int col0, float4 projected) const {
        if (col0 < ActiveCols) {
            out[static_cast<std::int64_t>(col0) * kIntermediate + row] =
                __float2bfloat16_rn(silu(projected.x) * projected.z);
        }
        if (col0 + 1 < ActiveCols) {
            out[static_cast<std::int64_t>(col0 + 1) * kIntermediate + row] =
                __float2bfloat16_rn(silu(projected.y) * projected.w);
        }
    }
};

using SmallTLauncher = void (*)(const Tensor&, const Weight&, Tensor&, cudaStream_t);

template <int ActiveCols>
void launch_small_t_active(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    constexpr int TileCols =
        ActiveCols <= 8 ? 8 : (ActiveCols <= 16 ? 16 : (ActiveCols <= 24 ? 24 : 32));
    constexpr int kBlocks = kIntermediate / Q4SwiGluSmallTRows::kOutputRowsPerCta;
    const Q4SwiGluSmallTEpilogue epilogue{static_cast<__nv_bfloat16*>(out.data)};
    q4_small_t_mma_kernel<Q4SwiGluSmallTGeometry, TileCols, ActiveCols, Q4SwiGluSmallTEpilogue,
                          Q4SwiGluSmallTRows>
        <<<kBlocks, Q4DraftSmallTSchedule::kThreads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(w.qdata),
            static_cast<const std::uint8_t*>(w.scales), static_cast<__nv_bfloat16*>(out.data),
            epilogue, Q4SwiGluSmallTRows{});
    CUDA_CHECK(cudaGetLastError());
}

template <std::size_t... Offsets>
constexpr auto make_small_t_launchers(std::index_sequence<Offsets...>) {
    return std::array<SmallTLauncher, sizeof...(Offsets)>{
        &launch_small_t_active<2 + static_cast<int>(Offsets)>...};
}

constexpr auto kSmallTLaunchers = make_small_t_launchers(std::make_index_sequence<31>{});

__device__ __forceinline__ void q4_issue_pair_tile(uint4 (*__restrict__ s_code)[kVecsPerWarpTile],
                                                   uint4 (*__restrict__ s_scale)[2],
                                                   const std::uint8_t* __restrict__ gate_code_row,
                                                   const std::uint8_t* __restrict__ gate_scale_row,
                                                   const std::uint8_t* __restrict__ up_code_row,
                                                   const std::uint8_t* __restrict__ up_scale_row,
                                                   int tile, int lane) {
    const int g0 = tile * kGroupsPerWarpTile;
    pipe_copy<16>(&s_code[0][lane],
                  reinterpret_cast<const uint4*>(gate_code_row + g0 * kBytesPerGroup) + lane);
    pipe_copy<16>(&s_code[1][lane],
                  reinterpret_cast<const uint4*>(up_code_row + g0 * kBytesPerGroup) + lane);
    if (lane < 2) {
        pipe_copy<16>(&s_scale[0][lane],
                      reinterpret_cast<const uint4*>(gate_scale_row + g0 * 2) + lane);
        pipe_copy<16>(&s_scale[1][lane],
                      reinterpret_cast<const uint4*>(up_scale_row + g0 * 2) + lane);
    }
    pipe_commit();
}

__global__ void q4_linear_swiglu_gemv_pair_kernel(const __nv_bfloat16* __restrict__ x,
                                                  const std::uint8_t* __restrict__ codes,
                                                  const std::uint8_t* __restrict__ scales,
                                                  __nv_bfloat16* __restrict__ out) {
    constexpr int kStages   = 3;
    constexpr int kPrefetch = kStages - 1;
    __shared__ __align__(16) __nv_bfloat16 x_sh[kK];
    __shared__ uint4 code_tile[kWarpsPerBlock][kStages][2][kVecsPerWarpTile];
    __shared__ uint4 scale_tile[kWarpsPerBlock][kStages][2][2];

    auto* x_sh_v    = reinterpret_cast<uint4*>(x_sh);
    const auto* x_g = reinterpret_cast<const uint4*>(x);
    for (int i = static_cast<int>(threadIdx.x); i < kXVecs; i += static_cast<int>(blockDim.x)) {
        x_sh_v[i] = x_g[i];
    }
    __syncthreads();

    const int lane    = static_cast<int>(threadIdx.x) & 31;
    const int warp    = static_cast<int>(threadIdx.x) >> 5;
    const int out_row = static_cast<int>(blockIdx.x) * kPairsPerBlock + warp;

    const std::uint8_t* gate_code_row =
        codes + static_cast<std::int64_t>(out_row) * kGroups * kBytesPerGroup;
    const std::uint8_t* gate_scale_row = scales + static_cast<std::int64_t>(out_row) * kGroups * 2;
    const std::uint8_t* up_code_row =
        codes + static_cast<std::int64_t>(out_row + kIntermediate) * kGroups * kBytesPerGroup;
    const std::uint8_t* up_scale_row =
        scales + static_cast<std::int64_t>(out_row + kIntermediate) * kGroups * 2;
    const auto* x2 = reinterpret_cast<const __nv_bfloat162*>(x_sh);

    float gate_acc = 0.0f;
    float up_acc   = 0.0f;
#pragma unroll
    for (int p = 0; p < kPrefetch; ++p) {
        if (p < kTiles) {
            q4_issue_pair_tile(code_tile[warp][p], scale_tile[warp][p], gate_code_row,
                               gate_scale_row, up_code_row, up_scale_row, p, lane);
        } else {
            pipe_commit();
        }
    }

#pragma unroll 1
    for (int tile = 0; tile < kTiles; ++tile) {
        const int fetch = tile + kPrefetch;
        if (fetch < kTiles) {
            const int buf = fetch % kStages;
            q4_issue_pair_tile(code_tile[warp][buf], scale_tile[warp][buf], gate_code_row,
                               gate_scale_row, up_code_row, up_scale_row, fetch, lane);
        } else {
            pipe_commit();
        }
        pipe_wait<kPrefetch>();
        __syncwarp();

        const int buf           = tile % kStages;
        const auto* gate_codes  = reinterpret_cast<const std::uint8_t*>(code_tile[warp][buf][0]);
        const auto* up_codes    = reinterpret_cast<const std::uint8_t*>(code_tile[warp][buf][1]);
        const auto* gate_scales = reinterpret_cast<const std::uint16_t*>(scale_tile[warp][buf][0]);
        const auto* up_scales   = reinterpret_cast<const std::uint16_t*>(scale_tile[warp][buf][1]);
#pragma unroll
        for (int tile_group = 0; tile_group < kGroupsPerWarpTile; ++tile_group) {
            const float gate_scale =
                __half2float(__ushort_as_half(static_cast<std::uint16_t>(gate_scales[tile_group])));
            const float up_scale =
                __half2float(__ushort_as_half(static_cast<std::uint16_t>(up_scales[tile_group])));

            const int gate_packed =
                static_cast<int>(gate_codes[tile_group * kBytesPerGroup + lane]);
            const int gate_q0   = sign_extend<4>(gate_packed & 0x0f);
            const int gate_q1   = sign_extend<4>(gate_packed >> 4);
            const int up_packed = static_cast<int>(up_codes[tile_group * kBytesPerGroup + lane]);
            const int up_q0     = sign_extend<4>(up_packed & 0x0f);
            const int up_q1     = sign_extend<4>(up_packed >> 4);
            const int k0        = (tile * kGroupsPerWarpTile + tile_group) * kGroupK + lane * 2;
            const float2 xv     = __bfloat1622float2(x2[k0 >> 1]);
            gate_acc            = fmaf(static_cast<float>(gate_q0) * gate_scale, xv.x, gate_acc);
            gate_acc            = fmaf(static_cast<float>(gate_q1) * gate_scale, xv.y, gate_acc);
            up_acc              = fmaf(static_cast<float>(up_q0) * up_scale, xv.x, up_acc);
            up_acc              = fmaf(static_cast<float>(up_q1) * up_scale, xv.y, up_acc);
        }
        __syncwarp();
    }

    gate_acc = warp_reduce_sum(gate_acc);
    up_acc   = warp_reduce_sum(up_acc);
    if (lane == 0) { out[out_row] = __float2bfloat16(silu(gate_acc) * up_acc); }
}

// 直接读寄存器的 swiglu 变体，接管 T=2..32。原来的 q4_small_t_mma 是 mma_bf16 的 FMA
// 仿真（ops/common/mma.cuh:87 的 m16n8k16 fragment emulation），在生产几何与调度下实测只有
// **31.3 GB/s**（tools/q4_small_t_mma_probe.cu）；而 gate_up 每 token 要流 6.06 GB，即 ~195 ms/趟，
// 占 MTP 验证轮 ~340 ms 的六成。本核：一个 warp 吃一个输出行（gate 行与 up 行成对读，两者共享
// 同一份激活），相位 = 256 K、每 lane 8 个权重、相位循环 unroll 2，行内 warp 归约后写 silu(gate)·up。
// 索引与 q4_rowsplit_gemm_simt 的直接读版逐项对齐；NINFER_SWIGLU_SIMT=0 回退到仿真 mma 版。
constexpr int kSimtRowsPerCta = 8;
constexpr int kSimtPhaseK     = 256;

template <int ColsPerTile, bool FullTile = false>
__global__ __launch_bounds__(kSimtRowsPerCta * 32, 1) void q4_linear_swiglu_simt_direct_kernel(
    const __nv_bfloat16* __restrict__ x, const std::uint8_t* __restrict__ codes,
    const std::uint8_t* __restrict__ scales, __nv_bfloat16* __restrict__ out,
    std::int32_t active) {
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int row  = static_cast<int>(blockIdx.x) * kSimtRowsPerCta + warp;

    const std::uint8_t* gate_code =
        codes + static_cast<std::int64_t>(row) * (kK / 2);
    const std::uint8_t* up_code =
        codes + static_cast<std::int64_t>(row + kIntermediate) * (kK / 2);
    const std::uint8_t* gate_scale =
        scales + static_cast<std::int64_t>(row) * kGroups * 2;
    const std::uint8_t* up_scale =
        scales + static_cast<std::int64_t>(row + kIntermediate) * kGroups * 2;

    float gate_acc[ColsPerTile];
    float up_acc[ColsPerTile];
#pragma unroll
    for (int col = 0; col < ColsPerTile; ++col) {
        gate_acc[col] = 0.0f;
        up_acc[col]   = 0.0f;
    }

    constexpr int kPhases = kK / kSimtPhaseK;
#pragma unroll 2
    for (int phase = 0; phase < kPhases; ++phase) {
        const std::uint32_t gate_packed = *reinterpret_cast<const std::uint32_t*>(
            gate_code + static_cast<std::int64_t>(phase) * (kSimtPhaseK / 2) + lane * 4);
        const std::uint32_t up_packed = *reinterpret_cast<const std::uint32_t*>(
            up_code + static_cast<std::int64_t>(phase) * (kSimtPhaseK / 2) + lane * 4);
        const std::uint16_t gate_scale_bits = *reinterpret_cast<const std::uint16_t*>(
            gate_scale + (phase * 4 + (lane >> 3)) * 2);
        const std::uint16_t up_scale_bits = *reinterpret_cast<const std::uint16_t*>(
            up_scale + (phase * 4 + (lane >> 3)) * 2);

        float gate_w[8];
        float up_w[8];
        Q4SimtDecodeAtom::decode_eight(gate_packed, gate_scale_bits, gate_w);
        Q4SimtDecodeAtom::decode_eight(up_packed, up_scale_bits, up_w);

        const std::int64_t xk = static_cast<std::int64_t>(phase) * kSimtPhaseK + lane * 8;
#pragma unroll
        for (int col = 0; col < ColsPerTile; ++col) {
            // FullTile：满 tile 时 active == ColsPerTile，掩码判定整列可省（ISA 里那些 v_cmp 就来自这里）。
            if constexpr (!FullTile) {
                if (col >= active) { continue; }
            }
            const uint4 values = load_vec<uint4>(x + static_cast<std::int64_t>(col) * kK + xk);
            const float2 x0    = bf16x2_bits_to_float2(values.x);
            const float2 x1    = bf16x2_bits_to_float2(values.y);
            const float2 x2    = bf16x2_bits_to_float2(values.z);
            const float2 x3    = bf16x2_bits_to_float2(values.w);
            gate_acc[col]      = fmaf(gate_w[0], x0.x, gate_acc[col]);
            gate_acc[col]      = fmaf(gate_w[1], x0.y, gate_acc[col]);
            gate_acc[col]      = fmaf(gate_w[2], x1.x, gate_acc[col]);
            gate_acc[col]      = fmaf(gate_w[3], x1.y, gate_acc[col]);
            gate_acc[col]      = fmaf(gate_w[4], x2.x, gate_acc[col]);
            gate_acc[col]      = fmaf(gate_w[5], x2.y, gate_acc[col]);
            gate_acc[col]      = fmaf(gate_w[6], x3.x, gate_acc[col]);
            gate_acc[col]      = fmaf(gate_w[7], x3.y, gate_acc[col]);
            up_acc[col]        = fmaf(up_w[0], x0.x, up_acc[col]);
            up_acc[col]        = fmaf(up_w[1], x0.y, up_acc[col]);
            up_acc[col]        = fmaf(up_w[2], x1.x, up_acc[col]);
            up_acc[col]        = fmaf(up_w[3], x1.y, up_acc[col]);
            up_acc[col]        = fmaf(up_w[4], x2.x, up_acc[col]);
            up_acc[col]        = fmaf(up_w[5], x2.y, up_acc[col]);
            up_acc[col]        = fmaf(up_w[6], x3.x, up_acc[col]);
            up_acc[col]        = fmaf(up_w[7], x3.y, up_acc[col]);
        }
    }

    // 一条 warp 独占一个输出行，32 条 lane 覆盖互不重叠的 K 片段，行内归约收尾。
#pragma unroll
    for (int col = 0; col < ColsPerTile; ++col) {
        if constexpr (!FullTile) {
            if (col >= active) { continue; }
        }
        const float gate = warp_reduce_sum(gate_acc[col]);
        const float up   = warp_reduce_sum(up_acc[col]);
        if (lane == 0) {
            out[static_cast<std::int64_t>(col) * kIntermediate + row] =
                __float2bfloat16_rn(silu(gate) * up);
        }
    }
}

template <int ColsPerTile, bool FullTile = false>
void launch_simt_direct(const Tensor& x, const Weight& w, Tensor& out, std::int32_t col0,
                        std::int32_t cols, cudaStream_t stream) {
    const dim3 grid(static_cast<unsigned>(kIntermediate / kSimtRowsPerCta), 1u, 1u);
    q4_linear_swiglu_simt_direct_kernel<ColsPerTile, FullTile>
        <<<grid, kSimtRowsPerCta * 32, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data) + static_cast<std::int64_t>(col0) * kK,
            static_cast<const std::uint8_t*>(w.qdata),
            static_cast<const std::uint8_t*>(w.scales),
            static_cast<__nv_bfloat16*>(out.data) +
                static_cast<std::int64_t>(col0) * kIntermediate,
            cols);
    CUDA_CHECK(cudaGetLastError());
}

// 行分块直读核（2026-09-29）：x 重读系数的直接验证与修复。simt_direct 每 phase 每 lane 权重
// 只有 8 B（gate+up 各 u32）而 x 是 ColsPerTile×16 B，C8 下 16:1；全核 x 流量 = 17408 warp ×
// 20 phase × 8 col × 512 B ≈ 1.43 GB（L2）vs 权重 89 MB（DRAM），按实测 275 GB/s 反推 x 重读
// 占核时 ~2/3。本核每 warp 吃 RowsPerWarp 个行对，x 每份喂 2×RowsPerWarp 条 fma 链，x 流量
// ÷RowsPerWarp。行内累加顺序与 simt_direct 逐位一致（同 phase 序 / 同 lane 片 / 同 warp 归约
// 树）→ 输出逐位相同。默认 RB=4（C8 +6.0%），NINFER_SWIGLU_RB=0/2 调档。
constexpr int kRbWarps = 8;

template <int ColsPerTile, bool FullTile, int RowsPerWarp>
__global__ __launch_bounds__(kRbWarps * 32, 1) void q4_linear_swiglu_simt_rb_kernel(
    const __nv_bfloat16* __restrict__ x, const std::uint8_t* __restrict__ codes,
    const std::uint8_t* __restrict__ scales, __nv_bfloat16* __restrict__ out,
    std::int32_t active) {
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int row_base =
        static_cast<int>(blockIdx.x) * (kRbWarps * RowsPerWarp) + warp * RowsPerWarp;

    const std::uint8_t* gate_code[RowsPerWarp];
    const std::uint8_t* up_code[RowsPerWarp];
    const std::uint8_t* gate_scale[RowsPerWarp];
    const std::uint8_t* up_scale[RowsPerWarp];
#pragma unroll
    for (int r = 0; r < RowsPerWarp; ++r) {
        const int row = row_base + r;
        gate_code[r]  = codes + static_cast<std::int64_t>(row) * (kK / 2);
        up_code[r]    = codes + static_cast<std::int64_t>(row + kIntermediate) * (kK / 2);
        gate_scale[r] = scales + static_cast<std::int64_t>(row) * kGroups * 2;
        up_scale[r]   = scales + static_cast<std::int64_t>(row + kIntermediate) * kGroups * 2;
    }

    float accg[RowsPerWarp][ColsPerTile];
    float accu[RowsPerWarp][ColsPerTile];
#pragma unroll
    for (int r = 0; r < RowsPerWarp; ++r) {
#pragma unroll
        for (int col = 0; col < ColsPerTile; ++col) {
            accg[r][col] = 0.0f;
            accu[r][col] = 0.0f;
        }
    }

    constexpr int kPhases = kK / kSimtPhaseK;
    // RW=4 时 unroll 2 会把两份瞬态权重数组（2×64 float）叠上 64 个累加器 → 240 VGPR +
    // 322 溢出槽（ISA 实测）；unroll 1 后瞬态与累加器重叠，落回正常预算。
    auto phase_step = [&](int phase) {
        std::uint32_t gate_packed[RowsPerWarp];
        std::uint32_t up_packed[RowsPerWarp];
        std::uint16_t gate_scale_bits[RowsPerWarp];
        std::uint16_t up_scale_bits[RowsPerWarp];
#pragma unroll
        for (int r = 0; r < RowsPerWarp; ++r) {
            gate_packed[r] = *reinterpret_cast<const std::uint32_t*>(
                gate_code[r] + static_cast<std::int64_t>(phase) * (kSimtPhaseK / 2) + lane * 4);
            up_packed[r] = *reinterpret_cast<const std::uint32_t*>(
                up_code[r] + static_cast<std::int64_t>(phase) * (kSimtPhaseK / 2) + lane * 4);
            gate_scale_bits[r] = *reinterpret_cast<const std::uint16_t*>(
                gate_scale[r] + (phase * 4 + (lane >> 3)) * 2);
            up_scale_bits[r] = *reinterpret_cast<const std::uint16_t*>(
                up_scale[r] + (phase * 4 + (lane >> 3)) * 2);
        }
        float gate_w[RowsPerWarp][8];
        float up_w[RowsPerWarp][8];
#pragma unroll
        for (int r = 0; r < RowsPerWarp; ++r) {
            Q4SimtDecodeAtom::decode_eight(gate_packed[r], gate_scale_bits[r], gate_w[r]);
            Q4SimtDecodeAtom::decode_eight(up_packed[r], up_scale_bits[r], up_w[r]);
        }

        const std::int64_t xk = static_cast<std::int64_t>(phase) * kSimtPhaseK + lane * 8;
#pragma unroll
        for (int col = 0; col < ColsPerTile; ++col) {
            if constexpr (!FullTile) {
                if (col >= active) { continue; }
            }
            const uint4 values = load_vec<uint4>(x + static_cast<std::int64_t>(col) * kK + xk);
            const float2 x0    = bf16x2_bits_to_float2(values.x);
            const float2 x1    = bf16x2_bits_to_float2(values.y);
            const float2 x2    = bf16x2_bits_to_float2(values.z);
            const float2 x3    = bf16x2_bits_to_float2(values.w);
#pragma unroll
            for (int r = 0; r < RowsPerWarp; ++r) {
                accg[r][col]      = fmaf(gate_w[r][0], x0.x, accg[r][col]);
                accg[r][col]      = fmaf(gate_w[r][1], x0.y, accg[r][col]);
                accg[r][col]      = fmaf(gate_w[r][2], x1.x, accg[r][col]);
                accg[r][col]      = fmaf(gate_w[r][3], x1.y, accg[r][col]);
                accg[r][col]      = fmaf(gate_w[r][4], x2.x, accg[r][col]);
                accg[r][col]      = fmaf(gate_w[r][5], x2.y, accg[r][col]);
                accg[r][col]      = fmaf(gate_w[r][6], x3.x, accg[r][col]);
                accg[r][col]      = fmaf(gate_w[r][7], x3.y, accg[r][col]);
                accu[r][col]      = fmaf(up_w[r][0], x0.x, accu[r][col]);
                accu[r][col]      = fmaf(up_w[r][1], x0.y, accu[r][col]);
                accu[r][col]      = fmaf(up_w[r][2], x1.x, accu[r][col]);
                accu[r][col]      = fmaf(up_w[r][3], x1.y, accu[r][col]);
                accu[r][col]      = fmaf(up_w[r][4], x2.x, accu[r][col]);
                accu[r][col]      = fmaf(up_w[r][5], x2.y, accu[r][col]);
                accu[r][col]      = fmaf(up_w[r][6], x3.x, accu[r][col]);
                accu[r][col]      = fmaf(up_w[r][7], x3.y, accu[r][col]);
            }
        }
    };
    if constexpr (RowsPerWarp >= 4) {
#pragma unroll 1
        for (int phase = 0; phase < kPhases; ++phase) { phase_step(phase); }
    } else {
#pragma unroll 2
        for (int phase = 0; phase < kPhases; ++phase) { phase_step(phase); }
    }

#pragma unroll
    for (int r = 0; r < RowsPerWarp; ++r) {
        const int row = row_base + r;
#pragma unroll
        for (int col = 0; col < ColsPerTile; ++col) {
            if constexpr (!FullTile) {
                if (col >= active) { continue; }
            }
            const float gate = warp_reduce_sum(accg[r][col]);
            const float up   = warp_reduce_sum(accu[r][col]);
            if (lane == 0) {
                out[static_cast<std::int64_t>(col) * kIntermediate + row] =
                    __float2bfloat16_rn(silu(gate) * up);
            }
        }
    }
}

template <int ColsPerTile, bool FullTile, int RowsPerWarp>
void launch_simt_rb(const Tensor& x, const Weight& w, Tensor& out, std::int32_t col0,
                    std::int32_t cols, cudaStream_t stream) {
    const dim3 grid(static_cast<unsigned>(kIntermediate / (kRbWarps * RowsPerWarp)), 1u, 1u);
    q4_linear_swiglu_simt_rb_kernel<ColsPerTile, FullTile, RowsPerWarp>
        <<<grid, kRbWarps * 32, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data) + static_cast<std::int64_t>(col0) * kK,
            static_cast<const std::uint8_t*>(w.qdata),
            static_cast<const std::uint8_t*>(w.scales),
            static_cast<__nv_bfloat16*>(out.data) +
                static_cast<std::int64_t>(col0) * kIntermediate,
            cols);
    CUDA_CHECK(cudaGetLastError());
}

// t2 级重设计第一刀（2026-09-29）：现行 simt_direct 每 phase 每 lane 只有 8 B 权重加载在飞
// （2×u32），launch_bounds(128,1) 不保证驻留，小 T 家族 275 GB/s 的天花板账指向加载延迟暴露。
// 本核四项结构改动：① 每 lane 的 k 片从 8 加宽到 32，权重加载升级 uint4（在飞字节 ×4）；
// ② 256 线程 CTA + minBlocks=3 驻留下限；③ unscaled 解码（gdn/swiglu 窄核已过 PPL 的数值类），
// 组 scale 每 phase 每 row 折叠一次；④ 激活保持全局直读（80 KB 工作集常驻 L1/L2，跨 CTA 共享）。
// LDS 零占用。NINFER_SWIGLU_SLIM=1 进路由。
constexpr int kSlabWarps  = 8;   // 1 warp = 1 row pair（gate+up）
constexpr int kSlabPhaseK = 1024;
constexpr int kSlabSliceK = 32;  // per lane; uint4 = 16 B = 32 codes

__device__ __forceinline__ void q4_slab_decode_raw(std::uint32_t packed, float (&raw)[8]) {
    const std::uint32_t word = packed ^ 0x88888888u;
    const __half2 bias       = __half2half2(__ushort_as_half(0x6408)); // 1032.0
#pragma unroll
    for (int pair = 0; pair < 4; ++pair) {
        const std::uint32_t bits = ((word >> (4 * pair)) & 0x000f000fu) | 0x64006400u;
        const float2 values      = __half22float2(__hsub2(half2_from_bits(bits), bias));
        raw[pair]                = values.x;
        raw[pair + 4]            = values.y;
    }
}

template <int ColsPerTile, bool FullTile, int MinBlocks = 3>
__global__ __launch_bounds__(kSlabWarps * 32, MinBlocks) void q4_linear_swiglu_slab_kernel(
    const __nv_bfloat16* __restrict__ x, const std::uint8_t* __restrict__ codes,
    const std::uint8_t* __restrict__ scales, __nv_bfloat16* __restrict__ out,
    std::int32_t active) {
    static_assert(kK % kSlabPhaseK == 0, "slab phases must tile K exactly");
    static_assert(kGroupK % kSlabSliceK == 0, "lane slice must nest inside a scale group");
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    // CTA 覆盖 kSlabWarps 个行对（每 warp 一对：gate 行 + 17408 偏移的 up 行）。
    const int gate_row = static_cast<int>(blockIdx.x) * kSlabWarps + warp;

    const std::uint8_t* gate_code = codes + static_cast<std::int64_t>(gate_row) * (kK / 2);
    const std::uint8_t* up_code =
        codes + static_cast<std::int64_t>(gate_row + kIntermediate) * (kK / 2);
    const std::uint16_t* gate_scale =
        reinterpret_cast<const std::uint16_t*>(scales + static_cast<std::int64_t>(gate_row) *
                                                                 kGroups * 2);
    const std::uint16_t* up_scale = gate_scale + kGroups;

    float accg[ColsPerTile];
    float accu[ColsPerTile];
#pragma unroll
    for (int col = 0; col < ColsPerTile; ++col) {
        accg[col] = 0.0f;
        accu[col] = 0.0f;
    }

    constexpr int kPhases      = kK / kSlabPhaseK;
    constexpr int kPhaseBytes  = kSlabPhaseK / 2;
    constexpr int kPhaseGroups = kSlabPhaseK / kGroupK;
#pragma unroll 2
    for (int phase = 0; phase < kPhases; ++phase) {
        const std::uint32_t gw[4] = {
            *reinterpret_cast<const std::uint32_t*>(gate_code + phase * kPhaseBytes + lane * 16),
            *reinterpret_cast<const std::uint32_t*>(gate_code + phase * kPhaseBytes + lane * 16 + 4),
            *reinterpret_cast<const std::uint32_t*>(gate_code + phase * kPhaseBytes + lane * 16 + 8),
            *reinterpret_cast<const std::uint32_t*>(gate_code + phase * kPhaseBytes + lane * 16 + 12)};
        const std::uint32_t uw[4] = {
            *reinterpret_cast<const std::uint32_t*>(up_code + phase * kPhaseBytes + lane * 16),
            *reinterpret_cast<const std::uint32_t*>(up_code + phase * kPhaseBytes + lane * 16 + 4),
            *reinterpret_cast<const std::uint32_t*>(up_code + phase * kPhaseBytes + lane * 16 + 8),
            *reinterpret_cast<const std::uint32_t*>(up_code + phase * kPhaseBytes + lane * 16 + 12)};
        // lane 的 32-k 片整落在一个 64-k 组内（lane*32 与组边界对齐）。
        const int group = phase * kPhaseGroups + (lane >> 1);
        const float gs  = __half2float(__ushort_as_half(gate_scale[group]));
        const float us  = __half2float(__ushort_as_half(up_scale[group]));

        float pg[ColsPerTile];
        float pu[ColsPerTile];
#pragma unroll
        for (int col = 0; col < ColsPerTile; ++col) {
            pg[col] = 0.0f;
            pu[col] = 0.0f;
        }
#pragma unroll
        for (int word = 0; word < 4; ++word) {
            float graw[8];
            float uraw[8];
            q4_slab_decode_raw(gw[word], graw);
            q4_slab_decode_raw(uw[word], uraw);
            const std::int64_t xk = static_cast<std::int64_t>(phase) * kSlabPhaseK +
                                    lane * kSlabSliceK + word * 8;
#pragma unroll
            for (int col = 0; col < ColsPerTile; ++col) {
                if constexpr (!FullTile) {
                    if (col >= active) { continue; }
                }
                const uint4 values = load_vec<uint4>(x + static_cast<std::int64_t>(col) * kK + xk);
                const float2 x0    = bf16x2_bits_to_float2(values.x);
                const float2 x1    = bf16x2_bits_to_float2(values.y);
                const float2 x2    = bf16x2_bits_to_float2(values.z);
                const float2 x3    = bf16x2_bits_to_float2(values.w);
                pg[col]            = fmaf(graw[0], x0.x, pg[col]);
                pg[col]            = fmaf(graw[1], x0.y, pg[col]);
                pg[col]            = fmaf(graw[2], x1.x, pg[col]);
                pg[col]            = fmaf(graw[3], x1.y, pg[col]);
                pg[col]            = fmaf(graw[4], x2.x, pg[col]);
                pg[col]            = fmaf(graw[5], x2.y, pg[col]);
                pg[col]            = fmaf(graw[6], x3.x, pg[col]);
                pg[col]            = fmaf(graw[7], x3.y, pg[col]);
                pu[col]            = fmaf(uraw[0], x0.x, pu[col]);
                pu[col]            = fmaf(uraw[1], x0.y, pu[col]);
                pu[col]            = fmaf(uraw[2], x1.x, pu[col]);
                pu[col]            = fmaf(uraw[3], x1.y, pu[col]);
                pu[col]            = fmaf(uraw[4], x2.x, pu[col]);
                pu[col]            = fmaf(uraw[5], x2.y, pu[col]);
                pu[col]            = fmaf(uraw[6], x3.x, pu[col]);
                pu[col]            = fmaf(uraw[7], x3.y, pu[col]);
            }
        }
#pragma unroll
        for (int col = 0; col < ColsPerTile; ++col) {
            accg[col] = fmaf(pg[col], gs, accg[col]);
            accu[col] = fmaf(pu[col], us, accu[col]);
        }
    }

#pragma unroll
    for (int col = 0; col < ColsPerTile; ++col) {
        if constexpr (!FullTile) {
            if (col >= active) { continue; }
        }
        const float gate = warp_reduce_sum(accg[col]);
        const float up   = warp_reduce_sum(accu[col]);
        if (lane == 0) {
            out[static_cast<std::int64_t>(col) * kIntermediate + gate_row] =
                __float2bfloat16_rn(silu(gate) * up);
        }
    }
}

template <int ColsPerTile, bool FullTile>
void launch_slab(const Tensor& x, const Weight& w, Tensor& out, std::int32_t col0,
                 std::int32_t cols, cudaStream_t stream) {
    const dim3 grid(static_cast<unsigned>(kIntermediate / kSlabWarps), 1u, 1u);
    q4_linear_swiglu_slab_kernel<ColsPerTile, FullTile>
        <<<grid, kSlabWarps * 32, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data) + static_cast<std::int64_t>(col0) * kK,
            static_cast<const std::uint8_t*>(w.qdata),
            static_cast<const std::uint8_t*>(w.scales),
            static_cast<__nv_bfloat16*>(out.data) +
                static_cast<std::int64_t>(col0) * kIntermediate,
            cols);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

void q4_linear_swiglu_gemv_pair_launch(const Tensor& x, const Weight& w, Tensor& out,
                                       cudaStream_t stream) {
    if (w.n != kN || w.k != kK || w.padded_shape[1] != kK) {
        throw std::invalid_argument("q4 linear_swiglu GEMV requires weight [34816,5120]");
    }
    const int grid = kIntermediate / kPairsPerBlock;
    q4_linear_swiglu_gemv_pair_kernel<<<grid, kBlockThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(w.qdata),
        static_cast<const std::uint8_t*>(w.scales), static_cast<__nv_bfloat16*>(out.data));
    CUDA_CHECK(cudaGetLastError());
}

void q4_linear_swiglu_small_t_exact_launch(const Tensor& x, const Weight& w, Tensor& out,
                                           cudaStream_t stream) {
    if (x.ne[1] < 2 || x.ne[1] > 32) {
        throw std::invalid_argument("Q4 LinearSwiGLU exact small-T requires T=2..32");
    }
    // T>=2 默认走直接读 SIMT 核（见上面 kernel 处的说明）：仿真 mma 版在同样形状下只有 31 GB/s。
    static const bool simt_enabled = [] {
        const char* value = std::getenv("NINFER_SWIGLU_SIMT");
        return value == nullptr || std::atoi(value) != 0;
    }();
    if (simt_enabled && w.n == kN && w.k == kK && w.padded_shape[1] == kK) {
        const std::int32_t cols = x.ne[1];
        // 每列 tile 的宽度 = 每次 launch 的权重遍数：4 列时 T=5..8 要跑两遍（2× 权重流量），
        // C8 的 T=8 与 MTP 的 T=6 都吃这个亏。8 列时每个 phase 的权重复用翻倍，
        // 而累加器按列独立 → **数值逐位不变**。默认关，先 A/B 再默认化。
        static const bool simt8_enabled = [] {
            const char* value = std::getenv("NINFER_SWIGLU_SIMT8");
            // 已默认开：C8 实测 80.0 → 87.6 t/s（+9.5%，与 attention 叠加 122.6）、数值逐位不变
            // （validate-pending 2026-09-23）；NINFER_SWIGLU_SIMT8=0 回退 4 列 tile。
            return value == nullptr || std::atoi(value) != 0;
        }();
        const std::int32_t tile = simt8_enabled ? 8 : 4;
        // 满 tile 时选无掩码特化：ISA 实测 Li8ELb0 → Li8ELb1 去掉 41 条 v_cmp、
        // 总指令 2278 → 1853（−19%）。**数值逐位不变**（满 tile 下没有列被掩掉），
        // 但"ISA 更短 ≠ 更快"是本仓踩过的坑，所以和 tile=8 一起挂在同一个 env 下等 A/B。
        // t2 级 slab 核（NINFER_SWIGLU_SLIM=1）：uint4 权重加载 + minBlocks 3 驻留下限 +
        // unscaled 组折叠。默认关，先 C8 A/B。
        static const bool slim_enabled = [] {
            const char* value = std::getenv("NINFER_SWIGLU_SLIM");
            return value != nullptr && std::atoi(value) != 0;
        }();
        // 行分块直读核（NINFER_SWIGLU_RB=2/4）：x 重读 ÷RowsPerWarp，数值与 SIMT8 逐位一致。
        // 已默认开 RB=4：同二进制 C8 122.0→129.3 t/s（+6.0%）、MLP 段 -3.3 ms/步、
        // T3 PPL 4.975734 逐位一致、mtp3 greedy 两臂逐字相同、C1 mtp0/mtp3 32.1/62.3 无回退
        // （validate 2026-09-29）；NINFER_SWIGLU_RB=0 回退单行对版。
        static const int rb_rows = [] {
            const char* value = std::getenv("NINFER_SWIGLU_RB");
            if (value == nullptr) { return 4; }
            const int v = std::atoi(value);
            return (v == 2 || v == 4 || v == 8) ? v : 0;
        }();
        for (std::int32_t col0 = 0; col0 < cols; col0 += tile) {
            const std::int32_t slice = (cols - col0 < tile) ? (cols - col0) : tile;
            const bool full          = slice == tile && simt8_enabled;
            if (rb_rows == 8) {
                if (full) {
                    launch_simt_rb<8, true, 8>(x, w, out, col0, slice, stream);
                } else {
                    launch_simt_rb<8, false, 8>(x, w, out, col0, slice, stream);
                }
                continue;
            }
            if (rb_rows == 4) {
                if (full) {
                    launch_simt_rb<8, true, 4>(x, w, out, col0, slice, stream);
                } else {
                    launch_simt_rb<8, false, 4>(x, w, out, col0, slice, stream);
                }
                continue;
            }
            if (rb_rows == 2) {
                if (full) {
                    launch_simt_rb<8, true, 2>(x, w, out, col0, slice, stream);
                } else {
                    launch_simt_rb<8, false, 2>(x, w, out, col0, slice, stream);
                }
                continue;
            }
            if (slim_enabled) {
                if (full) {
                    launch_slab<8, true>(x, w, out, col0, slice, stream);
                } else {
                    launch_slab<8, false>(x, w, out, col0, slice, stream);
                }
                continue;
            }
            if (tile == 8) {
                if (full) {
                    launch_simt_direct<8, true>(x, w, out, col0, slice, stream);
                } else {
                    launch_simt_direct<8>(x, w, out, col0, slice, stream);
                }
            } else {
                if (full) {
                    launch_simt_direct<4, true>(x, w, out, col0, slice, stream);
                } else {
                    launch_simt_direct<4>(x, w, out, col0, slice, stream);
                }
            }
        }
        return;
    }
    kSmallTLaunchers[static_cast<std::size_t>(x.ne[1] - 2)](x, w, out, stream);
}

} // namespace ninfer::ops::detail

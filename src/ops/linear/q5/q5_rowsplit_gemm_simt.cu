#include "core/device.h"
#include "ops/common/math.h"
#include "ops/common/token_slices.h"
#include "ops/linear/q5/q5_launch.h"
#include "ops/linear/q5/q5_rowsplit_gemm_simt.cuh"

#include <cuda_bf16.h>

#include <cstdint>
#include <cstdlib>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

constexpr int kRowsPerBlock = 8;
constexpr int kStages       = 2;

// 直接读寄存器的变体（与 q4 的同名改动同源）：生产核把 nibble/high/scale 三个面经 LDS 暂存，
// gfx11 上 cp_async 是 4B/lane 的 LDS DMA，再加 cp_wait/__syncwarp 的串行点，实测同形状只有
// 290-375 GB/s；直接读进寄存器实测 485-666 GB/s（tools/q4_shape_probe.cu 同布局对拍）。
// 索引与 LDS 版逐项对齐（见下方 phase 注释），累加顺序不变，结果逐位相同。
// NINFER_Q5_DIRECT=0 回退到 LDS 版。
constexpr int kDirectRowsPerCta = 8;
constexpr int kDirectPhaseK     = 256;

template <int ColsPerTile>
__global__ __launch_bounds__(kDirectRowsPerCta * 32, 1) void q5_rowsplit_gemm_simt_direct_kernel(
    const __nv_bfloat16* __restrict__ x, const std::uint8_t* __restrict__ codes,
    const std::uint8_t* __restrict__ high, const std::uint8_t* __restrict__ scales,
    __nv_bfloat16* __restrict__ out, std::int32_t out_ld, std::int32_t rows, std::int32_t k,
    std::int32_t cols, std::int32_t padded_k) {
    constexpr int kGroupK = Q5RowSplitStorage::kGroupK;

    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int row  = static_cast<int>(blockIdx.x) * kDirectRowsPerCta + warp;
    if (row >= rows) { return; }

    const int groups_per_row = padded_k / kGroupK;
    const std::uint8_t* code_row =
        codes + static_cast<std::int64_t>(row) * groups_per_row * 32;
    const std::uint8_t* high_row =
        high + static_cast<std::int64_t>(row) * groups_per_row *
                   Q5RowSplitSimtSchedule::kHighBytesPerGroup;
    const std::uint8_t* scale_row = scales + static_cast<std::int64_t>(row) * groups_per_row * 2;

    float acc[ColsPerTile];
#pragma unroll
    for (int col = 0; col < ColsPerTile; ++col) { acc[col] = 0.0f; }

    const int phases = k / kDirectPhaseK;
#pragma unroll 2
    for (int phase = 0; phase < phases; ++phase) {
        // 相位 = 256 个 K。LDS 版按 slab(1024 K)/相位(c) 索引，这里把 slab*4+c 折成 phase：
        //   nibble 面 uint32 下标 c*32+lane  -> 字节 c*128 + lane*4 = phase*128 + lane*4
        //   high   面 字节下标 c*32+lane     -> phase*32 + lane
        //   scale  面 uint16 下标 c*4+(lane>>3) -> (phase*4 + (lane>>3))*2
        const std::uint32_t word = *reinterpret_cast<const std::uint32_t*>(
            code_row + static_cast<std::int64_t>(phase) * (kDirectPhaseK / 2) + lane * 4);
        const std::uint32_t hc =
            static_cast<std::uint32_t>(high_row[static_cast<std::int64_t>(phase) * 32 + lane]) ^
            0xffu;
        const float scale = __half2float(__ushort_as_half(*reinterpret_cast<const std::uint16_t*>(
            scale_row + (phase * 4 + (lane >> 3)) * 2)));

        float weights[8];
        const __half2 bias = __half2half2(__ushort_as_half(0x6410));  // 1040.0
#pragma unroll
        for (int p = 0; p < 4; ++p) {
            std::uint32_t bits = ((word >> (4 * p)) & 0x000f000fu) | 0x64006400u;
            bits |= (((hc >> p) & 1u) << 4) | (((hc >> (p + 4)) & 1u) << 20);
            const __half2 h = __hsub2(half2_from_bits(bits), bias);
            const float2 f  = __half22float2(h);
            weights[p]      = f.x * scale;
            weights[p + 4]  = f.y * scale;
        }

        const std::int64_t xk = static_cast<std::int64_t>(phase) * kDirectPhaseK + lane * 8;
#pragma unroll
        for (int col = 0; col < ColsPerTile; ++col) {
            if (col >= cols) { continue; }
            const uint4 values = load_vec<uint4>(x + static_cast<std::int64_t>(col) * k + xk);
            const float2 x0    = bf16x2_bits_to_float2(values.x);
            const float2 x1    = bf16x2_bits_to_float2(values.y);
            const float2 x2    = bf16x2_bits_to_float2(values.z);
            const float2 x3    = bf16x2_bits_to_float2(values.w);
            acc[col]           = fmaf(weights[0], x0.x, acc[col]);
            acc[col]           = fmaf(weights[1], x0.y, acc[col]);
            acc[col]           = fmaf(weights[2], x1.x, acc[col]);
            acc[col]           = fmaf(weights[3], x1.y, acc[col]);
            acc[col]           = fmaf(weights[4], x2.x, acc[col]);
            acc[col]           = fmaf(weights[5], x2.y, acc[col]);
            acc[col]           = fmaf(weights[6], x3.x, acc[col]);
            acc[col]           = fmaf(weights[7], x3.y, acc[col]);
        }
    }

#pragma unroll
    for (int col = 0; col < ColsPerTile; ++col) {
        if (col >= cols) { continue; }
        const float sum = warp_reduce_sum(acc[col]);
        if (lane == 0) { out[static_cast<std::int64_t>(col) * out_ld + row] = __float2bfloat16(sum); }
    }
}

template <int ColsPerTile>
void launch_direct(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    const std::int32_t rows     = out.ne[0];
    const std::int32_t k        = x.ne[0];
    const std::int32_t out_ld   = static_cast<std::int32_t>(out.nb[1] / sizeof(__nv_bfloat16));
    const std::int32_t padded_k = w.padded_shape[1];
    const dim3 grid(static_cast<unsigned>(div_up(rows, kDirectRowsPerCta)), 1u, 1u);
    q5_rowsplit_gemm_simt_direct_kernel<ColsPerTile>
        <<<grid, kDirectRowsPerCta * 32, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(w.qdata),
            static_cast<const std::uint8_t*>(w.qhigh), static_cast<const std::uint8_t*>(w.scales),
            static_cast<__nv_bfloat16*>(out.data), out_ld, rows, k, x.ne[1], padded_k);
    CUDA_CHECK(cudaGetLastError());
}

bool q5_direct_eligible(const Tensor& x, const Weight& w) {
    const std::int32_t k        = x.ne[0];
    const std::int32_t padded_k = w.padded_shape[1];
    return k > 0 && (k % kDirectPhaseK) == 0 && padded_k >= k && (padded_k % 64) == 0 &&
           w.qhigh != nullptr;
}

// 行分块直读核（2026-09-29）：q4 swiglu RB 同款手术。q5 直读核（1 warp = 1 行）在 C8 下
// x:权重 = 8×512 B : 32×11.5 B ≈ 11:1，全核 x 流量 ≈ 1.43 GB（L2）vs 权重 58.5 MB（DRAM）；
// 且 T=7..24 的 down/gdn_out_proj 路由（r8_c8）根本没接直读路径，走的是 LDS+cp_async 暂存核。
// 本核每 warp 吃 RowsPerWarp 行，x 每份喂 RowsPerWarp 条 fma 链；行内累加顺序与 direct 核
// 逐位一致（同 phase 序 / 同 lane 片 / 同 warp 归约树）。NINFER_Q5_RB=2/4/8 进路由，默认关。
constexpr int kQ5RbWarps = 8;

template <int ColsPerTile, int RowsPerWarp>
__global__ __launch_bounds__(kQ5RbWarps * 32, 1) void q5_rowsplit_gemm_simt_rb_kernel(
    const __nv_bfloat16* __restrict__ x, const std::uint8_t* __restrict__ codes,
    const std::uint8_t* __restrict__ high, const std::uint8_t* __restrict__ scales,
    __nv_bfloat16* __restrict__ out, std::int32_t out_ld, std::int32_t rows, std::int32_t k,
    std::int32_t cols, std::int32_t padded_k) {
    constexpr int kGroupK = Q5RowSplitStorage::kGroupK;

    const int lane     = static_cast<int>(threadIdx.x) & 31;
    const int warp     = static_cast<int>(threadIdx.x) >> 5;
    const int row_base =
        static_cast<int>(blockIdx.x) * (kQ5RbWarps * RowsPerWarp) + warp * RowsPerWarp;

    const int groups_per_row = padded_k / kGroupK;
    const std::uint8_t* code_row[RowsPerWarp];
    const std::uint8_t* high_row[RowsPerWarp];
    const std::uint8_t* scale_row[RowsPerWarp];
    bool row_valid[RowsPerWarp];
#pragma unroll
    for (int r = 0; r < RowsPerWarp; ++r) {
        const int row = row_base + r;
        row_valid[r]  = row < rows;
        code_row[r]   = codes + static_cast<std::int64_t>(row) * groups_per_row * 32;
        high_row[r]   = high + static_cast<std::int64_t>(row) * groups_per_row *
                                   Q5RowSplitSimtSchedule::kHighBytesPerGroup;
        scale_row[r] = scales + static_cast<std::int64_t>(row) * groups_per_row * 2;
    }

    float acc[RowsPerWarp][ColsPerTile];
#pragma unroll
    for (int r = 0; r < RowsPerWarp; ++r) {
#pragma unroll
        for (int col = 0; col < ColsPerTile; ++col) { acc[r][col] = 0.0f; }
    }

    const int phases = k / kDirectPhaseK;
    // RW≥4 时 unroll 2 会把 RW×8 的瞬态权重数组翻倍叠上 RW×Cols 累加器（q4 RB 实测
    // 240 VGPR + 322 溢出槽），降 unroll 1。
    auto phase_step = [&](int phase) {
        std::uint32_t word[RowsPerWarp];
        std::uint32_t hc[RowsPerWarp];
        float scale[RowsPerWarp];
#pragma unroll
        for (int r = 0; r < RowsPerWarp; ++r) {
            if (row_valid[r]) {
                word[r] = *reinterpret_cast<const std::uint32_t*>(
                    code_row[r] + static_cast<std::int64_t>(phase) * (kDirectPhaseK / 2) +
                    lane * 4);
                hc[r] = static_cast<std::uint32_t>(
                            high_row[r][static_cast<std::int64_t>(phase) * 32 + lane]) ^
                        0xffu;
                scale[r] = __half2float(__ushort_as_half(*reinterpret_cast<const std::uint16_t*>(
                    scale_row[r] + (phase * 4 + (lane >> 3)) * 2)));
            }
        }
        float weights[RowsPerWarp][8];
        const __half2 bias = __half2half2(__ushort_as_half(0x6410));  // 1040.0
#pragma unroll
        for (int r = 0; r < RowsPerWarp; ++r) {
            if (!row_valid[r]) { continue; }
#pragma unroll
            for (int p = 0; p < 4; ++p) {
                std::uint32_t bits = ((word[r] >> (4 * p)) & 0x000f000fu) | 0x64006400u;
                bits |= (((hc[r] >> p) & 1u) << 4) | (((hc[r] >> (p + 4)) & 1u) << 20);
                const __half2 h   = __hsub2(half2_from_bits(bits), bias);
                const float2 f    = __half22float2(h);
                weights[r][p]     = f.x * scale[r];
                weights[r][p + 4] = f.y * scale[r];
            }
        }

        const std::int64_t xk = static_cast<std::int64_t>(phase) * kDirectPhaseK + lane * 8;
#pragma unroll
        for (int col = 0; col < ColsPerTile; ++col) {
            if (col >= cols) { continue; }
            const uint4 values = load_vec<uint4>(x + static_cast<std::int64_t>(col) * k + xk);
            const float2 x0    = bf16x2_bits_to_float2(values.x);
            const float2 x1    = bf16x2_bits_to_float2(values.y);
            const float2 x2    = bf16x2_bits_to_float2(values.z);
            const float2 x3    = bf16x2_bits_to_float2(values.w);
#pragma unroll
            for (int r = 0; r < RowsPerWarp; ++r) {
                if (!row_valid[r]) { continue; }
                acc[r][col] = fmaf(weights[r][0], x0.x, acc[r][col]);
                acc[r][col] = fmaf(weights[r][1], x0.y, acc[r][col]);
                acc[r][col] = fmaf(weights[r][2], x1.x, acc[r][col]);
                acc[r][col] = fmaf(weights[r][3], x1.y, acc[r][col]);
                acc[r][col] = fmaf(weights[r][4], x2.x, acc[r][col]);
                acc[r][col] = fmaf(weights[r][5], x2.y, acc[r][col]);
                acc[r][col] = fmaf(weights[r][6], x3.x, acc[r][col]);
                acc[r][col] = fmaf(weights[r][7], x3.y, acc[r][col]);
            }
        }
    };
    if constexpr (RowsPerWarp >= 4) {
#pragma unroll 1
        for (int phase = 0; phase < phases; ++phase) { phase_step(phase); }
    } else {
#pragma unroll 2
        for (int phase = 0; phase < phases; ++phase) { phase_step(phase); }
    }

#pragma unroll
    for (int r = 0; r < RowsPerWarp; ++r) {
        if (!row_valid[r]) { continue; }
        const int row = row_base + r;
#pragma unroll
        for (int col = 0; col < ColsPerTile; ++col) {
            if (col >= cols) { continue; }
            const float sum = warp_reduce_sum(acc[r][col]);
            if (lane == 0) {
                out[static_cast<std::int64_t>(col) * out_ld + row] = __float2bfloat16(sum);
            }
        }
    }
}

template <int ColsPerTile, int RowsPerWarp>
void launch_q5_rb(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    const std::int32_t rows     = out.ne[0];
    const std::int32_t k        = x.ne[0];
    const std::int32_t out_ld   = static_cast<std::int32_t>(out.nb[1] / sizeof(__nv_bfloat16));
    const std::int32_t padded_k = w.padded_shape[1];
    const dim3 grid(static_cast<unsigned>(div_up(rows, kQ5RbWarps * RowsPerWarp)), 1u, 1u);
    q5_rowsplit_gemm_simt_rb_kernel<ColsPerTile, RowsPerWarp>
        <<<grid, kQ5RbWarps * 32, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(w.qdata),
            static_cast<const std::uint8_t*>(w.qhigh), static_cast<const std::uint8_t*>(w.scales),
            static_cast<__nv_bfloat16*>(out.data), out_ld, rows, k, x.ne[1], padded_k);
    CUDA_CHECK(cudaGetLastError());
}

// NINFER_Q5_RB=2/4/8 → 每 warp 行数；未设/其它值 = 0（不路由）。独立于 NINFER_Q5_DIRECT。
int q5_rb_rows() {
    static const int rb_rows = [] {
        const char* value = std::getenv("NINFER_Q5_RB");
        if (value == nullptr) { return 0; }
        const int v = std::atoi(value);
        return (v == 2 || v == 4 || v == 8) ? v : 0;
    }();
    return rb_rows;
}

template <int ColsPerTile>
void launch_simt(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    const std::int32_t rows     = out.ne[0];
    const std::int32_t k        = x.ne[0];
    const std::int32_t cols     = x.ne[1];
    const std::int32_t out_ld   = static_cast<std::int32_t>(out.nb[1] / sizeof(__nv_bfloat16));
    const std::int32_t padded_k = w.padded_shape[1];
    const auto* xp              = static_cast<const __nv_bfloat16*>(x.data);
    const bool aligned_x = (k % 8) == 0 && (reinterpret_cast<std::uintptr_t>(xp) & 0xfu) == 0;
    const std::int32_t full_slabs = aligned_x ? k / 1024 : 0;
    constexpr int kThreads        = kRowsPerBlock * 32;
    const dim3 grid(static_cast<unsigned>(div_up(rows, kRowsPerBlock)),
                    static_cast<unsigned>(div_up(cols, ColsPerTile)), 1u);
    q5_rowsplit_gemm_simt_kernel<Q5RowSplitSimtSchedule, ColsPerTile, kRowsPerBlock, kStages>
        <<<grid, kThreads, 0, stream>>>(xp, static_cast<const std::uint8_t*>(w.qdata),
                                        static_cast<const std::uint8_t*>(w.qhigh),
                                        static_cast<const std::uint8_t*>(w.scales),
                                        static_cast<__nv_bfloat16*>(out.data), nullptr, rows,
                                        out_ld, k, cols, padded_k, full_slabs);
    CUDA_CHECK(cudaGetLastError());
}

template <int ColsPerTile>
void launch_simt_route(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    for_each_token_slice(x.ne[1], ColsPerTile, [&](std::int32_t offset, std::int32_t count) {
        const Tensor x_slice = x.slice(1, offset, count);
        Tensor out_slice     = out.slice(1, offset, count);
        launch_simt<ColsPerTile>(x_slice, w, out_slice, stream);
    });
}

template <int Cols, int FullSlabs, int Stride>
void launch_split2(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    constexpr int kThreads = 2 * 32;
    const dim3 grid(static_cast<unsigned>(out.ne[0]), 1u, 1u);
    q5_rowsplit_gemm_simt_split2_kernel<Q5RowSplitSimtSchedule, Cols, FullSlabs, Stride>
        <<<grid, kThreads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(w.qdata),
            static_cast<const std::uint8_t*>(w.qhigh), static_cast<const std::uint8_t*>(w.scales),
            static_cast<__nv_bfloat16*>(out.data), out.ne[0], x.ne[0], x.ne[1], w.padded_shape[1],
            FullSlabs);
}

template <int Cols>
void dispatch_split2_cols(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    if (w.k == 6144) {
        launch_split2<Cols, 6, 6144>(x, w, out, stream);
    } else if (w.k == 17408) {
        launch_split2<Cols, 17, 17408>(x, w, out, stream);
    } else {
        throw std::invalid_argument("q5 split2 SIMT: unsupported exact K");
    }
}

template <int Cols>
void launch_split4(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    constexpr int kThreads = 4 * 32;
    const dim3 grid(static_cast<unsigned>(out.ne[0]), 1u, 1u);
    const std::int32_t out_ld = static_cast<std::int32_t>(out.nb[1] / sizeof(__nv_bfloat16));
    q5_rowsplit_gemm_simt_split4_kernel<Q5RowSplitSimtSchedule, Cols, 5, 5120>
        <<<grid, kThreads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(w.qdata),
            static_cast<const std::uint8_t*>(w.qhigh), static_cast<const std::uint8_t*>(w.scales),
            static_cast<__nv_bfloat16*>(out.data), nullptr, out.ne[0], out_ld, x.ne[0], x.ne[1],
            w.padded_shape[1], 5);
}

template <class Launch>
void dispatch_exact_cols(std::int32_t cols, Launch&& launch) {
    switch (cols) {
    case 2:
        launch.template operator()<2>();
        return;
    case 3:
        launch.template operator()<3>();
        return;
    case 4:
        launch.template operator()<4>();
        return;
    case 5:
        launch.template operator()<5>();
        return;
    case 6:
        launch.template operator()<6>();
        return;
    default:
        throw std::invalid_argument("q5 exact SIMT: column count must be in [2,6]");
    }
}

} // namespace

void launch_q5_simt_r8_c4(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    static const bool direct_enabled = [] {
        const char* value = std::getenv("NINFER_Q5_DIRECT");
        return value == nullptr || std::atoi(value) != 0;
    }();
    if (direct_enabled && q5_direct_eligible(x, w)) {
        for_each_token_slice(x.ne[1], 4, [&](std::int32_t offset, std::int32_t count) {
            const Tensor x_slice = x.slice(1, offset, count);
            Tensor out_slice     = out.slice(1, offset, count);
            launch_direct<4>(x_slice, w, out_slice, stream);
        });
        return;
    }
    launch_simt_route<4>(x, w, out, stream);
}

void launch_q5_simt_r8_c8(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    if (q5_rb_rows() != 0 && q5_direct_eligible(x, w)) {
        for_each_token_slice(x.ne[1], 8, [&](std::int32_t offset, std::int32_t count) {
            const Tensor x_slice = x.slice(1, offset, count);
            Tensor out_slice     = out.slice(1, offset, count);
            const int rb         = q5_rb_rows();
            if (rb == 8) {
                launch_q5_rb<8, 8>(x_slice, w, out_slice, stream);
            } else if (rb == 4) {
                launch_q5_rb<8, 4>(x_slice, w, out_slice, stream);
            } else {
                launch_q5_rb<8, 2>(x_slice, w, out_slice, stream);
            }
        });
        return;
    }
    launch_simt_route<8>(x, w, out, stream);
}

void launch_q5_simt_split2_exact(const Tensor& x, const Weight& w, Tensor& out,
                                 cudaStream_t stream) {
    if (q5_rb_rows() != 0 && q5_direct_eligible(x, w)) {
        dispatch_exact_cols(x.ne[1], [&]<int Cols>() {
            const int rb = q5_rb_rows();
            if (rb == 8) {
                launch_q5_rb<Cols, 8>(x, w, out, stream);
            } else if (rb == 4) {
                launch_q5_rb<Cols, 4>(x, w, out, stream);
            } else {
                launch_q5_rb<Cols, 2>(x, w, out, stream);
            }
        });
        return;
    }
    dispatch_exact_cols(x.ne[1],
                        [&]<int Cols>() { dispatch_split2_cols<Cols>(x, w, out, stream); });
    CUDA_CHECK(cudaGetLastError());
}

void launch_q5_simt_split4_exact(const Tensor& x, const Weight& w, Tensor& out,
                                 cudaStream_t stream) {
    dispatch_exact_cols(x.ne[1], [&]<int Cols>() { launch_split4<Cols>(x, w, out, stream); });
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail

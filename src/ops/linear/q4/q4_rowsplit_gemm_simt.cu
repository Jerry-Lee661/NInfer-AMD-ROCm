#include "ops/linear/q4/q4_rowsplit_gemm_simt.cuh"

#include "core/device.h"
#include "ops/common/token_slices.h"
#include "ops/common/math.h"
#include "ops/linear/q4/q4_launch.h"

#include <cstdint>
#include <cstdlib>

namespace ninfer::ops::detail {
namespace {

using Q4SimtR8C4Schedule = Q4RowSplitSimtGemmSchedule<8, 4, 16, 2, Cache::ca, 1>;
using Q4SimtR8C8Schedule = Q4RowSplitSimtGemmSchedule<8, 8, 16, 2, Cache::ca, 1>;

// 直接读寄存器的 T<=4 变体。生产核把 codes 经 LDS 暂存，而 gfx11 的 cp_async 是 4B/lane 的
// LDS DMA（每条 wave32 指令只搬 128 B），加上 cp_wait/__syncwarp 的串行点，在 gate_up 的
// 形状上实测只有 375 GB/s（tools/q4_shape_probe.cu，同形状同布局对拍）；把 codes 直接读进
// 寄存器（q5 split2/split4 已经是这个形状）实测 586-696 GB/s。NINFER_Q4_DIRECT=0 回退到 LDS 版。
constexpr int kDirectRowsPerCta = 8;
constexpr int kDirectPhaseK     = 256;  // 每 lane 每相位 8 个权重（4 字节码）

template <int ColsPerTile>
__global__ __launch_bounds__(kDirectRowsPerCta * 32, 1) void q4_rowsplit_gemm_simt_direct_kernel(
    const __nv_bfloat16* __restrict__ x, const std::uint8_t* __restrict__ codes,
    const std::uint8_t* __restrict__ scales, __nv_bfloat16* __restrict__ out, std::int32_t out_ld,
    std::int32_t rows, std::int32_t k, std::int32_t cols, std::int32_t padded_k) {
    constexpr int kGroupK = Q4RowSplitStorage::kGroupK;

    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int row  = static_cast<int>(blockIdx.x) * kDirectRowsPerCta + warp;
    if (row >= rows) { return; }

    const int groups_per_row = padded_k / kGroupK;
    const std::uint8_t* code_row =
        codes + static_cast<std::int64_t>(row) * groups_per_row * (kGroupK / 2);
    const std::uint8_t* scale_row = scales + static_cast<std::int64_t>(row) * groups_per_row * 2;

    float acc[ColsPerTile];
#pragma unroll
    for (int col = 0; col < ColsPerTile; ++col) { acc[col] = 0.0f; }

    const int phases = k / kDirectPhaseK;
#pragma unroll 2
    for (int phase = 0; phase < phases; ++phase) {
        const std::uint32_t packed = *reinterpret_cast<const std::uint32_t*>(
            code_row + static_cast<std::int64_t>(phase) * (kDirectPhaseK / 2) + lane * 4);
        const std::uint16_t scale_bits = *reinterpret_cast<const std::uint16_t*>(
            scale_row + (phase * 4 + (lane >> 3)) * 2);

        float weights[8];
        Q4SimtDecodeAtom::decode_eight(packed, scale_bits, weights);

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

    // 一条 warp 独占一行，32 条 lane 覆盖互不重叠的 K 片段，行内归约收尾。
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
    q4_rowsplit_gemm_simt_direct_kernel<ColsPerTile>
        <<<grid, kDirectRowsPerCta * 32, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(w.qdata),
            static_cast<const std::uint8_t*>(w.scales), static_cast<__nv_bfloat16*>(out.data),
            out_ld, rows, k, x.ne[1], padded_k);
    CUDA_CHECK(cudaGetLastError());
}

// 只接管整块形状：K 必须是相位整数倍（否则尾部组会被漏掉），列按 4 切分。
bool q4_direct_eligible(const Tensor& x, const Weight& w) {
    const std::int32_t k        = x.ne[0];
    const std::int32_t padded_k = w.padded_shape[1];
    return k > 0 && (k % kDirectPhaseK) == 0 && padded_k >= k && (padded_k % 64) == 0;
}

template <class Schedule, bool Full>
void launch_schedule(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    const std::int32_t rows     = out.ne[0];
    const std::int32_t k        = x.ne[0];
    const std::int32_t cols     = x.ne[1];
    const std::int32_t out_ld   = static_cast<std::int32_t>(out.nb[1] / sizeof(__nv_bfloat16));
    const std::int32_t padded_k = w.padded_shape[1];

    const dim3 grid(static_cast<unsigned>(div_up(rows, Schedule::kRowsPerCta)),
                    static_cast<unsigned>(div_up(cols, Schedule::kColsPerTile)), 1u);

    q4_rowsplit_gemm_simt_kernel<Schedule, Full><<<grid, Schedule::kThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(w.qdata),
        static_cast<const std::uint8_t*>(w.scales), static_cast<__nv_bfloat16*>(out.data), nullptr,
        out_ld, 0, rows, k, cols, padded_k);
    CUDA_CHECK(cudaGetLastError());
}

template <class Schedule>
void launch_route(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    const bool full = (out.ne[0] % Schedule::kRowsPerCta) == 0 &&
                      ((x.ne[0] / Q4RowSplitStorage::kGroupK) % Schedule::kGroupsPerStage) == 0 &&
                      (x.ne[1] % Schedule::kColsPerTile) == 0;
    for_each_token_slice(x.ne[1], Schedule::kColsPerTile,
                         [&](std::int32_t offset, std::int32_t count) {
                             const Tensor x_slice = x.slice(1, offset, count);
                             Tensor out_slice     = out.slice(1, offset, count);
                             if (full) {
                                 launch_schedule<Schedule, true>(x_slice, w, out_slice, stream);
                             } else {
                                 launch_schedule<Schedule, false>(x_slice, w, out_slice, stream);
                             }
                         });
}

} // namespace

void launch_q4_simt_r8_c4(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    // decode 的 T<=4 档是 gate_up（6.06 GB/token）与 GDN/attention 投影的必经之路，占每 token
    // 权重流的绝大部分；直接读寄存器版本在同形状下快 1.6 倍，所以默认走它。
    static const bool direct_enabled = [] {
        const char* value = std::getenv("NINFER_Q4_DIRECT");
        return value == nullptr || std::atoi(value) != 0;
    }();
    if (direct_enabled && q4_direct_eligible(x, w)) {
        for_each_token_slice(x.ne[1], 4, [&](std::int32_t offset, std::int32_t count) {
            const Tensor x_slice = x.slice(1, offset, count);
            Tensor out_slice     = out.slice(1, offset, count);
            launch_direct<4>(x_slice, w, out_slice, stream);
        });
        return;
    }
    launch_route<Q4SimtR8C4Schedule>(x, w, out, stream);
}

void launch_q4_simt_r8_c8(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    launch_route<Q4SimtR8C8Schedule>(x, w, out, stream);
}

} // namespace ninfer::ops::detail

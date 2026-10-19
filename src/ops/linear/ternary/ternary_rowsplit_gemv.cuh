#pragma once

// Decode-shaped ternary GEMV (T == 1) for PQ2_0: one warp owns one output row.
//
// Why: the reference kernel in ternary_rowsplit_gemm.cuh gives each output row a full 128-thread
// CTA and seven block-wide barriers. That is fine for correctness but leaves decode far from the
// card's measured bandwidth. Here each lane covers four consecutive weights of every 128-group, so
// a warp reads exactly one 32-byte code span plus one 2-byte scale per group and reduces through
// shuffles with no __syncthreads at all.
//
// PQ2_0 packing makes the mapping exact rather than approximate: a group is 32 bytes holding four
// two-bit codes each, so for lane l the four columns 4l..4l+3 live entirely in byte l. Lane l
// loads one byte, decodes four weights, and consumes four bf16 activations.
//
// Occupancy, not instruction count, is what this kernel is tuned for: a GEMV needs many warps in
// flight to cover DRAM latency. A two-rows-per-warp variant (which reuses the activation loads)
// measured SLOWER (32.8 vs 46.4 t/s) because the extra accumulators and row bases cost registers
// and dropped the resident warp count. So: one row per warp, and only the micro-optimisations that
// remove instructions without adding state -- one 16-bit scale load, paired bf16 activation
// loads, and the per-group scale multiply hoisted out of the four FMAs.
//
// LAYOUT: ninfer/ggml put ne[0] on the contiguous axis, so a [k, 1] activation keeps element
// (column, 0) at column, and a one-token output row is simply out[row].

// ternary_pow3() and the already-verified PTQ1_0 decode live here; PTQ1 gets its weight fetch from
// that single source of truth instead of a second hand-written copy of the trit formula.
#include "ops/linear/ternary/ternary_rowsplit_storage.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace ninfer::ops::detail {

// Warps per block for the GEMV below.
inline constexpr int kGemvWarpsPerBlock = 8;

inline constexpr int kGemvCodeBytesPerGroup  = 32;
inline constexpr int kGemvScaleBytesPerGroup = 2;
inline constexpr int kGemvGroupK             = 128;

__device__ __forceinline__ float gemv_scale(const std::uint8_t* scale_ptr) {
    // One 16-bit load instead of two byte loads plus a shift/or; the scale plane is 2-byte aligned.
    const std::uint16_t bits = *reinterpret_cast<const std::uint16_t*>(scale_ptr);
    return __half2float(__ushort_as_half(bits));
}

#if defined(NINFER_HIP)
// ---- dp4a 配方(NINFER_PQ2_DP4A=1, PrismML mmvq 同款, P-A 探针定案 1.54×) ----
// 激活 q8_1 预量化(int8 + 每 32 元素 fp16 scale), 权重码字节 v_perm 解码, dp4a 整数点积。
// 只在 HIP 构建存在: __builtin_amdgcn_perm/sudot4 是 AMDGCN 内建, nvcc 侧无对应物。

__device__ __forceinline__ int pq2_dp4a_symbols4(unsigned int b) {
    const unsigned int y = (b & 0x33u) | ((b & 0xCCu) << 6);
    const unsigned int z = (y & 0x0303u) | ((y & 0x3030u) << 12);
    return static_cast<int>(__builtin_amdgcn_perm(0x020100FFu, 0x020100FFu, z));
}

__device__ __forceinline__ int pq2_dp4a_dot4(int a, int b, int c) {
    return __builtin_amdgcn_sudot4(true, a, true, b, c, false);
}

// 激活量化 pass: 一线程一个 32 子块, absmax/127 → int8 + fp16 scale。
// blockIdx.y = token 行(T=1 decode 或 T=2..4 verify tile), xq/ds8 按 token 行主存放。
__global__ __launch_bounds__(256)
void ternary_pq2_dp4a_quant_kernel(const __nv_bfloat16* __restrict__ x,
                                   std::int8_t* __restrict__ xq,
                                   std::uint16_t* __restrict__ ds8, std::int32_t k) {
    const std::int32_t g = static_cast<std::int32_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::int32_t t = static_cast<std::int32_t>(blockIdx.y);
    if (g * 32 >= k) { return; }
    const __nv_bfloat16* xt = x + static_cast<std::int64_t>(t) * k;
    float amax = 0.0f;
#pragma unroll 4
    for (int i = 0; i < 32; ++i) {
        amax = fmaxf(amax, fabsf(__bfloat162float(xt[g * 32 + i])));
    }
    const float d   = amax / 127.0f;
    const float inv = d > 0.0f ? 1.0f / d : 0.0f;
#pragma unroll 4
    for (int i = 0; i < 32; ++i) {
        const int q = std::lrint(__bfloat162float(xt[g * 32 + i]) * inv);
        xq[static_cast<std::int64_t>(t) * k + g * 32 + i] =
            static_cast<std::int8_t>(std::max(-127, std::min(127, q)));
    }
    ds8[static_cast<std::int64_t>(t) * (k / 32) + g] = __half_as_ushort(__float2half(d));
}

// T=1 decode GEMV 的 dp4a 版: 与 ternary_pq2_gemv_kernel 同骨架(warp 一行, shuffle 归约,
// 无 barrier), 内循环从 ~5.5 ALU/权重(bf16 读+手工解码+fp32 fma)换到 ~2.5(1B 码 +
// v_perm + 1×u32 int8 激活 + dp4a + scale 合并)。
__global__ __launch_bounds__(kGemvWarpsPerBlock * 32)
void ternary_pq2_gemv_dp4a_kernel(const std::int8_t* __restrict__ xq,
                                  const std::uint16_t* __restrict__ ds8,
                                  const std::uint8_t* __restrict__ codes,
                                  const std::uint8_t* __restrict__ scales,
                                  __nv_bfloat16* __restrict__ out, std::int32_t rows,
                                  std::int32_t groups_per_row,
                                  const __nv_bfloat16* residual = nullptr) {
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp =
        static_cast<int>(blockIdx.x) * kGemvWarpsPerBlock + (static_cast<int>(threadIdx.x) >> 5);
    if (warp >= rows) { return; }
    const std::uint8_t* code_row =
        codes + static_cast<std::int64_t>(warp) * groups_per_row * kGemvCodeBytesPerGroup;
    const std::uint8_t* scale_row =
        scales + static_cast<std::int64_t>(warp) * groups_per_row * kGemvScaleBytesPerGroup;

    // M2 刀二: 残差 epilogue(ternary linear_add 专用)。residual 与 out 可能同址(原地
    // residual += W*x), 所以 residual 不带 __restrict__, 且在写 out 之前先读进寄存器 ——
    // 编译器因此必须假设两者别名, 不会把读排到写后面。
    float residual_value = 0.0f;
    if (residual != nullptr && lane == 0) { residual_value = __bfloat162float(residual[warp]); }

    float accumulator = 0.0f;
#pragma unroll 8
    for (int group = 0; group < groups_per_row; ++group) {
        const std::uint8_t raw = code_row[group * kGemvCodeBytesPerGroup + lane];
        const int w4           = pq2_dp4a_symbols4(raw);
        const int x4           = *reinterpret_cast<const int*>(
            xq + static_cast<std::int64_t>(group) * kGemvGroupK + lane * 4);
        const int dot = pq2_dp4a_dot4(w4, x4, 0);
        const float wsc =
            gemv_scale(scale_row + group * kGemvScaleBytesPerGroup) *
            __half2float(__ushort_as_half(ds8[group * 4 + (lane >> 3)]));
        accumulator = fmaf(wsc, static_cast<float>(dot), accumulator);
    }
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        accumulator += __shfl_xor_sync(0xffffffffffffffffULL, accumulator, offset);
    }
    if (lane == 0) {
        // 舍入点与 compose 路径逐位一致: 先舍 bf16 投影结果, 再加残差, 再舍一次
        // (residual_add_pair 的 x+y 在浮点上与 y+x 同值)。
        const __nv_bfloat16 projected = __float2bfloat16_rn(accumulator);
        out[warp] = residual == nullptr
                        ? projected
                        : __float2bfloat16_rn(residual_value + __bfloat162float(projected));
    }
}

// x-LDS 共享臂(NINFER_GEMV_XLDS=1, 默认关, 2026-10-05 P-D): T=1 PQ2 GEMV 的
// CTA 级 x 重排。现行 8 warp/CTA 各自全局直读同一 x(重复 8x, 占 L1 与发射槽);
// 本臂 32 warp/CTA, 启动先全量装 x 进 shared(k=5120 时 10KB), 各 warp 从 LDS 读。
// 探针(probe_xlds.cu): 位级同值, [262144,5120] 码流 423→495 GB/s(+17%)。
constexpr int kGemvXLdsWarpsPerBlock = 32;

__global__ __launch_bounds__(kGemvXLdsWarpsPerBlock * 32)
void ternary_pq2_gemv_xlds_kernel(const __nv_bfloat16* __restrict__ x,
                                  const std::uint8_t* __restrict__ codes,
                                  const std::uint8_t* __restrict__ scales,
                                  __nv_bfloat16* __restrict__ out, std::int32_t rows,
                                  std::int32_t groups_per_row, std::int32_t k) {
    __shared__ __nv_bfloat16 xs[5120];
    static_assert(kGemvXLdsWarpsPerBlock * 32 >= 5120 / 8, "cta too small to stage x");
    const int tid = static_cast<int>(threadIdx.x);
    for (int i = tid; i < k; i += kGemvXLdsWarpsPerBlock * 32) { xs[i] = x[i]; }
    __syncthreads();

    const int lane = tid & 31;
    const int warp = static_cast<int>(blockIdx.x) * kGemvXLdsWarpsPerBlock + (tid >> 5);
    if (warp >= rows) { return; }
    const std::uint8_t* code_row =
        codes + static_cast<std::int64_t>(warp) * groups_per_row * kGemvCodeBytesPerGroup;
    const std::uint8_t* scale_row =
        scales + static_cast<std::int64_t>(warp) * groups_per_row * kGemvScaleBytesPerGroup;

    float accumulator = 0.0f;
#pragma unroll 8
    for (int group = 0; group < groups_per_row; ++group) {
        const std::int32_t base = group * kGemvGroupK + lane * 4;
        const float2 low  = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(xs + base));
        const float2 high = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(xs + base + 2));
        const std::uint8_t raw = code_row[group * kGemvCodeBytesPerGroup + lane];
        const float dot = fmaf(static_cast<float>(static_cast<int>(raw & 3u) - 1), low.x,
                               fmaf(static_cast<float>(static_cast<int>((raw >> 2) & 3u) - 1),
                                    low.y,
                                    fmaf(static_cast<float>(static_cast<int>((raw >> 4) & 3u) - 1),
                                         high.x,
                                         static_cast<float>(static_cast<int>((raw >> 6) & 3u) - 1) *
                                             high.y)));
        accumulator = fmaf(gemv_scale(scale_row + group * kGemvScaleBytesPerGroup), dot,
                           accumulator);
    }
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        accumulator += __shfl_xor_sync(0xffffffffffffffffULL, accumulator, offset);
    }
    if (lane == 0) { out[warp] = __float2bfloat16_rn(accumulator); }
}

// T=2..kCols verify tile 的 dp4a 版: 权重码字节每组解码一次(v_perm), 每 token 一条 dp4a。
// 与 bf16 tile 核同骨架: 权重单次读, 输出 [t * out_row_stride + row]。
// kCols = 列宽模板(P0b, 2026-10-18: 4→6 放开 rung); 发射点按 tokens 选宽, ≤4 的实例
// 与旧核逐字同(kCols=4 实例保持原 codegen)。
template <int kCols>
__global__ __launch_bounds__(kGemvWarpsPerBlock * 32)
void ternary_pq2_gemv_tile_dp4a_kernel(const std::int8_t* __restrict__ xq,
                                       const std::uint16_t* __restrict__ ds8,
                                       const std::uint8_t* __restrict__ codes,
                                       const std::uint8_t* __restrict__ scales,
                                       __nv_bfloat16* __restrict__ out, std::int32_t rows,
                                       std::int32_t groups_per_row, std::int32_t tokens,
                                       std::int32_t out_row_stride,
                                       const __nv_bfloat16* residual = nullptr) {
    static_assert(kCols >= 1 && kCols <= 8, "tile size must fit the register accumulators");
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp =
        static_cast<int>(blockIdx.x) * kGemvWarpsPerBlock + (static_cast<int>(threadIdx.x) >> 5);
    if (warp >= rows) { return; }
    const std::uint8_t* code_row =
        codes + static_cast<std::int64_t>(warp) * groups_per_row * kGemvCodeBytesPerGroup;
    const std::uint8_t* scale_row =
        scales + static_cast<std::int64_t>(warp) * groups_per_row * kGemvScaleBytesPerGroup;

    // M2 刀二: 残差 epilogue(同 T=1 核: 先读后写, residual 不带 __restrict__)。
    float residual_value[kCols];
#pragma unroll
    for (int t = 0; t < kCols; ++t) { residual_value[t] = 0.0f; }
    if (residual != nullptr && lane == 0) {
#pragma unroll
        for (int t = 0; t < kCols; ++t) {
            if (t < tokens) {
                residual_value[t] =
                    __bfloat162float(residual[static_cast<std::int64_t>(t) * out_row_stride + warp]);
            }
        }
    }

    float accumulator[kCols];
#pragma unroll
    for (int t = 0; t < kCols; ++t) { accumulator[t] = 0.0f; }

    for (int group = 0; group < groups_per_row; ++group) {
        const std::uint8_t raw = code_row[group * kGemvCodeBytesPerGroup + lane];
        const int w4           = pq2_dp4a_symbols4(raw);
        const float wsc        = gemv_scale(scale_row + group * kGemvScaleBytesPerGroup);
        const int sub          = group * 4 + (lane >> 3);
        const std::int64_t xbase =
            static_cast<std::int64_t>(group) * kGemvGroupK + lane * 4;
#pragma unroll
        for (int t = 0; t < kCols; ++t) {
            if (t < tokens) {
                const int x4 = *reinterpret_cast<const int*>(
                    xq + static_cast<std::int64_t>(t) * groups_per_row * kGemvGroupK + xbase);
                const float dsw =
                    wsc * __half2float(__ushort_as_half(
                              ds8[static_cast<std::int64_t>(t) * groups_per_row * 4 + sub]));
                accumulator[t] =
                    fmaf(dsw, static_cast<float>(pq2_dp4a_dot4(w4, x4, 0)), accumulator[t]);
            }
        }
    }

#pragma unroll
    for (int t = 0; t < kCols; ++t) {
        float value = accumulator[t];
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            value += __shfl_xor_sync(0xffffffffffffffffULL, value, offset);
        }
        if (lane == 0 && t < tokens) {
            // 舍入点与 compose 路径逐位一致(同 T=1 核的 epilogue)。
            const __nv_bfloat16 projected = __float2bfloat16_rn(value);
            out[static_cast<std::int64_t>(t) * out_row_stride + warp] =
                residual == nullptr
                    ? projected
                    : __float2bfloat16_rn(residual_value[t] + __bfloat162float(projected));
        }
    }
}

// 双段合并(2026-10-08, megakernel M2c): 同一父权重的两个行区一段发射(attn q|k、gdn value|z),
// 消掉一半 launch。逐发带宽随发射规模爬坡(冷 DRAM 探针:1024 行 164 GB/s → 17408 行 591;
// 引擎每发多在 1-6K 行坏区),合并 = 直接把 GEMV 家族推上大尺寸臂。逐行数学与分片发射
// 逐位一致(同码流同 scale 同累加序), md5 门直接判。
// rows/epilogue: warp 行号 < rowsA 写 (outA, strideA), 否则写 (outB, strideB) 偏移 rowsA。
//
// ⚠️ T=1 必须走下面 seg2_t1(专用单累加器 + group unroll-8)而不是这个 tile 结构:
// 探针实测 tile 结构在 T=1 比专用核慢 2.7-2.9×(7168 行 54.7 vs 20.2µs)——第一版 seg2
// 因此判负(-9.5%)。tile 结构只服务 T=2..kCols(每权重字节摊 kCols 份算术,原设计域)。
template <int kCols>
__global__ __launch_bounds__(kGemvWarpsPerBlock * 32)
void ternary_pq2_gemv_tile_dp4a_seg2_kernel(const std::int8_t* __restrict__ xq,
                                            const std::uint16_t* __restrict__ ds8,
                                            const std::uint8_t* __restrict__ codes,
                                            const std::uint8_t* __restrict__ scales,
                                            __nv_bfloat16* __restrict__ out_a,
                                            std::int32_t rows_a, std::int32_t stride_a,
                                            __nv_bfloat16* __restrict__ out_b,
                                            std::int32_t stride_b, std::int32_t rows,
                                            std::int32_t groups_per_row, std::int32_t tokens) {
    static_assert(kCols >= 1 && kCols <= 8, "tile size must fit the register accumulators");
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp =
        static_cast<int>(blockIdx.x) * kGemvWarpsPerBlock + (static_cast<int>(threadIdx.x) >> 5);
    if (warp >= rows) { return; }
    const std::uint8_t* code_row =
        codes + static_cast<std::int64_t>(warp) * groups_per_row * kGemvCodeBytesPerGroup;
    const std::uint8_t* scale_row =
        scales + static_cast<std::int64_t>(warp) * groups_per_row * kGemvScaleBytesPerGroup;

    float accumulator[kCols];
#pragma unroll
    for (int t = 0; t < kCols; ++t) { accumulator[t] = 0.0f; }

    for (int group = 0; group < groups_per_row; ++group) {
        const std::uint8_t raw = code_row[group * kGemvCodeBytesPerGroup + lane];
        const int w4           = pq2_dp4a_symbols4(raw);
        const float wsc        = gemv_scale(scale_row + group * kGemvScaleBytesPerGroup);
        const int sub          = group * 4 + (lane >> 3);
        const std::int64_t xbase =
            static_cast<std::int64_t>(group) * kGemvGroupK + lane * 4;
#pragma unroll
        for (int t = 0; t < kCols; ++t) {
            if (t < tokens) {
                const int x4 = *reinterpret_cast<const int*>(
                    xq + static_cast<std::int64_t>(t) * groups_per_row * kGemvGroupK + xbase);
                const float dsw =
                    wsc * __half2float(__ushort_as_half(
                              ds8[static_cast<std::int64_t>(t) * groups_per_row * 4 + sub]));
                accumulator[t] =
                    fmaf(dsw, static_cast<float>(pq2_dp4a_dot4(w4, x4, 0)), accumulator[t]);
            }
        }
    }

    const bool in_a           = warp < rows_a;
    const std::int32_t segRow = in_a ? warp : warp - rows_a;
    __nv_bfloat16* out        = in_a ? out_a : out_b;
    const std::int32_t stride = in_a ? stride_a : stride_b;
#pragma unroll
    for (int t = 0; t < kCols; ++t) {
        float value = accumulator[t];
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            value += __shfl_xor_sync(0xffffffffffffffffULL, value, offset);
        }
        if (lane == 0 && t < tokens) {
            out[static_cast<std::int64_t>(t) * stride + segRow] = __float2bfloat16_rn(value);
        }
    }
}

// seg2 的 T=1 专用形态: 与 ternary_pq2_gemv_dp4a_kernel 同构(单累加器、group unroll-8、
// 无 t 环), 仅 epilogue 换成段选。上面 tile 形态在 T=1 慢 2.7-2.9×(探针实测), 别用。

// segn 表驱动合并(2026-10-08, megakernel M2d): seg2 的推广 —— 最多 4 段、各段可有
// 不同的权重基址(跨父权重), 一段发射。结构 = 已探针定案的 T=1 专用形态(单累加器 +
// group unroll-8), 相对专用核只多 3 次整数比较(段查找);16384 行落在 585 GB/s 大尺寸臂。
// 表按值传参(图捕获时烙进 kernel args, 地址稳定)。仅 T=1。
// 表约定:row_begin[0]=0,row_begin[i]=第 i 段起点;[nsegs..3] 填 INT_MAX(查找永不选中);
// 总行数单独传参做 grid 覆盖判断。
struct TernarySegTable {
    const std::uint8_t* codes[4];
    const std::uint8_t* scales[4];
    __nv_bfloat16*      out[4];
    std::int32_t        row_begin[4];
    std::int32_t        nsegs;
};
__global__ __launch_bounds__(kGemvWarpsPerBlock * 32)
void ternary_pq2_gemv_dp4a_segn_t1_kernel(const std::int8_t* __restrict__ xq,
                                          const std::uint16_t* __restrict__ ds8,
                                          std::int32_t groups_per_row, std::int32_t rows,
                                          TernarySegTable table) {
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp =
        static_cast<int>(blockIdx.x) * kGemvWarpsPerBlock + (static_cast<int>(threadIdx.x) >> 5);
    if (warp >= rows) { return; }
    std::int32_t seg = 0;
#pragma unroll
    for (int i = 1; i < 4; ++i) {
        if (warp >= table.row_begin[i]) { seg = i; }
    }
    const std::int32_t local = warp - table.row_begin[seg];
    const std::uint8_t* code_row =
        table.codes[seg] + static_cast<std::int64_t>(local) * groups_per_row * kGemvCodeBytesPerGroup;
    const std::uint8_t* scale_row =
        table.scales[seg] + static_cast<std::int64_t>(local) * groups_per_row * kGemvScaleBytesPerGroup;

    float accumulator = 0.0f;
#pragma unroll 8
    for (int group = 0; group < groups_per_row; ++group) {
        const std::uint8_t raw = code_row[group * kGemvCodeBytesPerGroup + lane];
        const int w4           = pq2_dp4a_symbols4(raw);
        const int x4           = *reinterpret_cast<const int*>(
            xq + static_cast<std::int64_t>(group) * kGemvGroupK + lane * 4);
        const int dot = pq2_dp4a_dot4(w4, x4, 0);
        const float wsc =
            gemv_scale(scale_row + group * kGemvScaleBytesPerGroup) *
            __half2float(__ushort_as_half(ds8[group * 4 + (lane >> 3)]));
        accumulator = fmaf(wsc, static_cast<float>(dot), accumulator);
    }
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        accumulator += __shfl_xor_sync(0xffffffffffffffffULL, accumulator, offset);
    }
    if (lane == 0) { table.out[seg][local] = __float2bfloat16_rn(accumulator); }
}

__global__ __launch_bounds__(kGemvWarpsPerBlock * 32)
void ternary_pq2_gemv_dp4a_seg2_t1_kernel(const std::int8_t* __restrict__ xq,
                                          const std::uint16_t* __restrict__ ds8,
                                          const std::uint8_t* __restrict__ codes,
                                          const std::uint8_t* __restrict__ scales,
                                          __nv_bfloat16* __restrict__ out_a, std::int32_t rows_a,
                                          std::int32_t stride_a, __nv_bfloat16* __restrict__ out_b,
                                          std::int32_t stride_b, std::int32_t rows,
                                          std::int32_t groups_per_row) {
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp =
        static_cast<int>(blockIdx.x) * kGemvWarpsPerBlock + (static_cast<int>(threadIdx.x) >> 5);
    if (warp >= rows) { return; }
    const std::uint8_t* code_row =
        codes + static_cast<std::int64_t>(warp) * groups_per_row * kGemvCodeBytesPerGroup;
    const std::uint8_t* scale_row =
        scales + static_cast<std::int64_t>(warp) * groups_per_row * kGemvScaleBytesPerGroup;

    float accumulator = 0.0f;
#pragma unroll 8
    for (int group = 0; group < groups_per_row; ++group) {
        const std::uint8_t raw = code_row[group * kGemvCodeBytesPerGroup + lane];
        const int w4           = pq2_dp4a_symbols4(raw);
        const int x4           = *reinterpret_cast<const int*>(
            xq + static_cast<std::int64_t>(group) * kGemvGroupK + lane * 4);
        const int dot = pq2_dp4a_dot4(w4, x4, 0);
        const float wsc =
            gemv_scale(scale_row + group * kGemvScaleBytesPerGroup) *
            __half2float(__ushort_as_half(ds8[group * 4 + (lane >> 3)]));
        accumulator = fmaf(wsc, static_cast<float>(dot), accumulator);
    }
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        accumulator += __shfl_xor_sync(0xffffffffffffffffULL, accumulator, offset);
    }
    if (lane == 0) {
        const bool in_a           = warp < rows_a;
        const std::int32_t segRow = in_a ? warp : warp - rows_a;
        __nv_bfloat16* out        = in_a ? out_a : out_b;
        const std::int32_t stride = in_a ? stride_a : stride_b;
        out[segRow] = __float2bfloat16_rn(accumulator);
    }
}
#endif  // NINFER_HIP

__global__ __launch_bounds__(kGemvWarpsPerBlock * 32)
void ternary_pq2_gemv_kernel(const __nv_bfloat16* __restrict__ x,
                             const std::uint8_t* __restrict__ codes,
                             const std::uint8_t* __restrict__ scales,
                             __nv_bfloat16* __restrict__ out, std::int32_t rows,
                             std::int32_t groups_per_row) {
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp =
        static_cast<int>(blockIdx.x) * kGemvWarpsPerBlock + (static_cast<int>(threadIdx.x) >> 5);
    if (warp >= rows) { return; }

    const std::uint8_t* code_row =
        codes + static_cast<std::int64_t>(warp) * groups_per_row * kGemvCodeBytesPerGroup;
    const std::uint8_t* scale_row =
        scales + static_cast<std::int64_t>(warp) * groups_per_row * kGemvScaleBytesPerGroup;

    float accumulator = 0.0f;

#pragma unroll 4
    for (int group = 0; group < groups_per_row; ++group) {
        const std::int32_t base = group * kGemvGroupK + lane * 4;
        const float2 low        = __bfloat1622float2(
            *reinterpret_cast<const __nv_bfloat162*>(x + base));
        const float2 high = __bfloat1622float2(
            *reinterpret_cast<const __nv_bfloat162*>(x + base + 2));

        const std::uint8_t raw = code_row[group * kGemvCodeBytesPerGroup + lane];
        const float dot = fmaf(static_cast<float>(static_cast<int>(raw & 3u) - 1), low.x,
                               fmaf(static_cast<float>(static_cast<int>((raw >> 2) & 3u) - 1),
                                    low.y,
                                    fmaf(static_cast<float>(static_cast<int>((raw >> 4) & 3u) - 1),
                                         high.x,
                                         static_cast<float>(static_cast<int>((raw >> 6) & 3u) - 1) *
                                             high.y)));
        accumulator = fmaf(gemv_scale(scale_row + group * kGemvScaleBytesPerGroup), dot,
                           accumulator);
    }

#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        accumulator +=  __shfl_down_sync(0xffffffffffffffffULL, accumulator, offset);
    }
    if (lane == 0) { out[warp] = __float2bfloat16_rn(accumulator); }
}

// Small-token-tile variant, for the speculative VERIFY pass (T = draft + 1, i.e. 2..4).
//
// A per-token GEMV would re-read every weight for every token, and weights are exactly what the
// decode path is bound by, so a verify round would cost T full decode passes and speculation could
// never pay for itself (measured: 15.7 t/s with MTP N=2 against 62.1 t/s without). This kernel
// keeps the weights single-read: the code byte and scale are loaded once per group and reused for
// all kT tokens, while each token contributes its own four activations.
template <int kT>
__global__ __launch_bounds__(kGemvWarpsPerBlock * 32)
void ternary_pq2_gemv_tile_kernel(const __nv_bfloat16* __restrict__ x,
                                  const std::uint8_t* __restrict__ codes,
                                  const std::uint8_t* __restrict__ scales,
                                  __nv_bfloat16* __restrict__ out, std::int32_t rows,
                                  std::int32_t groups_per_row, std::int32_t tokens,
                                  std::int32_t out_row_stride) {
    static_assert(kT >= 1 && kT <= 8, "tile size must be small enough to keep accumulators in registers");
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp =
        static_cast<int>(blockIdx.x) * kGemvWarpsPerBlock + (static_cast<int>(threadIdx.x) >> 5);
    if (warp >= rows) { return; }

    const std::uint8_t* code_row =
        codes + static_cast<std::int64_t>(warp) * groups_per_row * kGemvCodeBytesPerGroup;
    const std::uint8_t* scale_row =
        scales + static_cast<std::int64_t>(warp) * groups_per_row * kGemvScaleBytesPerGroup;

    float accumulator[kT];
#pragma unroll
    for (int t = 0; t < kT; ++t) { accumulator[t] = 0.0f; }

    for (int group = 0; group < groups_per_row; ++group) {
        // Weights: one byte of codes + one 16-bit scale, reused across every token in the tile.
        const std::uint8_t raw = code_row[group * kGemvCodeBytesPerGroup + lane];
        const float scale = gemv_scale(scale_row + group * kGemvScaleBytesPerGroup);
        const float weight0 = static_cast<float>(static_cast<int>(raw & 3u) - 1);
        const float weight1 = static_cast<float>(static_cast<int>((raw >> 2) & 3u) - 1);
        const float weight2 = static_cast<float>(static_cast<int>((raw >> 4) & 3u) - 1);
        const float weight3 = static_cast<float>(static_cast<int>((raw >> 6) & 3u) - 1);

        const std::int32_t base = group * kGemvGroupK + lane * 4;
#pragma unroll
        for (int t = 0; t < kT; ++t) {
            if (t < tokens) {
                const __nv_bfloat16* x_token =
                    x + static_cast<std::int64_t>(t) * groups_per_row * kGemvGroupK;
                const float2 low = __bfloat1622float2(
                    *reinterpret_cast<const __nv_bfloat162*>(x_token + base));
                const float2 high = __bfloat1622float2(
                    *reinterpret_cast<const __nv_bfloat162*>(x_token + base + 2));
                const float dot = fmaf(weight0, low.x,
                                       fmaf(weight1, low.y, fmaf(weight2, high.x, weight3 * high.y)));
                accumulator[t] = fmaf(scale, dot, accumulator[t]);
            }
        }
    }

#pragma unroll
    for (int t = 0; t < kT; ++t) {
        float value = accumulator[t];
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            value +=  __shfl_down_sync(0xffffffffffffffffULL, value, offset);
        }
        if (lane == 0 && t < tokens) {
            // Token-major output: element (row, token) lives at token * out_row_stride + row.
            out[static_cast<std::int64_t>(t) * out_row_stride + warp] =
                __float2bfloat16_rn(value);
        }
    }
}

// One lane owns four consecutive columns (4l..4l+3). For PTQ1_0 those four always fall inside ONE
// region, and inside a region the trit multiplier is a constant OF THE LANE, not of the column --
// so the region branch and the 3^n chain are hoisted out of the group loop and the inner loop keeps
// only a multiply, a uint8 wrap, a shift and a subtract per weight. That matters here because the
// PTQ1_0 unpack is instruction-bound rather than bandwidth-bound (the PQ2_0 "one lane owns one code
// byte" trick does not transfer, since a base-3 byte carries parts of five different columns):
//   region 0: weights   0..79 : byte codes[j & 15],            trit = j >> 4      (c=16 stage)
//   region 1: weights  80..119: byte codes[16 + ((j-80) & 7)], trit = (j-80) >> 3 (c=8 stage)
//   region 2: weights 120..127: bytes high[0],high[1],high[0],high[1] with trits t,t,t+1,t+1
inline constexpr int kPtq1CodeBytesPerGroup = 24;
inline constexpr int kPtq1HighBytesPerGroup = 2;

struct Ptq1LanePlan {
    const std::uint8_t* row;      // code plane row, or high plane row in region 2
    int bytes_per_group;
    int src_off;                  // byte offset of the lane's first weight inside the group
    int pow_a;                    // 3^trit for columns 0 and 1
    int pow_b;                    // 3^trit for columns 2 and 3 (region 2 only; == pow_a otherwise)
    int idx0, idx1, idx2, idx3;   // byte index of each column inside the lane's span
};

__device__ __forceinline__ Ptq1LanePlan ptq1_lane_plan(int lane,
                                                       const std::uint8_t* code_row,
                                                       const std::uint8_t* high_row) {
    Ptq1LanePlan p{};
    const int col0 = lane * 4;
    if (lane < 20) {
        p.row = code_row;
        p.bytes_per_group = kPtq1CodeBytesPerGroup;
        p.src_off         = col0 & 15;
        p.pow_a = p.pow_b = ternary_pow3(col0 >> 4);
        p.idx0 = 0; p.idx1 = 1; p.idx2 = 2; p.idx3 = 3;
    } else if (lane < 30) {
        const int local = col0 - 80;
        p.row = code_row;
        p.bytes_per_group = kPtq1CodeBytesPerGroup;
        p.src_off         = 16 + (local & 7);
        p.pow_a = p.pow_b = ternary_pow3(local >> 3);
        p.idx0 = 0; p.idx1 = 1; p.idx2 = 2; p.idx3 = 3;
    } else {
        const int local = col0 - 120;
        p.row = high_row;
        p.bytes_per_group = kPtq1HighBytesPerGroup;
        p.src_off         = 0;
        p.pow_a           = ternary_pow3(local >> 1);
        p.pow_b           = ternary_pow3((local >> 1) + 1);
        p.idx0 = 0; p.idx1 = 1; p.idx2 = 0; p.idx3 = 1;
    }
    return p;
}

// ((uint16)(q * 3^n) * 3) >> 8 with the uint8 wrap-around the C dequantizer relies on, minus the
// scale multiply (hoisted, exactly as the PQ2 kernel hoists its own).
__device__ __forceinline__ float ptq1_trit_from_byte(std::uint8_t raw, int pow3) {
    const std::uint8_t q = static_cast<std::uint8_t>(raw * pow3);
    return static_cast<float>(static_cast<int>((static_cast<std::uint16_t>(q) * 3u) >> 8) - 1);
}

__device__ __forceinline__ float ptq1_dot4(const Ptq1LanePlan& p, int group,
                                           const float2& low, const float2& high_act) {
    const std::uint8_t* src =
        p.row + static_cast<std::int64_t>(group) * p.bytes_per_group + p.src_off;
    const float w0 = ptq1_trit_from_byte(src[p.idx0], p.pow_a);
    const float w1 = ptq1_trit_from_byte(src[p.idx1], p.pow_a);
    const float w2 = ptq1_trit_from_byte(src[p.idx2], p.pow_b);
    const float w3 = ptq1_trit_from_byte(src[p.idx3], p.pow_b);
    return fmaf(w0, low.x, fmaf(w1, low.y, fmaf(w2, high_act.x, w3 * high_act.y)));
}

// T == 1 decode: same shape as the PQ2 kernel above (one warp per output row, shuffle reduce, no
// __syncthreads), so it inherits the same occupancy-first reasoning.
__global__ __launch_bounds__(kGemvWarpsPerBlock * 32)
void ternary_ptq1_gemv_kernel(const __nv_bfloat16* __restrict__ x,
                              const std::uint8_t* __restrict__ codes,
                              const std::uint8_t* __restrict__ high,
                              const std::uint8_t* __restrict__ scales,
                              __nv_bfloat16* __restrict__ out, std::int32_t rows,
                              std::int32_t groups_per_row) {
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp =
        static_cast<int>(blockIdx.x) * kGemvWarpsPerBlock + (static_cast<int>(threadIdx.x) >> 5);
    if (warp >= rows) { return; }

    const std::uint8_t* code_row =
        codes + static_cast<std::int64_t>(warp) * groups_per_row * kPtq1CodeBytesPerGroup;
    const std::uint8_t* high_row =
        high + static_cast<std::int64_t>(warp) * groups_per_row * kPtq1HighBytesPerGroup;
    const std::uint8_t* scale_row =
        scales + static_cast<std::int64_t>(warp) * groups_per_row * kGemvScaleBytesPerGroup;

    const Ptq1LanePlan plan = ptq1_lane_plan(lane, code_row, high_row);

    float accumulator = 0.0f;

    for (int group = 0; group < groups_per_row; ++group) {
        const std::int32_t base = group * kGemvGroupK + lane * 4;
        const float2 low        = __bfloat1622float2(
            *reinterpret_cast<const __nv_bfloat162*>(x + base));
        const float2 high_act = __bfloat1622float2(
            *reinterpret_cast<const __nv_bfloat162*>(x + base + 2));

        const float dot = ptq1_dot4(plan, group, low, high_act);
        accumulator = fmaf(gemv_scale(scale_row + group * kGemvScaleBytesPerGroup), dot,
                           accumulator);
    }

#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        accumulator +=  __shfl_down_sync(0xffffffffffffffffULL, accumulator, offset);
    }
    if (lane == 0) { out[warp] = __float2bfloat16_rn(accumulator); }
}

// Small-token-tile variant: identical policy to the PQ2 tile kernel above -- the four weight values
// are decoded once per group and reused by every token in the tile, which is what keeps the
// speculative verify pass (T = draft + 1) from costing T full decode passes.
template <int kT>
__global__ __launch_bounds__(kGemvWarpsPerBlock * 32)
void ternary_ptq1_gemv_tile_kernel(const __nv_bfloat16* __restrict__ x,
                                   const std::uint8_t* __restrict__ codes,
                                   const std::uint8_t* __restrict__ high,
                                   const std::uint8_t* __restrict__ scales,
                                   __nv_bfloat16* __restrict__ out, std::int32_t rows,
                                   std::int32_t groups_per_row, std::int32_t tokens,
                                   std::int32_t out_row_stride) {
    static_assert(kT >= 1 && kT <= 8, "tile size must be small enough to keep accumulators in registers");
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp =
        static_cast<int>(blockIdx.x) * kGemvWarpsPerBlock + (static_cast<int>(threadIdx.x) >> 5);
    if (warp >= rows) { return; }

    const std::uint8_t* code_row =
        codes + static_cast<std::int64_t>(warp) * groups_per_row * kPtq1CodeBytesPerGroup;
    const std::uint8_t* high_row =
        high + static_cast<std::int64_t>(warp) * groups_per_row * kPtq1HighBytesPerGroup;
    const std::uint8_t* scale_row =
        scales + static_cast<std::int64_t>(warp) * groups_per_row * kGemvScaleBytesPerGroup;

    const Ptq1LanePlan plan = ptq1_lane_plan(lane, code_row, high_row);

    float accumulator[kT];
#pragma unroll
    for (int t = 0; t < kT; ++t) { accumulator[t] = 0.0f; }

    for (int group = 0; group < groups_per_row; ++group) {
        const std::uint8_t* src =
            plan.row + static_cast<std::int64_t>(group) * plan.bytes_per_group + plan.src_off;
        const float scale   = gemv_scale(scale_row + group * kGemvScaleBytesPerGroup);
        const float weight0 = ptq1_trit_from_byte(src[plan.idx0], plan.pow_a);
        const float weight1 = ptq1_trit_from_byte(src[plan.idx1], plan.pow_a);
        const float weight2 = ptq1_trit_from_byte(src[plan.idx2], plan.pow_b);
        const float weight3 = ptq1_trit_from_byte(src[plan.idx3], plan.pow_b);

        const std::int32_t base = group * kGemvGroupK + lane * 4;
#pragma unroll
        for (int t = 0; t < kT; ++t) {
            if (t < tokens) {
                const __nv_bfloat16* x_token =
                    x + static_cast<std::int64_t>(t) * groups_per_row * kGemvGroupK;
                const float2 low = __bfloat1622float2(
                    *reinterpret_cast<const __nv_bfloat162*>(x_token + base));
                const float2 high_act = __bfloat1622float2(
                    *reinterpret_cast<const __nv_bfloat162*>(x_token + base + 2));
                const float dot = fmaf(weight0, low.x,
                                       fmaf(weight1, low.y, fmaf(weight2, high_act.x, weight3 * high_act.y)));
                accumulator[t] = fmaf(scale, dot, accumulator[t]);
            }
        }
    }

#pragma unroll
    for (int t = 0; t < kT; ++t) {
        float value = accumulator[t];
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            value +=  __shfl_down_sync(0xffffffffffffffffULL, value, offset);
        }
        if (lane == 0 && t < tokens) {
            out[static_cast<std::int64_t>(t) * out_row_stride + warp] =
                __float2bfloat16_rn(value);
        }
    }
}

} // namespace ninfer::ops::detail

#pragma once

// S1(2026-10-17, 契约 v2 verify 摊销手术): iu8-WMMA 小 T 三元 GEMM tile(PQ2_0_G128, T∈[2,16])。
//
// 替换对象 = ternary_pq2_gemv_tile_dp4a_kernel(每 4 权重每列 1 sudot4 + 2 载入, 指令流随列
// 线性放大 ⇒ T=3 verify 每列 ≈ 一个单步, 零摊销; 见 bench/evidence/t7-verify-surgery-design-2026-10-17.md)。
// 结构(wmmqa 合同): 一个 warp 一个 16 行 tile; A = lane(l&15) 持其行的 16 个连续 K 的码字节
// (v_perm 四字节解码, 与 dp4a 同助手 ⇒ 整数域同值); B = lane(l&15) 持其列(即 token)l 的
// 16 个连续 K 的 int8(16B 向量载入); wmma i32 16x16x16 累加; 每 32-K 子块 resolve 成 fp32
// (ds8 是 per-32 子块 scale, 粗粒度 flush 会让一个子块的 scale 污染另一个 —— wmmqa 缝)。
// C 布局: lane ol 的 cell j ↔ (行 2j+(ol>>4), 列 ol&15)。
// 合同来源(W0 §2 / C0/J1 事实包 / hipfire wmmqa 217 行参考件)+ 探针实测
// (scripts/probe_s1_iu8_smallt_gemm.cu): 对 dp4a tile 全 9 形状 ≥2.0×(T=3 2.00-2.16×),
// bf16 输出域不等率 0.002-0.057% 且几乎全 ≤1 ulp, 两跑逐字节确定; fp64 host 参考四核全对。
//
// 只在 HIP 构建存在(__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32 无 nvcc 对应物)。

#include "ops/linear/ternary/ternary_rowsplit_gemv.cuh"
#include "ops/linear/ternary/ternary_s8_scratch.h"

#if defined(NINFER_HIP)

namespace ninfer::ops::detail {

using PqIu8V4I32 = int __attribute__((ext_vector_type(4)));
using PqIu8V8I32 = int __attribute__((ext_vector_type(8)));

// 记录(2026-10-17 M4-T1a): "单乘数展开解码"(z = (b*M) & 0x03030303) 经穷举搜索证明不可行
// (4 项 2 的幂乘数全组合 0 解; 见 scripts/probe_s1_iu8_smallt_gemm.cu 注释与 .tmp/find_mul.py)。
// 解码轴保持原式; 本核的 kFast 只含"夹地址去守卫载入"一项(探针 VGPR 109→80)。

// kDeferScale = true(探针 K3 档, T=3/4 略优于逐子块档): 子块只乘激活侧 d_x 进组内 u[],
// 行 scale 在组末统一乘。两档数学等价(T=3/4 K3 实测 2.02-2.16× vs K2 2.00-2.06×)。
// residual(可空, M2 刀二同款): 非空时走残差 epilogue(先读后写; 舍入点与 dp4a tile 逐字一致:
// 先舍投影 bf16, 再加残差, 再舍), 原地别名安全 —— 指针不带 __restrict__, 同格读在写前。
template <bool kDeferScale, bool kFast = false>
__global__ __launch_bounds__(32)
void ternary_pq2_iu8_wmma_tile_kernel(const std::int8_t* __restrict__ xq,
                                      const std::uint16_t* __restrict__ ds8,
                                      const std::uint8_t* __restrict__ codes,
                                      const std::uint8_t* __restrict__ scales,
                                      __nv_bfloat16* __restrict__ out, const std::int32_t rows,
                                      const std::int32_t groups_per_row, const std::int32_t tokens,
                                      const std::int32_t out_row_stride,
                                      const __nv_bfloat16* residual = nullptr) {
    const int lane    = static_cast<int>(threadIdx.x) & 31;
    const int lane16  = lane & 15;
    const int row_start = static_cast<int>(blockIdx.x) * 16;
    if (row_start >= rows) { return; }
    // A 行越界夹到末行(wmmqa 同款), 写回处再屏蔽。
    const int a_row = (row_start + lane16 < rows) ? (row_start + lane16) : (rows - 1);
    const int t     = lane16;   // 本 lane 负责的列 = token
    const bool t_valid = t < tokens;
    // kFast: 无效列夹到末 token(tokens ≥ 2 恒真), B/ds8 载入无条件执行; 无效列结果落进
    // 无效 lane 的 acc(wmma 列间独立), 存储仍按 t_valid 守卫 ⇒ 有效输出逐位同。
    const int tc = kFast ? ((t < tokens) ? t : (tokens - 1)) : t;
    const std::uint8_t* code_row =
        codes + static_cast<std::int64_t>(a_row) * groups_per_row * kGemvCodeBytesPerGroup;
    const std::uint8_t* scale_row =
        scales + static_cast<std::int64_t>(a_row) * groups_per_row * kGemvScaleBytesPerGroup;
    const std::int8_t* x_row = xq + static_cast<std::int64_t>(tc) * groups_per_row * kGemvGroupK;

    float acc[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) { acc[j] = 0.0f; }

    for (int g = 0; g < groups_per_row; ++g) {
        const float wsc_self = gemv_scale(scale_row + g * kGemvScaleBytesPerGroup);
        float wsc_row[8];
#pragma unroll
        for (int j = 0; j < 8; ++j) { wsc_row[j] = __shfl(wsc_self, 2 * j + (lane >> 4)); }
        const std::uint8_t* gp = code_row + g * kGemvCodeBytesPerGroup;
        const std::int8_t* xg = x_row + g * kGemvGroupK;
        float u[8];
#pragma unroll
        for (int j = 0; j < 8; ++j) { u[j] = 0.0f; }

#pragma unroll
        for (int sb = 0; sb < 4; ++sb) {
            PqIu8V8I32 cacc = {0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
            for (int kt = 0; kt < 2; ++kt) {
                // A: 本行该 16-K 段的 4 个码字节 → 4×v_perm → 16 个 int8(K 升序)。
                const unsigned pk = *reinterpret_cast<const unsigned*>(gp + sb * 8 + kt * 4);
                const PqIu8V4I32 av = {pq2_dp4a_symbols4((pk >> 0) & 0xFFu),
                                       pq2_dp4a_symbols4((pk >> 8) & 0xFFu),
                                       pq2_dp4a_symbols4((pk >> 16) & 0xFFu),
                                       pq2_dp4a_symbols4((pk >> 24) & 0xFFu)};
                // B: 本列(token)该 16-K 段的 16 个 int8(16B 对齐向量载入)。
                PqIu8V4I32 bv = {0, 0, 0, 0};
                if (kFast || t_valid) {
                    bv = *reinterpret_cast<const PqIu8V4I32*>(xg + sb * 32 + kt * 16);
                }
                cacc = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, av, true, bv, cacc, true);
            }
            const float d_x = kFast
                ? __half2float(__ushort_as_half(
                      ds8[static_cast<std::int64_t>(tc) * groups_per_row * 4 + g * 4 + sb]))
                : (t_valid ? __half2float(__ushort_as_half(
                                 ds8[static_cast<std::int64_t>(t) * groups_per_row * 4 + g * 4 + sb]))
                           : 0.0f);
            // 每 32-K 子块 resolve(ds8 粒度 = 32; 见文件头 wmmqa 缝)。
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                if (kDeferScale) {
                    u[j] = fmaf(d_x, static_cast<float>(cacc[j]), u[j]);
                } else {
                    acc[j] = fmaf(wsc_row[j] * d_x, static_cast<float>(cacc[j]), acc[j]);
                }
            }
        }
        if (kDeferScale) {
#pragma unroll
            for (int j = 0; j < 8; ++j) { acc[j] = fmaf(wsc_row[j], u[j], acc[j]); }
        }
    }

    if (t_valid) {
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const int row_out = row_start + 2 * j + (lane >> 4);
            if (row_out < rows) {
                const std::int64_t idx = static_cast<std::int64_t>(t) * out_row_stride + row_out;
                const __nv_bfloat16 projected = __float2bfloat16_rn(acc[j]);
                // 残差 epilogue: 同格读在写前(单线程序), 舍入点与 dp4a tile 逐字一致。
                out[idx] = residual == nullptr
                               ? projected
                               : __float2bfloat16_rn(__bfloat162float(residual[idx]) +
                                                     __bfloat162float(projected));
            }
        }
    }
}

// M4-T2 后 P0a(2026-10-18): A 侧深度预取变体(探针 K17/K18/K19 的生产化)。
// 依据(探针 evidence/p0-verify-column-decomp-2026-10-18.md §5 + scripts/probe_s1):
// 16384 行档 +4~5%, 4096 行档 depth 2/3/4 = +13/+25/+36%(站点并行度越低越吃)。
// 语义: 与主核逐字相同(同解码/同 fmaf 序/同 resolve), 仅改 A 组载入时点 ⇒ 输出逐位同;
// 滚动缓冲按深度手工展开(槽下标编译期常量 ⇒ 驻寄存器)。env NINFER_PQ2_IU8_PF = 2/3/4
// 选择深度(按值解析), 0/缺省 = 主核(默认路径零改动)。
template <int kAhead, bool kDeferScale, bool kFast>
__global__ __launch_bounds__(32)
void ternary_pq2_iu8_wmma_tile_pf_kernel(const std::int8_t* __restrict__ xq,
                                         const std::uint16_t* __restrict__ ds8,
                                         const std::uint8_t* __restrict__ codes,
                                         const std::uint8_t* __restrict__ scales,
                                         __nv_bfloat16* __restrict__ out, const std::int32_t rows,
                                         const std::int32_t groups_per_row,
                                         const std::int32_t tokens, const std::int32_t out_row_stride,
                                         const __nv_bfloat16* residual = nullptr) {
    const int lane    = static_cast<int>(threadIdx.x) & 31;
    const int lane16  = lane & 15;
    const int row_start = static_cast<int>(blockIdx.x) * 16;
    if (row_start >= rows) { return; }
    const int a_row = (row_start + lane16 < rows) ? (row_start + lane16) : (rows - 1);
    const int t     = lane16;
    const bool t_valid = t < tokens;
    const int tc = kFast ? ((t < tokens) ? t : (tokens - 1)) : t;
    const std::uint8_t* code_row =
        codes + static_cast<std::int64_t>(a_row) * groups_per_row * kGemvCodeBytesPerGroup;
    const std::uint8_t* scale_row =
        scales + static_cast<std::int64_t>(a_row) * groups_per_row * kGemvScaleBytesPerGroup;
    const std::int8_t* x_row = xq + static_cast<std::int64_t>(tc) * groups_per_row * kGemvGroupK;

    float acc[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) { acc[j] = 0.0f; }

    unsigned pk_buf[kAhead][8];
    float sc_buf[kAhead];
#pragma unroll
    for (int d = 0; d < kAhead; ++d) {
        if (d < groups_per_row) {
            const std::uint8_t* gp = code_row + d * kGemvCodeBytesPerGroup;
#pragma unroll
            for (int q = 0; q < 8; ++q) {
                pk_buf[d][q] = *reinterpret_cast<const unsigned*>(gp + q * 4);
            }
            sc_buf[d] = gemv_scale(scale_row + d * kGemvScaleBytesPerGroup);
        }
    }

    for (int g0 = 0; g0 < groups_per_row; g0 += kAhead) {
#pragma unroll
        for (int d = 0; d < kAhead; ++d) {
            const int g = g0 + d;
            if (g >= groups_per_row) { break; }
            float wsc_row[8];
#pragma unroll
            for (int j = 0; j < 8; ++j) { wsc_row[j] = __shfl(sc_buf[d], 2 * j + (lane >> 4)); }
            const std::int8_t* xg = x_row + g * kGemvGroupK;
            float u[8];
#pragma unroll
            for (int j = 0; j < 8; ++j) { u[j] = 0.0f; }

#pragma unroll
            for (int sb = 0; sb < 4; ++sb) {
                PqIu8V8I32 cacc = {0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
                for (int kt = 0; kt < 2; ++kt) {
                    const unsigned pk = pk_buf[d][sb * 2 + kt];
                    const PqIu8V4I32 av = {pq2_dp4a_symbols4((pk >> 0) & 0xFFu),
                                           pq2_dp4a_symbols4((pk >> 8) & 0xFFu),
                                           pq2_dp4a_symbols4((pk >> 16) & 0xFFu),
                                           pq2_dp4a_symbols4((pk >> 24) & 0xFFu)};
                    const PqIu8V4I32 bv =
                        *reinterpret_cast<const PqIu8V4I32*>(xg + sb * 32 + kt * 16);
                    cacc = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, av, true, bv, cacc, true);
                }
                const float d_x = __half2float(__ushort_as_half(
                    ds8[static_cast<std::int64_t>(tc) * groups_per_row * 4 + g * 4 + sb]));
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    if (kDeferScale) {
                        u[j] = fmaf(d_x, static_cast<float>(cacc[j]), u[j]);
                    } else {
                        acc[j] = fmaf(wsc_row[j] * d_x, static_cast<float>(cacc[j]), acc[j]);
                    }
                }
            }
            if (kDeferScale) {
#pragma unroll
                for (int j = 0; j < 8; ++j) { acc[j] = fmaf(wsc_row[j], u[j], acc[j]); }
            }

            const int gn = g + kAhead;
            if (gn < groups_per_row) {
                const std::uint8_t* gp = code_row + gn * kGemvCodeBytesPerGroup;
#pragma unroll
                for (int q = 0; q < 8; ++q) {
                    pk_buf[d][q] = *reinterpret_cast<const unsigned*>(gp + q * 4);
                }
                sc_buf[d] = gemv_scale(scale_row + gn * kGemvScaleBytesPerGroup);
            }
        }
    }

    if (t_valid) {
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const int row_out = row_start + 2 * j + (lane >> 4);
            if (row_out < rows) {
                const std::int64_t idx = static_cast<std::int64_t>(t) * out_row_stride + row_out;
                const __nv_bfloat16 projected = __float2bfloat16_rn(acc[j]);
                out[idx] = residual == nullptr
                               ? projected
                               : __float2bfloat16_rn(__bfloat162float(residual[idx]) +
                                                     __bfloat162float(projected));
            }
        }
    }
}

// split-K 变体(M4-T1d 接线, 2026-10-18): 与 pf 核逐字同体(fmaf 链序不变), 只把组区间切成
// [g0, g1) = blockIdx.y 段, 逐元素写 f32 部分和(布局 [ks][t][row], 行步长 = rows);
// 舍入与残差由 combine 核统一做(舍入点与主核 epilogue 逐位同: 先 bf16 舍投影, 再加残差, 再舍)。
// 数值口径 = 良性分歧档(f32 求和结合序变, 探针 K14/K16 ≤1ulp), 非逐位 ⇒ md5 需重基线。
// 形状定律(探针 K14/K16): 只在 warp 饥饿区(rows ≤ kPq2Iu8KsRows)兑现(小行数 1.37-1.64×,
// 16384 行中性), 接线由发射端按行数条件选。
template <int kAhead, bool kDeferScale, bool kFast, int kSplit>
__global__ __launch_bounds__(32)
void ternary_pq2_iu8_wmma_tile_ks_kernel(const std::int8_t* __restrict__ xq,
                                         const std::uint16_t* __restrict__ ds8,
                                         const std::uint8_t* __restrict__ codes,
                                         const std::uint8_t* __restrict__ scales,
                                         float* __restrict__ partial, const std::int32_t rows,
                                         const std::int32_t groups_per_row,
                                         const std::int32_t tokens) {
    const int lane    = static_cast<int>(threadIdx.x) & 31;
    const int lane16  = lane & 15;
    const int row_start = static_cast<int>(blockIdx.x) * 16;
    if (row_start >= rows) { return; }
    const int a_row = (row_start + lane16 < rows) ? (row_start + lane16) : (rows - 1);
    const int t     = lane16;
    const bool t_valid = t < tokens;
    const int tc = kFast ? ((t < tokens) ? t : (tokens - 1)) : t;
    const std::uint8_t* code_row =
        codes + static_cast<std::int64_t>(a_row) * groups_per_row * kGemvCodeBytesPerGroup;
    const std::uint8_t* scale_row =
        scales + static_cast<std::int64_t>(a_row) * groups_per_row * kGemvScaleBytesPerGroup;
    const std::int8_t* x_row = xq + static_cast<std::int64_t>(tc) * groups_per_row * kGemvGroupK;

    const int per = groups_per_row / kSplit;
    const int g_lo = static_cast<int>(blockIdx.y) * per;
    const int g_hi = g_lo + per;

    float acc[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) { acc[j] = 0.0f; }

    unsigned pk_buf[kAhead][8];
    float sc_buf[kAhead];
#pragma unroll
    for (int d = 0; d < kAhead; ++d) {
        if (g_lo + d < g_hi) {
            const std::uint8_t* gp = code_row + (g_lo + d) * kGemvCodeBytesPerGroup;
#pragma unroll
            for (int q = 0; q < 8; ++q) {
                pk_buf[d][q] = *reinterpret_cast<const unsigned*>(gp + q * 4);
            }
            sc_buf[d] = gemv_scale(scale_row + (g_lo + d) * kGemvScaleBytesPerGroup);
        }
    }

    for (int gb = g_lo; gb < g_hi; gb += kAhead) {
#pragma unroll
        for (int d = 0; d < kAhead; ++d) {
            const int g = gb + d;
            if (g >= g_hi) { break; }
            float wsc_row[8];
#pragma unroll
            for (int j = 0; j < 8; ++j) { wsc_row[j] = __shfl(sc_buf[d], 2 * j + (lane >> 4)); }
            const std::int8_t* xg = x_row + g * kGemvGroupK;
            float u[8];
#pragma unroll
            for (int j = 0; j < 8; ++j) { u[j] = 0.0f; }

#pragma unroll
            for (int sb = 0; sb < 4; ++sb) {
                PqIu8V8I32 cacc = {0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
                for (int kt = 0; kt < 2; ++kt) {
                    const unsigned pk = pk_buf[d][sb * 2 + kt];
                    const PqIu8V4I32 av = {pq2_dp4a_symbols4((pk >> 0) & 0xFFu),
                                           pq2_dp4a_symbols4((pk >> 8) & 0xFFu),
                                           pq2_dp4a_symbols4((pk >> 16) & 0xFFu),
                                           pq2_dp4a_symbols4((pk >> 24) & 0xFFu)};
                    const PqIu8V4I32 bv =
                        *reinterpret_cast<const PqIu8V4I32*>(xg + sb * 32 + kt * 16);
                    cacc = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, av, true, bv, cacc, true);
                }
                const float d_x = __half2float(__ushort_as_half(
                    ds8[static_cast<std::int64_t>(tc) * groups_per_row * 4 + g * 4 + sb]));
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    if (kDeferScale) {
                        u[j] = fmaf(d_x, static_cast<float>(cacc[j]), u[j]);
                    } else {
                        acc[j] = fmaf(wsc_row[j] * d_x, static_cast<float>(cacc[j]), acc[j]);
                    }
                }
            }
            if (kDeferScale) {
#pragma unroll
                for (int j = 0; j < 8; ++j) { acc[j] = fmaf(wsc_row[j], u[j], acc[j]); }
            }

            const int gn = g + kAhead;
            if (gn < g_hi) {
                const std::uint8_t* gp = code_row + gn * kGemvCodeBytesPerGroup;
#pragma unroll
                for (int q = 0; q < 8; ++q) {
                    pk_buf[d][q] = *reinterpret_cast<const unsigned*>(gp + q * 4);
                }
                sc_buf[d] = gemv_scale(scale_row + gn * kGemvScaleBytesPerGroup);
            }
        }
    }

    if (t_valid) {
        float* base =
            partial + (static_cast<std::int64_t>(blockIdx.y) * tokens + t) * rows;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const int row_out = row_start + 2 * j + (lane >> 4);
            if (row_out < rows) { base[row_out] = acc[j]; }
        }
    }
}

// 部分和合并 + bf16 舍入 + 残差 epilogue(舍入点与主核 epilogue 逐位同)。
template <int kSplit>
__global__ __launch_bounds__(256)
void ternary_pq2_iu8_ksplit_combine_kernel(const float* __restrict__ partial,
                                           __nv_bfloat16* __restrict__ out,
                                           const std::int32_t rows, const std::int32_t tokens,
                                           const std::int32_t out_row_stride,
                                           const __nv_bfloat16* residual = nullptr) {
    const std::int64_t total = static_cast<std::int64_t>(rows) * tokens;
    for (std::int64_t idx = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < total; idx += static_cast<std::int64_t>(gridDim.x) * blockDim.x) {
        const std::int32_t t = static_cast<std::int32_t>(idx / rows);
        const std::int32_t r = static_cast<std::int32_t>(idx % rows);
        float sum = 0.0f;
#pragma unroll
        for (int ks = 0; ks < kSplit; ++ks) {
            sum += partial[(static_cast<std::int64_t>(ks) * tokens + t) * rows + r];
        }
        const std::int64_t oi = static_cast<std::int64_t>(t) * out_row_stride + r;
        const __nv_bfloat16 projected = __float2bfloat16_rn(sum);
        out[oi] = residual == nullptr
                      ? projected
                      : __float2bfloat16_rn(__bfloat162float(residual[oi]) +
                                            __bfloat162float(projected));
    }
}

// 发射入口(调用方 = launch_pq2_dp4a_tile, 已完成量化 pass 与 env/准入检查)。
// kFast 选择: 默认开(探针 +6% / d2 +3.36% / serve mtp2 +2.5% 全绿, md5+验收+逐请求全同);
// NINFER_PQ2_IU8_FAST=0 回退到原路径(守卫载入)。
inline void launch_pq2_iu8_wmma_tile(const std::int8_t* xq, const std::uint16_t* ds8,
                                     const std::uint8_t* codes, const std::uint8_t* scales,
                                     __nv_bfloat16* out, const std::int32_t rows,
                                     const std::int32_t groups_per_row, const std::int32_t tokens,
                                     const std::int32_t out_row_stride, cudaStream_t stream,
                                     const __nv_bfloat16* residual = nullptr,
                                     float* ksplit_partial = nullptr) {
    static const bool fast_env = [] {
        const char* value = std::getenv("NINFER_PQ2_IU8_FAST");
        return value == nullptr || std::atoi(value) != 0;
    }();
    // M4-T1d wallet 账: 形状计数(诊断门, 默认关)。图捕获期每站点一行 ⇒ 一次 d2 跑得
    // 站点清单 (rows, k, tokens), 用于内插 K14/K16 曲线算 split-K 条件接线收益。
    static const bool shape_log = [] {
        const char* value = std::getenv("NINFER_PQ2_IU8_SHAPE_LOG");
        return value != nullptr && std::atoi(value) != 0;
    }();
    if (shape_log) {
        std::fprintf(stderr, "[pq2-iu8-shape] rows=%d k=%d tokens=%d\n", rows,
                     groups_per_row * kGemvGroupK, tokens);
    }
    const unsigned grid = static_cast<unsigned>((rows + 15) / 16);
    // split-K 条件接线(M4-T1d, 2026-10-18): 按值解析 `NINFER_PQ2_IU8_KS`(缺省/0 = 关,
    // 2|4 = 切分路数)。只在 warp 饥饿区兑现: rows ≤ kPq2Iu8KsRows 且组数可整除
    // (探针 K16: 4096 行 1.37-1.64×, 16384 行中性 ⇒ 大站不切, 走下方单发核)。
    // 数值 = 良性分歧档(f32 部分和结合序变, ≤1ulp)⇒ md5 与非切路径不同, 门 = 连贯+验收+PPL。
    // 部分和片来自 scratch(dp4a_ksplit, 容量 = ternary_pq2_iu8_ksplit_bytes); 空指针 = 关。
    // 与 fast 门联动: FAST=0(守卫载入逃生臂)时不切。
    static const int ks_env = [] {
        const char* value = std::getenv("NINFER_PQ2_IU8_KS");
        const int ks = value == nullptr ? 0 : std::atoi(value);
        if (ks >= 2) {
            std::fprintf(stderr, "[pq2-iu8] split-K path ON (ks=%d, rows<=%d)\n", ks >= 4 ? 4 : 2,
                         kPq2Iu8KsRows);
        }
        return ks;
    }();
    if (ks_env >= 2 && ksplit_partial != nullptr && fast_env && rows <= kPq2Iu8KsRows) {
        const bool ks4 = ks_env != 2;
        if (ks4 && (groups_per_row % 4) == 0) {
            const dim3 ks_grid(grid, 4u, 1u);
            ternary_pq2_iu8_wmma_tile_ks_kernel<4, true, true, 4><<<ks_grid, 32, 0, stream>>>(
                xq, ds8, codes, scales, ksplit_partial, rows, groups_per_row, tokens);
            const unsigned cgrid =
                static_cast<unsigned>((static_cast<std::int64_t>(rows) * tokens + 255) / 256);
            ternary_pq2_iu8_ksplit_combine_kernel<4><<<cgrid, 256, 0, stream>>>(
                ksplit_partial, out, rows, tokens, out_row_stride, residual);
            CUDA_CHECK(cudaGetLastError());
            return;
        }
        if (!ks4 && (groups_per_row % 2) == 0) {
            const dim3 ks_grid(grid, 2u, 1u);
            ternary_pq2_iu8_wmma_tile_ks_kernel<4, true, true, 2><<<ks_grid, 32, 0, stream>>>(
                xq, ds8, codes, scales, ksplit_partial, rows, groups_per_row, tokens);
            const unsigned cgrid =
                static_cast<unsigned>((static_cast<std::int64_t>(rows) * tokens + 255) / 256);
            ternary_pq2_iu8_ksplit_combine_kernel<2><<<cgrid, 256, 0, stream>>>(
                ksplit_partial, out, rows, tokens, out_row_stride, residual);
            CUDA_CHECK(cudaGetLastError());
            return;
        }
    }
    // P0a 深预取门(2026-10-18): 默认开 depth=4。装运臂 A/B(d2 4 对交错): 42.3 → 45.5 =
    // **+7.57%**(pf2 +6.60%; 4/4 对 pf4>pf2>base), md5 3857bdb9 三臂逐位同, 验收 1.81 同,
    // PPL 逐位同, nospec A/B 三臂逐位同(256 口径 cfb1b419)。
    // 按值解析: **=0 回退主核**(逃生门), =2/=3 选浅深度, 缺省 = 4。
    // 捕获期每站点读一次(图内固化所选实例化, 无逐 replay 分支); 输出与主核逐位同(探针两跑确定)。
    static const int pf_depth = [] {
        const char* value = std::getenv("NINFER_PQ2_IU8_PF");
        const int depth = value == nullptr ? 4 : std::atoi(value);
        if (depth >= 2) {
            std::fprintf(stderr, "[pq2-iu8] prefetch path ON (depth=%d)\n", depth >= 4 ? 4 : depth);
        }
        return depth;
    }();
    if (pf_depth >= 2) {
        if (pf_depth == 2) {
            ternary_pq2_iu8_wmma_tile_pf_kernel<2, true, true><<<grid, 32, 0, stream>>>(
                xq, ds8, codes, scales, out, rows, groups_per_row, tokens, out_row_stride, residual);
        } else if (pf_depth == 3) {
            ternary_pq2_iu8_wmma_tile_pf_kernel<3, true, true><<<grid, 32, 0, stream>>>(
                xq, ds8, codes, scales, out, rows, groups_per_row, tokens, out_row_stride, residual);
        } else {
            ternary_pq2_iu8_wmma_tile_pf_kernel<4, true, true><<<grid, 32, 0, stream>>>(
                xq, ds8, codes, scales, out, rows, groups_per_row, tokens, out_row_stride, residual);
        }
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    if (fast_env) {
        // 一次性路由宣告(A/B 的"路没到被测分支"守卫; 数值两臂逐位同 ⇒ md5 无法区分路由)。
        static const bool announced = [] {
            std::fprintf(stderr, "[pq2-iu8] fast path ON (clamped unguarded loads)\n");
            return true;
        }();
        (void)announced;
        ternary_pq2_iu8_wmma_tile_kernel<true, true><<<grid, 32, 0, stream>>>(
            xq, ds8, codes, scales, out, rows, groups_per_row, tokens, out_row_stride, residual);
    } else {
        ternary_pq2_iu8_wmma_tile_kernel<true, false><<<grid, 32, 0, stream>>>(
            xq, ds8, codes, scales, out, rows, groups_per_row, tokens, out_row_stride, residual);
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail

#endif // defined(NINFER_HIP)

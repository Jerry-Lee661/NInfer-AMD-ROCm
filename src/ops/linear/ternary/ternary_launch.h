#pragma once

#include "core/tensor.h"
#include "ops/linear/ternary/ternary_s8_scratch.h"

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

// The token tile is the only schedule knob on the reference path: one token per CTA at
// T = 1 (decode), eight otherwise (a small prefill). Both share one kernel template.
//
// out_row_stride is the row count of the OUT tensor's parent allocation (the token stride), so a
// caller can have several weights write disjoint row ranges of one fused output -- the split GDN
// and attention parents do exactly that. Pass w.n when the output is the whole tensor.
//
// The int8 rung (#14) needs an activation-quantization scratch buffer, which the caller owns
// because this op runs inside captured CUDA graphs (a lazy cudaMalloc at launch time would be
// illegal there). Callers that have no workspace -- the *_basis entry points used by the fused
// GDN/attention parents -- pass a null scratch and therefore stay on the bf16 rungs.
using TernaryLaunch = void (*)(const Tensor&, const Weight&, Tensor&, std::int32_t, cudaStream_t,
                               TernaryS8Scratch);

void launch_ternary_gemm_t1(const Tensor& x, const Weight& w, Tensor& out,
                            std::int32_t out_row_stride, cudaStream_t stream,
                            TernaryS8Scratch scratch);
void launch_ternary_gemm_t8(const Tensor& x, const Weight& w, Tensor& out,
                            std::int32_t out_row_stride, cudaStream_t stream,
                            TernaryS8Scratch scratch);

// 量化上提(2026-10-08, megakernel M1a;NINFER_DP4A_QHOIST=1): 融合调用点(同一激活
// 喂多个权重分片,如 attn_input_proj 的 4 分片、gdn_input_proj 的 3 分片)在分片派发前
// 量化一次并武装窗口;窗口内匹配 (x.data, tokens) 的 dp4a tile 跳过重复量化。
// 契约:begin/end 必须成对且与分片组同作用域(宿主派发单线程,静态窗口);
// 量化写 scratch.dp4a_xq/ds8(容量端须按 k 分配);数值逐位不变(同 x 同量化结果)。
void ternary_pq2_dp4a_hoist_begin(const Tensor& x, std::int32_t k, TernaryS8Scratch scratch,
                                  cudaStream_t stream);
void ternary_pq2_dp4a_hoist_end();

// dp4a rung(PQ2_0, T=1..4, NINFER_PQ2_DP4A=1, HIP-only): q8_1 量化 pass + v_perm/dp4a
// gemv/tile。调用方负责门检查与 scratch.dp4a_xq 的分配; nvcc 构建下为抛错桩(调用分支已 guard)。
// residual(M2 刀二, 可空): 非空时残差在 GEMV epilogue 里加(原地 residual += W*x), 省一个
// residual_add 节点; 舍入点与 compose 路径逐位一致(先舍投影 bf16, 再加, 再舍)。
void launch_pq2_dp4a_tile(const Tensor& x, const Weight& w, Tensor& out,
                          std::int32_t out_row_stride, cudaStream_t stream,
                          TernaryS8Scratch scratch, const Tensor* residual = nullptr);

// 双段合并发射(M2c, 2026-10-08): 同一父权重 w 的行区 [0,rows_a) 写 out_a、[rows_a, w.n)
// 写 out_b, 一段发射(消一半 launch;逐发带宽随尺寸爬坡, 冷 DRAM 探针 164→591 GB/s)。
// 调用方保证 rows_a + rows_b == w.n; 量化/上提与单段路径同构。仅 dp4a rung(HIP-only)。
void launch_pq2_dp4a_seg2(const Tensor& x, const Weight& w, Tensor& out_a, std::int32_t rows_a,
                          std::int32_t stride_a, Tensor& out_b, std::int32_t stride_b,
                          cudaStream_t stream, TernaryS8Scratch scratch);

// segn 表驱动合并(M2d, 2026-10-08): seg2 的推广 —— 2..4 段、各段可有不同权重基址
// (跨父权重), T=1 一段发射。段 = 已切片的权重行区(ternary_row_view 产物, 基址已带偏移)。
// 结构 = 已探针定案的 T=1 专用形态(单累加器 + group unroll-8, 相对专用核只多 3 次整数比较)。
struct TernarySegDesc {
    const Weight* w;      // 行区切片(ternary_row_view 产物)
    Tensor*       out;    // 输出([slice.n, T])
    std::int32_t  stride; // 输出 token 步长(T=1 不使用, 回退路径用)
};
void launch_pq2_dp4a_segn(const Tensor& x, const TernarySegDesc* segs, std::int32_t nsegs,
                          cudaStream_t stream, TernaryS8Scratch scratch);

// dp4a rung 准入判据的单一事实源(2026-10-08, M2): 派发分支与"旋转延迟"判定共用同一函数,
// 两处条件不许各写一份(一旦漂移, 延迟的旋转可能没人消费 → 在未旋转激活上乘折叠权重 = 数值垃圾)。
// scratch.dp4a_xq 非空 = 调用方已按 T≤4 分配了量化 scratch(见 allocate_ternary_s8_scratch)。
// 行切片(ternary_row_view)只改 n/shape[0]/padded_shape[0], qtype/k/padded_shape[1] 原样带过来,
// 所以对父权重判定等价于对每个分片判定。
[[nodiscard]] inline bool ternary_pq2_dp4a_admits(const Tensor& x, const Weight& w,
                                                  const TernaryS8Scratch& scratch) {
    return x.ne[1] <= 4 && ternary_pq2_dp4a_enabled() && scratch.dp4a_xq != nullptr &&
           w.qtype == QType::PQ2_0_G128 && w.padded_shape[1] == w.k && (w.k % 128) == 0;
}

} // namespace ninfer::ops::detail

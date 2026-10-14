#pragma once

#include "core/arena.h"
#include "ninfer/ops/linear.h"
#include "ops/linear/ternary/ternary_s8_scratch.h"
#include "ops/linear/ternary/ternary_launch.h"

#include <cstdint>

namespace ninfer::ops::detail {

TernaryLaunch select_ternary_launch(std::int32_t n, std::int32_t k, std::int32_t t,
                                    LinearPolicy policy);

// The int8 rung's activation-quantization scratch, allocated from a caller-provided arena. Callers
// that fold the activation themselves (the split attention and GDN input projections rotate once and
// feed several weights) use this and hand the result to the basis entry points below, so those
// projections can take the int8 rung too. Without it they stay on the bf16 rungs no matter how many
// tokens there are -- measured 2026-09-23 on a 4,044-token prefill: those 7 projections per layer
// (attention q/g/k/v + GDN qk/value/z, 13 of the 24 probe lines) were running the bf16 wide-t rung
// at 56.94 ms (PQ2) / 86.93 ms (PTQ1_0) per forward instead of the s8 rung's 36.90 / 34.58 ms.
// The arena capacity already counts these bytes: ternary_rotation_workspace_bytes() is the capacity
// statement for every ternary linear call (see linear.cpp), and it includes the s8 scratch.
TernaryS8Scratch allocate_ternary_s8_scratch(WorkspaceArena& workspace, std::int32_t k,
                                             std::int32_t tokens);

// Rotate the activation into the folded basis when the weight needs it, then run the GEMM.
void ternary_dispatch(const Tensor& x, const Weight& w, Tensor& out, LinearPolicy policy,
                      WorkspaceArena* workspace, cudaStream_t stream);

// M2 刀二(2026-10-08, megakernel): `residual_out += W * x` 的 dp4a 融合入口。
// 返回 true = 已由 dp4a rung 直接写进 residual_out(残差在 GEMV epilogue 里加, 省一个
// residual_add 节点);false = 该形状走不了 dp4a(或门关/无 workspace), 调用方自己
// compose:投影到 scratch 再 residual_add。
// 数值契约:与 compose 逐位一致 —— epilogue 复刻 residual_add 的舍入(先舍 bf16 投影结果,
// 再与残差相加, 再舍一次), 所以 md5 门直接判。
[[nodiscard]] bool ternary_dispatch_add(const Tensor& x, const Weight& w, Tensor& residual_out,
                                        LinearPolicy policy, WorkspaceArena* workspace,
                                        cudaStream_t stream, bool pre_rotated = false);

// Run the GEMM when the activation is ALREADY in the folded basis. Callers that feed several
// folded weights from one activation (the split GDN input projection feeds three) rotate once and
// then call this, instead of paying for -- and diverging on -- a redundant rotation per weight.
// `scratch` is optional: pass allocate_ternary_s8_scratch(...)'s result to let this call take the
// int8 rung, or nothing to stay on the bf16 rungs.
void ternary_dispatch_basis(const Tensor& x_folded, const Weight& w, Tensor& out,
                            LinearPolicy policy, cudaStream_t stream,
                            TernaryS8Scratch scratch = {});

// M2c(2026-10-08): 同一父权重 w 的两个行区一段发射 —— [0,rows_a) 写 out_a、[rows_a,w.n) 写
// out_b(调用方保证 rows_a + rows_b == w.n)。dp4a 准入(T≤4)时一段发射消一半 launch;否则
// 回退为两次行切片 strided 派发(旧行为, 逐位同)。
void ternary_dispatch_basis_seg2(const Tensor& x_folded, const Weight& w, Tensor& out_a,
                                 std::int32_t rows_a, std::int32_t stride_a, Tensor& out_b,
                                 std::int32_t stride_b, LinearPolicy policy, cudaStream_t stream,
                                 TernaryS8Scratch scratch = {});

// M2d(2026-10-08): segn 表驱动 —— 2..4 段(可跨父权重)在 T=1 一段发射;不满足准入
// (或 NINFER_SEGN_FUSE=0)时回退为逐段 strided 派发(旧行为, 逐位同)。
void ternary_dispatch_basis_segn(const Tensor& x_folded, const TernarySegDesc* segs,
                                 std::int32_t nsegs, LinearPolicy policy, cudaStream_t stream,
                                 TernaryS8Scratch scratch = {});

// Same, for a weight that writes a ROW RANGE of a larger fused output. `out` must already point at
// the range start (Tensor::slice over ne[0], the contiguous axis, does that) and out_row_stride is
// the parent's row count -- which is also the token stride under ninfer's ne[0]-contiguous layout.
void ternary_dispatch_basis_strided(const Tensor& x_folded, const Weight& w, Tensor& out,
                                    std::int32_t out_row_stride, LinearPolicy policy,
                                    cudaStream_t stream, TernaryS8Scratch scratch = {});

} // namespace ninfer::ops::detail

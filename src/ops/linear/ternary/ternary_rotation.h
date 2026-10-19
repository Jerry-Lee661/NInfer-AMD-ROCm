#pragma once

#include "core/arena.h"
#include "core/tensor.h"
#include "ops/linear/ternary/ternary_s8_scratch.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::detail {

// Folded (rotated) ternary basis support for the Prism/Bonsai ports.
//
// The ternary linear weights are stored folded into a rotated basis, so the model computes
//
//     y = W' * (H * (s * P * x))
//
// where P is an optional column permutation (only the GDN output projection carries one), s is
// an explicit +-1 diagonal, and H is the normalized Sylvester-Hadamard transform over blocks of
// 1024. Everything below is driven by data the binder already attached to Weight
// (hadamard_signs / hadamard_n_blk / hadamard_perm_*), so the op layer never needs to know which
// model layers are folded: `qtype` alone decides.
//
// The transform is applied to the ACTIVATION, never to the weight. Folding H and s back into W
// would make W a dense float matrix (`W = W' * H * s`), which is not ternary representable and
// would re-inflate the artifact from 7 GB to ~17 GB -- the exact failure mode this port exists
// to avoid.

// True when this weight's activation must be mapped into the folded basis. False for every
// non-ternary weight, which is why the non-ternary pipelines are unaffected.
[[nodiscard]] bool ternary_weight_is_folded(const Weight& weight) noexcept;

// NINFER_TERNARY_HADAMARD=0 disables the transform at run time.
//
// This is the bisection switch between the two ways an end-to-end ternary run can be wrong:
// decode errors (wrong codes / wrong scale / wrong row order) and rotation errors (wrong
// butterfly, wrong sign row, wrong permutation). With the transform off, the engine still runs
// the full forward pass and still reports a decode speed, so the M4 throughput number can be
// measured before the transform is trusted; the output is numerically meaningless either way,
// and a perplexity that is *not* in the hundreds is the signal that the rotation is correct.
[[nodiscard]] bool ternary_rotation_enabled();

// Whether the folded GDN feature permutation P is applied.
//
// llama.cpp applies P because ITS runtime keeps the GDN V channels tiled, matching the GGUF's
// `attn_qkv` row order, while the folded `ssm_out` keeps its training (grouped) column order
// (`conversion/qwen.py:617-626`). The GGUF metadata records this as
// `prism.hadamard.gdn_v_grouped = 1` -- "v is already grouped, do not permute a second time".
//
// This port's packer normalizes EVERY GDN tensor to the grouped (HF) order: `gdn/value_z` applies
// tiled->grouped to the attn_qkv V rows and to attn_gate, and `ssm_conv1d`/`ssm_alpha`/`ssm_beta`
// are reordered the same way. The runtime therefore already produces grouped V, so applying P here
// permutes a SECOND time and scrambles 48 of the 64 layers.
//
// Default: off. NINFER_TERNARY_GDN_PERM=1 restores the llama.cpp behaviour, which is what a
// tile-ordered artifact would need.
[[nodiscard]] bool ternary_gdn_perm_enabled();

// Device bytes needed to hold a rotated activation of shape [k, tokens], BF16.
[[nodiscard]] std::size_t ternary_rotation_workspace_bytes(std::int32_t k, std::int32_t tokens);

// Bytes a fused family needs to decompose into a ternary GEMM: a scoped [output_rows, tokens]
// BF16 projection scratch (the GEMM writes [N, T] and the family's epilogue then folds it in) plus
// the [input_rows, tokens] rotation buffer the GEMM maps the activation through.
[[nodiscard]] std::size_t ternary_projection_workspace_bytes(std::int32_t output_rows,
                                                             std::int32_t input_rows,
                                                             std::int32_t tokens);

// 折叠几何的单一事实源(2026-10-08, M2): 前向旋转核与旋转+量化融合核共用同一份 P 判定
// (含 NINFER_TERNARY_GDN_PERM 门与跨度校验), 两处不许各推一次。
struct TernaryRotationGeometry {
    std::int32_t perm_hd  = 0;
    std::int32_t perm_nk  = 0;
    std::int32_t perm_rep = 1;
};
[[nodiscard]] TernaryRotationGeometry ternary_rotation_geometry(const Weight& weight);

// Map `x` ([k, tokens], BF16) into the folded basis of `weight`, writing [k, tokens] BF16.
// Applies P, then the signs, then H: out = H * (s * (P * x)).
void launch_ternary_rotation(const Tensor& x, Tensor& out, const Weight& weight,
                             cudaStream_t stream);

// 旋转 + q8_1 量化融合(M2): 直接在未旋转的 x 上做 (H∘s∘P) 并写出 dp4a 的 q8_1 激活
// (scratch.dp4a_xq / dp4a_ds8), 省掉 bf16 中间缓冲与一次 launch。
// 几何与 signs 取自 scratch.rot_* 字段(由 folded_activation_or_defer 从权重填好)。
void launch_ternary_rotation_quantized(const Tensor& x, std::int32_t k, TernaryS8Scratch scratch,
                                       cudaStream_t stream);

// Inverse for the token-embedding path: `out = s * (H * x)`, applied IN PLACE to a [k, tokens]
// BF16 activation.
//
// The embedding table stores its rows folded into the rotated basis, so the looked-up rows must be
// mapped back before they can enter the residual stream. There is no permutation on this path
// (llama-model.cpp gates P on `.ssm_out.` alone), and H and s are both involutions, so the inverse
// is the same transform with the sign multiply on the other side.
//
// In place on purpose: the gathered embedding is a freshly produced activation that nothing else
// reads, so mapping it in place costs no scratch buffer at all -- which matters because the
// embedding op has no workspace parameter.
void launch_ternary_rotation_inverse_inplace(Tensor& data, const Weight& weight,
                                            cudaStream_t stream);

// Convenience wrapper around launch_ternary_rotation for callers that feed several folded weights
// from one activation: allocates the scratch from `workspace` and returns the mapped activation.
//
// Returns `x` unchanged when NINFER_TERNARY_HADAMARD=0, and throws when the rotation is enabled
// but the weight carries no sign block -- a folded weight without signs means the artifact is
// missing text/hadamard_signs, and running it untransformed would produce plausible garbage.
//
// The CALLER owns the workspace scope: the returned Tensor points into the arena and stays valid
// only until that scope ends.
[[nodiscard]] Tensor folded_activation(const Tensor& x, const Weight& weight,
                                       WorkspaceArena& workspace, cudaStream_t stream);

// 折叠激活的入口(2026-10-08, M2 起): 能延迟时返回"未旋转的 x"并把旋转几何写进 scratch
// (下游 dp4a 量化核用融合核就地旋转+量化, 省一个 rotate 节点), 否则与 folded_activation 同行为。
//
// 判定 = NINFER_ROTQ_FUSE 门开 && 权重带 sign block && ternary_pq2_dp4a_admits(x, weight, scratch)
// (与派发分支同一判据)。scratch 以引用传入并被填写; 调用方随后必须把同一个 scratch 交给
// ternary_dispatch_basis*(或 launch_pq2_dp4a_hoist_begin), 否则派发侧会以
// "延迟的旋转没人消费"抛错, 而不是在未旋转的激活上算出垃圾。
[[nodiscard]] Tensor folded_activation_or_defer(const Tensor& x, const Weight& weight,
                                                WorkspaceArena& workspace, cudaStream_t stream,
                                                TernaryS8Scratch& scratch);

// T4(2026-10-16, M3-1): rmsnorm + rotquant 融合的接线面(arm 机制, 与 QHOIST 静态窗口同族)。
//
// 分离路径: rmsnorm 节点(execution 层发射) -> bf16 h -> ops wrapper 的 folded_activation_or_defer
// 决定"旋转 or 延迟" -> dp4a 量化发射点(launch_ternary_rotation_quantized)读 h。
// 融合路径把 norm 节点并进量化发射点: arm 命中时该发射点改跑
// ternary_rmsnorm_rotate_quant_q8_kernel(norm 平方和归约 + 旋转 + 量化一核, 直接写
// scratch.dp4a_xq/dp4a_ds8), h 整个不被读写, 每站点省一个节点(64 层 x 2 个范数原点)。
//
// 安全面(为什么 arm 之外还要 guard): arm 只登记 {expect_x, raw_x, norm_w, eps}(执行层看不到
// 消费权重的形状); "跳过 norm 是否安全"由消费投影在自己的层里裁决 = ternary_rmsnorm_rotq_guard:
//   消费权重可纳 dp4a 延迟(与 NINFER_ROTQ_FUSE 同一判据) => 无事(下游融合核会接手, h 不被读);
//   否则 => 立即补做 unit-offset rmsnorm 写回 h 再放行(退化为原两节点路径, 数值等价)。
// 因此 arm 打错时只会多跑一次 rmsnorm, 永远不会让消费端读到未初始化的 h。
//
// 生命周期: 单会话静态(宿主派发与图捕获均单线程, 同 QHOIST 契约); 每个融合站点在自己的消费
// 调用前 arm 一次, 由融合发射或 guard 回退消费后失效; 图回放不再执行宿主代码。
struct RmsnormRotqArm {
    const void* expect_x  = nullptr;   // 消费层将看到的激活地址(= h.data)
    const Tensor* raw_x   = nullptr;   // arm 之前被跳过 norm 的原始激活
    const Tensor* norm_w  = nullptr;   // 该 norm 的 unit-offset 权重
    float eps             = 0.0f;
    std::int32_t k        = 0;         // 范数行宽(融合核只服务 5120)
};

// arm 成功返回 true => 调用方跳过 ops::rmsnorm。判定 = NINFER_RMSNORM_ROTQ_FUSE=1(按值解析)
// && k == 5120 && tokens 在 dp4a 窗口(1..4)。消费权重维度的裁决留给 guard。
[[nodiscard]] bool ternary_rmsnorm_rotq_arm(const void* expect_x, const Tensor& raw_x,
                                            const Tensor& norm_weight, float eps, std::int32_t k);
void ternary_rmsnorm_rotq_disarm();
[[nodiscard]] const RmsnormRotqArm* rmsnorm_rotq_arm_state();

// 消费守卫(投影所在层调用, 每个可能吃到 arm 激活的消费入口一次): 见上"安全面"。
void ternary_rmsnorm_rotq_guard(const Tensor& hidden, const Weight& consumer, cudaStream_t stream);

} // namespace ninfer::ops::detail

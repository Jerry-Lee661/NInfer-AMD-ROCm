#pragma once

// T5 片 1(2026-10-16): MLP 持久链核 —— 把每层 MLP 尾的 6 个节点(norm → rotquant →
// gate_up → silu → rotquant → down+res)并成 1 个持久核(段间网格屏障替代发射)。
//
// 编排依据 = bench/evidence/t5-pricing-and-notes-2026-10-16.md(节点价 3.2µs, 屏障 733ns);
// 数值契约 = 与原 6 节点路径逐字节一致(段体逐字复用 + 行/块归属改变不改变逐行算术),
// 仲裁探针 scripts/probe_mlp_chain.cu 三闸全绿(REF↔CHAIN 8/8、自复位、改造体↔原版体 4/4),
// 证据 = bench/evidence/t5-slice1-chain-probe-2026-10-16.md。
//
// 门: NINFER_MLP_CHAIN=0 回退(**默认开**; 翻转依据 = 三件套全绿: md5 逐字节 +
// 节点 970→650 + 同窗墙钟 +9.8%, 证据 = bench/evidence/t5-slice1-wiring-2026-10-16.md)。
// 准入还要求: decode T=1、PQ2_0_G128 折叠权重、大形状(5120/17408/34816)、旋转几何无 P
// (perm_rep<=1, 与 v3 旋转量化核的逐字段体一致); 网格按物理 CU 数夹取保证共驻留。
// 返回 true = 已由链核接手(调用方 return, 不再走原路径); false = 未准入, 调用方走原路径。
//
// 生命周期约束(同 QHOIST/T4 arm 契约): 单会话静态 —— 链核的屏障计数槽是设备全局,
// 同一设备上并发回放两张含链核的图(多引擎/多流)不受支持; 单引擎(CLI/serve 单实例)安全。

#include "core/arena.h"
#include "core/tensor.h"
#include "ninfer/ops/linear.h"
#include "ops/linear/ternary/ternary_s8_scratch.h"

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

[[nodiscard]] bool ternary_mlp_chain(const Tensor& x, const Tensor& post_norm,
                                     const Weight& gate_up_weight, const Weight& down_weight,
                                     float eps, WorkspaceArena& workspace, cudaStream_t stream);

// T5 片 2a(2026-10-16): 前置链 —— 每型层 mixer 尾的 [激活闸门 + o_proj 旋转量化 + o_proj 投影]
// 3 个节点并成 1 个持久核(2 道网格屏障)。两型层共用同核, 激活闸门分岔:
//   attn: sigmoid_mul(原地) + rotq(k=6144) + o_proj GEMV+残差;
//   gdn : gated_rmsnorm(d=128, 48 行) + rotq(k=6144) + o_proj GEMV+残差。
// 门: NINFER_OUTPROJ_CHAIN=0 回退(默认开; 翻转依据 = 同片 1 三件套纪律)。准入: decode T=1、
// o_proj 为 PQ2_0_G128 折叠权重(n=5120, k=6144, perm_rep<=1)、旋转/dp4a/rotq 融合/残差融合
// 四闸在开、激活与残差激活形状逐维匹配。返回 true = 已接手(调用方 return); false = 走原路径。
// 生命周期约束同 ternary_mlp_chain(单会话静态)。
[[nodiscard]] bool ternary_out_proj_chain_attn(const Tensor& gate, Tensor& act,
                                               const Weight& o_proj_weight, Tensor& x,
                                               WorkspaceArena& workspace, cudaStream_t stream);

[[nodiscard]] bool ternary_out_proj_chain_gdn(const Tensor& o, const Tensor& norm_weight,
                                              const Tensor& z, Tensor& on,
                                              const Weight& o_proj_weight, Tensor& x, float eps,
                                              WorkspaceArena& workspace, cudaStream_t stream);

// T5 片 2b(2026-10-16): 统一尾链 —— mixer 站的前置链与 mlp_tail 站的 MLP 链在节点流里严格
// 相邻(run_layers 两站之间零节点), 并成 1 个持久核(两链的段体整体复用, 无算术改动)。
// 接线: 发射点仍在 mixer 站(前置段输入在其 scope 内; MLP 侧权重是常驻权重结构, 跨站可见),
// 成功后置 pending 旗标, 由 mlp_tail 的链入口消费一次(那站跳过发射)。门: NINFER_TAIL_CHAIN=0
// 回退(默认开, 翻转依据见 .cu 门注释); 复用前置链与 MLP 链两个门与准入(两者都必须开)。
[[nodiscard]] bool ternary_tail_chain_attn(const Tensor& gate, Tensor& act,
                                           const Weight& o_proj_weight, Tensor& x,
                                           const Tensor& post_norm, const Weight& gate_up_weight,
                                           const Weight& down_weight, float eps,
                                           WorkspaceArena& workspace, cudaStream_t stream);

[[nodiscard]] bool ternary_tail_chain_gdn(const Tensor& o, const Tensor& norm_weight,
                                          const Tensor& z, Tensor& on,
                                          const Weight& o_proj_weight, Tensor& x,
                                          const Tensor& post_norm, const Weight& gate_up_weight,
                                          const Weight& down_weight, float eps,
                                          WorkspaceArena& workspace, cudaStream_t stream);

// R1 刀(2026-10-17 剩余节点组): [GDN recurrent 批更新 + 统一尾链] 2 节点并 1。接线面 =
// text_context_impl GDN Verify 支(快照路径 T=1/width=1/batch=1)的 batch_update 节点。
// recurrent 段 = recurrent_batch_update_kernel 的 T=1 体逐字复用(同一组 device 函数 +
// BatchUpdateAccess; 仅坐标由引擎几何平铺重映射到链网格, 逐 WG 自洽)。准入 = 统一尾链
// 全部前提 + recurrent 侧几何/形状/网格自洽。门 NINFER_REC_TAIL_CHAIN(默认关, `=1` 开)。
[[nodiscard]] bool ternary_tail_chain_gdn_recurrent(
    const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& g, const Tensor& beta,
    float scale, const Tensor& recurrent_states, const Tensor& source_state_slots,
    const Tensor& destination_state_slots, Tensor& o, const Tensor& norm_weight, const Tensor& z,
    Tensor& on, const Weight& o_proj_weight, Tensor& x, const Tensor& post_norm,
    const Weight& gate_up_weight, const Weight& down_weight, float eps,
    WorkspaceArena& workspace, cudaStream_t stream);

// pending 旗标消费(一次; 带层身份核对 = post_norm 缓冲地址, 每层唯一且跨捕获稳定):
// true = 本层 MLP 尾已由统一尾链接手, MLP 链入口直接返回(不发射); 核对不中即清(防粘滞)。
[[nodiscard]] bool ternary_mlp_chain_consume_pending(const void* norm_identity);

// T5 片 3(2026-10-16): 输入投影链 —— 每层 X 侧的三支链路(k 同为 5120), 各以 1 道/2 道网格
// 屏障替代发射。门: NINFER_INPUT_CHAIN=0 回退(默认开, 2026-10-16 门全绿后翻转; 按值解析)。准入: decode T=1、折叠 PQ2 父权重
// (k=5120, n_blk=5, perm_rep<=1)、相关融合闸在开、张量形状/对齐逐维匹配、网格 >= 覆盖门槛。
// 返回 true = 已接手(调用方不再走原路径); false = 未准入, 调用方走原路径。
//
// attn 支: [input_norm + rotquant(k=5120) + 四段 qkv GEMV] 3 节点 → 1
// (q|k 来自 query_key 父, gate|v 来自 gate_value 父)。
[[nodiscard]] bool ternary_input_chain_attn(const Tensor& x_raw, const Tensor& norm_w, float eps,
                                            const Weight& query_key_w, const Weight& gate_value_w,
                                            Tensor& q, Tensor& gate, Tensor& k, Tensor& v,
                                            Tensor& h, WorkspaceArena& workspace,
                                            cudaStream_t stream);

// gdn 刀一: [input_norm + 闸门 GEMV(a/b 96 行)] 2 节点 → 1(gdn_mix 两支同构接入)。
// 闸门体 = bf16_gdn_gating_proj_gemv_kernel 逐字; 与旋转无关, 不设旋转闸。
[[nodiscard]] bool ternary_norm_gating_chain(const Tensor& x_raw, const Tensor& norm_w, float eps,
                                             const Weight& a_w, const Weight& b_w,
                                             const Tensor& a_log, const Tensor& dt_bias, Tensor& h,
                                             Tensor& g, Tensor& beta, cudaStream_t stream);

// gdn 刀二: [rotquant(k=5120) + 三段 qk|value|z GEMV] 2 节点 → 1(接线面 = launch_ternary_split
// 入口, 必须早于量化上提臂)。前提 = 调用方 scratch 的旋转延迟态(激活未旋转, 链核就地旋转
// + 量化); scratch 的量化缓冲由调用方分配(k=5120/T=1 尺寸)。
[[nodiscard]] bool ternary_segn_chain_split(const Tensor& activation, const Weight& qk_w,
                                            const Weight& value_z_w, Tensor& qkv, Tensor& z,
                                            std::int32_t qk_rows, std::int32_t value_rows,
                                            std::int32_t z_rows, TernaryS8Scratch scratch,
                                            cudaStream_t stream);

// S2a(2026-10-17 剩余节点组): [rotquant(k=5120) + 三段 qk|value|z GEMV + gdn 投影卷积]
// 3 节点 → 1。接线面 = gdn_input_proj_conv_snapshot 的 ternary 分支(快照路径 T=1/单列)。
// 前提与刀二逐条一致(scratch 旋转延迟态); conv 侧仅 27B 几何(10240 = 2048+2048+6144),
// conv_state_read/write 同缓冲(conv_states)。门 NINFER_SEGN_CONV_CHAIN=0 回退(默认开)。
[[nodiscard]] bool ternary_segn_conv_chain_split(
    const Tensor& activation, const Weight& qk_w, const Weight& value_z_w, Tensor& qkv, Tensor& z,
    std::int32_t qk_rows, std::int32_t value_rows, std::int32_t z_rows, const Tensor& conv_weight,
    Tensor& conv_states, const Tensor& valid_columns, const Tensor& initial_state_slots,
    const Tensor& snapshot_base_slots, Tensor& query, Tensor& key, Tensor& value,
    std::int32_t query_rows, std::int32_t key_rows, TernaryS8Scratch scratch, cudaStream_t stream);

} // namespace ninfer::ops::detail

#include "ops/linear/ternary/ternary_rotation.h"

#include "ops/linear/ternary/ternary_launch.h"
#include "ops/linear/ternary/ternary_s8_scratch.h"

#include "ninfer/ops/rmsnorm.h"

#include <cstdlib>
#include <stdexcept>
#include <string>

namespace ninfer::ops::detail {
namespace {

constexpr std::size_t kActivationBytesPerElement = 2; // BF16

bool ternary_qtype(QType qtype) noexcept {
    return qtype == QType::PTQ1_0_G128 || qtype == QType::PQ2_0_G128;
}

// T4: rmsnorm+rotquant 融合 arm(单会话静态; 宿主派发与图捕获均单线程, 同 QHOIST 契约)。
RmsnormRotqArm g_rmsnorm_rotq_arm;

bool rmsnorm_rotq_fuse_enabled() {
    // 与 kernels 头同名门(唯一事实源 = 同名 env); 本文件由宿主编译器编译, 不能包含带核的
    // kernels 头, 故此处独立按值解析。"1" 开, 其余(含 "0"/未设)关。
    static const bool enabled = [] {
        const char* value = std::getenv("NINFER_RMSNORM_ROTQ_FUSE");
        return value != nullptr && value[0] == '1' && value[1] == '\0';
    }();
    return enabled;
}

// 延迟(以及据此成立的融合)的几何判据里 scratch 无关的部分。folded_activation_or_defer 与
// arm/guard 共用这一份, 不许两处各推一次。
bool folded_dp4a_admits(std::int32_t tokens, const Weight& weight) {
    return ternary_rotq_fuse_enabled() && ternary_rotation_enabled() &&
           ternary_weight_is_folded(weight) && tokens >= 1 && tokens <= kPq2Dp4aMaxTokens &&
           ternary_pq2_dp4a_enabled() && weight.qtype == QType::PQ2_0_G128 &&
           weight.padded_shape[1] == weight.k && (weight.k % 128) == 0;
}

} // namespace

bool ternary_weight_is_folded(const Weight& weight) noexcept {
    return ternary_qtype(weight.qtype) && weight.hadamard_signs != nullptr &&
           weight.hadamard_n_blk > 0;
}

bool ternary_rotation_enabled() {
    // Read once: the value must not change between graph construction and graph replay, because
    // the rotation decides whether a kernel appears in the captured graph at all.
    static const bool enabled = [] {
        const char* value = std::getenv("NINFER_TERNARY_HADAMARD");
        return value == nullptr || std::string(value) != "0";
    }();
    return enabled;
}

bool ternary_gdn_perm_enabled() {
    // Default OFF: this port's packer already normalized every GDN tensor to the grouped order,
    // so the reference's feature permutation would be a second permutation. See the header.
    static const bool enabled = [] {
        const char* value = std::getenv("NINFER_TERNARY_GDN_PERM");
        return value != nullptr && std::string(value) == "1";
    }();
    return enabled;
}

std::size_t ternary_rotation_workspace_bytes(std::int32_t k, std::int32_t tokens) {
    if (k <= 0 || tokens <= 0) { return 0; }
    // The rotation buffer plus the int8 rung's activation-quantization scratch (one int8 code row
    // per token, one fp32 scale per token, and a little slack for the 256-byte arena alignment).
    // They are declared together because this function IS the capacity statement for the ternary
    // linear op (see linear.cpp), so anything the op allocates from the arena must be counted here.
    // The quantization pass itself only runs for T >= kTernaryS8MinTokens, but the capacity is the
    // max over token counts the plan allows, which is what the planner asks for.
    const std::size_t rotated = static_cast<std::size_t>(k) * static_cast<std::size_t>(tokens) *
                                kActivationBytesPerElement;
    const std::size_t s8_scratch =
        ternary_s8_codes_bytes(k, tokens) + ternary_s8_scales_bytes(tokens) + 512u;
    // dp4a rung(PQ2 decode GEMV/tile, T=1..kPq2Dp4aMaxTokens): 容量按 rung 上限计
    // (k×6 int8 + k/32×6 fp16), 与 allocate_ternary_s8_scratch(按实际 tokens 分配, ≤ 上限)
    // 取最大值一致。
    const std::size_t dp4a_scratch = ternary_pq2_dp4a_xq_bytes(k, kPq2Dp4aMaxTokens) +
                                     ternary_pq2_dp4a_ds8_bytes(k, kPq2Dp4aMaxTokens) + 256u;
    // split-K 部分和片(M4-T1d): 固定按区域上界计(只与 tokens 有关, 见 s8_scratch.h)。
    const std::size_t ksplit_scratch = ternary_pq2_iu8_ksplit_bytes(tokens);
    return rotated + s8_scratch + dp4a_scratch + ksplit_scratch;
}

std::size_t ternary_projection_workspace_bytes(std::int32_t output_rows, std::int32_t input_rows,
                                               std::int32_t tokens) {
    return ternary_rotation_workspace_bytes(output_rows, tokens) +
           ternary_rotation_workspace_bytes(input_rows, tokens);
}

Tensor folded_activation(const Tensor& x, const Weight& weight, WorkspaceArena& workspace,
                         cudaStream_t stream) {
    if (!ternary_rotation_enabled()) { return x; }
    if (!ternary_weight_is_folded(weight)) {
        throw std::invalid_argument(
            "folded ternary weight has no sign block; the artifact must carry "
            "text/hadamard_signs and text/hadamard_widths");
    }
    const DeviceSpan span =
        workspace.alloc_bytes(ternary_rotation_workspace_bytes(weight.k, x.ne[1]));
    Tensor rotated(span.data, DType::BF16, {weight.k, x.ne[1]});
    launch_ternary_rotation(x, rotated, weight, stream);
    return rotated;
}

Tensor folded_activation_or_defer(const Tensor& x, const Weight& weight, WorkspaceArena& workspace,
                                  cudaStream_t stream, TernaryS8Scratch& scratch) {
    // 延迟条件里带 folded_dp4a_admits(几何判据) + ternary_pq2_dp4a_admits(= 派发分支的同一判据):
    // 只要这次派发会走 dp4a rung, 融合量化核就一定会跑, 旋转不会落空。判据不成立时老老实实
    // 先旋转(旧行为)。
    if (!scratch.rot_deferred() && folded_dp4a_admits(x.ne[1], weight) &&
        ternary_pq2_dp4a_admits(x, weight, scratch)) {
        const TernaryRotationGeometry geometry = ternary_rotation_geometry(weight);
        scratch.rot_signs    = weight.hadamard_signs;
        scratch.rot_n_blk    = weight.hadamard_n_blk;
        scratch.rot_perm_hd  = geometry.perm_hd;
        scratch.rot_perm_nk  = geometry.perm_nk;
        scratch.rot_perm_rep = geometry.perm_rep;
        return x;
    }
    return folded_activation(x, weight, workspace, stream);
}

// ---- T4: rmsnorm+rotquant 融合 arm/guard(契约与安全面见头文件) ----

bool ternary_rmsnorm_rotq_arm(const void* expect_x, const Tensor& raw_x,
                              const Tensor& norm_weight, float eps, std::int32_t k) {
    if (!rmsnorm_rotq_fuse_enabled() || expect_x == nullptr || k != 5120) { return false; }
    // 单次量化不变量: attn/gdn 的量化上提窗口(QHOIST, 默认开)保证每站点只发一次量化,
    // 融合核"消费即失效"才安全; QHOIST 关时不 arm(回退常规两节点路径, 零风险)。
    if (!ternary_pq2_dp4a_qhoist_enabled()) { return false; }
    const std::int32_t tokens = raw_x.ne[1];
    if (tokens < 1 || tokens > 4) { return false; }
    if (raw_x.ne[0] != k || norm_weight.ne[0] != k) { return false; }
    g_rmsnorm_rotq_arm = RmsnormRotqArm{expect_x, &raw_x, &norm_weight, eps, k};
    return true;
}

void ternary_rmsnorm_rotq_disarm() { g_rmsnorm_rotq_arm = RmsnormRotqArm{}; }

const RmsnormRotqArm* rmsnorm_rotq_arm_state() {
    return g_rmsnorm_rotq_arm.expect_x != nullptr ? &g_rmsnorm_rotq_arm : nullptr;
}

void ternary_rmsnorm_rotq_guard(const Tensor& hidden, const Weight& consumer, cudaStream_t stream) {
    const RmsnormRotqArm* arm = rmsnorm_rotq_arm_state();
    if (arm == nullptr || arm->expect_x != hidden.data) { return; }
    if (folded_dp4a_admits(hidden.ne[1], consumer) && consumer.k == arm->k) { return; }
    // 回退: 补做被跳过的 unit-offset rmsnorm 写回 hidden(与执行层原调用逐字同参数), 再放行原路径。
    Tensor out(hidden.data, DType::BF16, {hidden.ne[0], hidden.ne[1]});
    ops::rmsnorm(*arm->raw_x, *arm->norm_w, arm->eps, /*unit_offset=*/true, out, stream);
    ternary_rmsnorm_rotq_disarm();
}

} // namespace ninfer::ops::detail

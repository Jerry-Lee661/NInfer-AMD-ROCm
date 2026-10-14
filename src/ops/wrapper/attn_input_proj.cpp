#include "ninfer/ops/attn_input_proj.h"

#include "ops/attn_input_proj/bf16/bf16_attn_input_plan.h"
#include "ops/attn_input_proj/fp8/fp8_attn_input_plan.h"
#include "ops/attn_input_proj/nvfp4/nvfp4_attn_input_plan.h"
#include "ops/attn_input_proj/q4_q5/q4_q5_attn_input_plan.h"
#include "ops/attn_input_proj/w8/w8_attn_input_plan.h"
#include "ops/linear/fp8/fp8_config.h"
#include "ops/linear/fp8/fp8_format.h"
#include "ops/linear/nvfp4/nvfp4_config.h"
#include "ops/linear/nvfp4/nvfp4_format.h"
#include "ops/linear/ternary/ternary_dispatch.h"
#include "ops/linear/ternary/ternary_launch.h"
#include "ops/linear/ternary/ternary_row_view.h"
#include "ops/linear/ternary/ternary_rotation.h"

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

bool aligned_to(const void* pointer, std::uintptr_t alignment) {
    return pointer != nullptr && (reinterpret_cast<std::uintptr_t>(pointer) & (alignment - 1)) == 0;
}

void require_matrix(const Tensor& tensor, std::int32_t rows, std::int32_t cols, const char* label) {
    if (tensor.dtype != DType::BF16 || tensor.ne[0] != rows || tensor.ne[1] != cols ||
        tensor.ne[2] != 1 || tensor.ne[3] != 1 || !tensor.is_contiguous() ||
        !aligned_to(tensor.data, 16)) {
        throw std::invalid_argument(std::string("attn_input_proj: invalid ") + label);
    }
}

void require_rowsplit(const Weight& weight, QType qtype, std::int32_t rows, const char* label) {
    const bool q4_planes =
        qtype != QType::Q4G64_F16S || (weight.qhigh == nullptr && weight.high_plane_bytes == 0);
    const bool q5_planes =
        qtype != QType::Q5G64_F16S || (weight.qhigh != nullptr && weight.high_plane_bytes != 0);
    if (weight.qtype != qtype || weight.layout != QuantLayout::RowSplit ||
        weight.scale_dtype != DType::FP16 || weight.group_size != 64 || weight.group != 64 ||
        weight.ndim != 2 || weight.n != rows || weight.k != 5120 || weight.shape[0] != rows ||
        weight.shape[1] != 5120 || weight.padded_shape[0] != rows ||
        weight.padded_shape[1] != 5120 || !q4_planes || !q5_planes ||
        !aligned_to(weight.qdata, 16) || !aligned_to(weight.scales, 4) ||
        (qtype == QType::Q5G64_F16S && !aligned_to(weight.qhigh, 16))) {
        throw std::invalid_argument(std::string("attn_input_proj: invalid ") + label);
    }
}

void require_w8_rowsplit(const Weight& weight, std::int32_t rows, const char* label) {
    if (weight.qtype != QType::W8G32_F16S || weight.layout != QuantLayout::RowSplit ||
        weight.scale_dtype != DType::FP16 || weight.group_size != 32 || weight.group != 32 ||
        weight.ndim != 2 || weight.n != rows || weight.k != 2048 || weight.shape[0] != rows ||
        weight.shape[1] != 2048 || weight.padded_shape[0] != rows ||
        weight.padded_shape[1] != 2048 || weight.qhigh != nullptr || weight.high_plane_bytes != 0 ||
        !aligned_to(weight.qdata, 16) || !aligned_to(weight.scales, 16)) {
        throw std::invalid_argument(std::string("attn_input_proj: invalid ") + label);
    }
}

void require_bf16_contiguous(const Weight& weight, std::int32_t rows, std::int32_t hidden,
                             const char* label) {
    const std::uint64_t payload_bytes = static_cast<std::uint64_t>(rows) *
                                        static_cast<std::uint64_t>(hidden) * sizeof(std::uint16_t);
    if (weight.qtype != QType::BF16_CTRL || weight.layout != QuantLayout::Contiguous ||
        weight.payload_bytes < payload_bytes || weight.high_plane_bytes != 0 || weight.ndim != 2 ||
        weight.n != rows || weight.k != hidden || weight.shape[0] != rows ||
        weight.shape[1] != hidden || weight.padded_shape[0] != rows ||
        weight.padded_shape[1] != hidden || weight.qhigh != nullptr || weight.scales != nullptr ||
        weight.group_size != 0 || weight.group != 0 || !aligned_to(weight.qdata, 16)) {
        throw std::invalid_argument(std::string("attn_input_proj: invalid ") + label);
    }
}

void validate_policy(LinearPolicy policy) {
    switch (policy) {
    case LinearPolicy::A16Only:
    case LinearPolicy::AllowA8:
    case LinearPolicy::AllowA4:
        return;
    }
    throw std::invalid_argument("attn_input_proj: invalid compute policy");
}

void dispatch_single_parent(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate,
                            Tensor& k, Tensor& v, LinearPolicy policy, WorkspaceArena* workspace,
                            cudaStream_t stream) {
    validate_policy(policy);
    if (weight.qtype == QType::BF16_CTRL) {
        constexpr std::int32_t kHidden = 5120;
        constexpr std::int32_t kQRows  = 6144;
        constexpr std::int32_t kKvRows = 1024;
        constexpr std::int32_t kRows   = 14336;
        const std::int32_t cols        = x.ne[1];
        if (cols <= 0) { throw std::invalid_argument("attn_input_proj: T must be positive"); }
        if (policy != LinearPolicy::A16Only) {
            throw std::invalid_argument("BF16 attn_input_proj admits only A16");
        }
        require_matrix(x, kHidden, cols, "x");
        require_matrix(q, kQRows, cols, "q");
        require_matrix(gate, kQRows, cols, "gate");
        require_matrix(k, kKvRows, cols, "k");
        require_matrix(v, kKvRows, cols, "v");
        require_bf16_contiguous(weight, kRows, kHidden, "query/key/gate/value weight");
        detail::bf16_attn_input_dispatch(x, weight, q, gate, k, v, stream);
        return;
    }

    if (weight.qtype == QType::NVFP4) {
        constexpr std::int32_t kHidden = 5120;
        constexpr std::int32_t kQRows  = 6144;
        constexpr std::int32_t kKvRows = 1024;
        constexpr std::int32_t kRows   = 14336;
        const std::int32_t cols        = x.ne[1];
        if (cols <= 0) { throw std::invalid_argument("attn_input_proj: T must be positive"); }
        if (policy != LinearPolicy::A16Only && policy != LinearPolicy::AllowA4) {
            throw std::invalid_argument("NVFP4 attn_input_proj admits only A16 or A4");
        }
        require_matrix(x, kHidden, cols, "x");
        require_matrix(q, kQRows, cols, "q");
        require_matrix(gate, kQRows, cols, "gate");
        require_matrix(k, kKvRows, cols, "k");
        require_matrix(v, kKvRows, cols, "v");
        detail::validate_nvfp4_weight(weight, "nvfp4 attn_input_proj");
        if (weight.n != kRows || weight.k != kHidden) {
            throw std::invalid_argument("nvfp4 attn_input_proj: unsupported weight shape");
        }
        detail::nvfp4_attn_input_dispatch(x, weight, q, gate, k, v, policy, workspace, stream);
        return;
    }

    if (weight.qtype == QType::FP8_E4M3FN_ROW_BF16S) {
        constexpr std::int32_t kHidden = 5120;
        constexpr std::int32_t kQRows  = 6144;
        constexpr std::int32_t kKvRows = 1024;
        constexpr std::int32_t kRows   = 14336;
        const std::int32_t cols        = x.ne[1];
        if (cols <= 0) { throw std::invalid_argument("attn_input_proj: T must be positive"); }
        if (policy != LinearPolicy::A16Only && policy != LinearPolicy::AllowA8) {
            throw std::invalid_argument("FP8 attn_input_proj admits only A16 or A8");
        }
        require_matrix(x, kHidden, cols, "x");
        require_matrix(q, kQRows, cols, "q");
        require_matrix(gate, kQRows, cols, "gate");
        require_matrix(k, kKvRows, cols, "k");
        require_matrix(v, kKvRows, cols, "v");
        detail::validate_fp8_weight(weight, "fp8 attn_input_proj");
        if (weight.n != kRows || weight.k != kHidden) {
            throw std::invalid_argument("fp8 attn_input_proj: unsupported weight shape");
        }
        detail::fp8_attn_input_dispatch(x, weight, q, gate, k, v, policy, workspace, stream);
        return;
    }

    constexpr std::int32_t kHidden = 2048;
    constexpr std::int32_t kQRows  = 4096;
    constexpr std::int32_t kKvRows = 512;
    constexpr std::int32_t kRows   = 9216;
    const std::int32_t cols        = x.ne[1];
    if (cols <= 0) { throw std::invalid_argument("attn_input_proj: T must be positive"); }
    if (policy != LinearPolicy::A16Only) {
        throw std::invalid_argument("W8 attn_input_proj admits only A16");
    }
    require_matrix(x, kHidden, cols, "x");
    require_matrix(q, kQRows, cols, "q");
    require_matrix(gate, kQRows, cols, "gate");
    require_matrix(k, kKvRows, cols, "k");
    require_matrix(v, kKvRows, cols, "v");
    require_w8_rowsplit(weight, kRows, "query/key/gate/value weight");
    detail::w8_attn_input_dispatch(x, weight, q, gate, k, v, stream);
}

// --- folded (rotated-basis) ternary split parents -----------------------------------------
//
// The ternary parents carry the row-split layout of Q4/Q5 but a 128-wide group geometry, so the
// Q4/Q5 required-geometry checks cannot accept them and their fused dispatch cannot run them.

bool is_ternary_attn_parent(QType qtype) noexcept {
    return qtype == QType::PTQ1_0_G128 || qtype == QType::PQ2_0_G128;
}

void require_ternary_attn_parents(const Weight& query_key_weight,
                                  const Weight& gate_value_weight, std::int32_t rows,
                                  std::int32_t hidden) {
    const Weight* const parents[]{&query_key_weight, &gate_value_weight};
    for (const Weight* parent : parents) {
        if (!is_ternary_attn_parent(parent->qtype) || parent->layout != QuantLayout::RowSplit ||
            parent->scale_dtype != DType::FP16 || parent->group != 128 || parent->ndim != 2 ||
            parent->k != hidden || parent->shape[1] != hidden ||
            parent->padded_shape[1] != hidden || (parent->k % 1024) != 0) {
            throw std::invalid_argument(
                "attn_input_proj: unsupported ternary split parent geometry");
        }
    }
    const Weight* const both[]{&query_key_weight, &gate_value_weight};
    for (const Weight* parent : both) {
        if (parent->n != rows || parent->shape[0] != rows) {
            throw std::invalid_argument(
                "attn_input_proj: ternary split parent has the wrong row count");
        }
    }
    if (query_key_weight.qtype != gate_value_weight.qtype) {
        throw std::invalid_argument(
            "attn_input_proj: both ternary split parents must use the same format");
    }
}

// The activation is `hidden` wide for all four projections, so one rotation serves all of them.
// `scratch` is the int8 rung's activation-quantization scratch (empty = stay on the bf16 rungs);
// the caller allocates it from its arena because this function never sees one -- see
// allocate_ternary_s8_scratch. Passing it is what lets these four projections take the s8 rung on
// a long prefill, where they used to fall all the way back to bf16 wide-t.
void launch_ternary_attn(const Tensor& activation, const Weight& query_key_weight,
                         const Weight& gate_value_weight, Tensor& q, Tensor& gate, Tensor& k,
                         Tensor& v, std::int32_t query_rows, std::int32_t kv_rows,
                         detail::TernaryS8Scratch scratch, cudaStream_t stream) {
    // 量化上提(megakernel M1a): 四个分片同一激活,量化一次(NINFER_DP4A_QHOIST=1)。
    const bool qhoist = detail::ternary_pq2_dp4a_qhoist_enabled() && scratch.dp4a_xq != nullptr;
    if (qhoist) { detail::ternary_pq2_dp4a_hoist_begin(activation, query_key_weight.k, scratch, stream); }
    if (activation.ne[1] == 1) {
        // 表驱动合并(M2d, NINFER_SEGN_FUSE=0 回退): 四切片跨两父权重一段发射, 2 → 1。
        const Weight q_head = detail::ternary_row_view(query_key_weight, 0, query_rows);
        const Weight k_tail = detail::ternary_row_view(query_key_weight, query_rows, kv_rows);
        const Weight g_head = detail::ternary_row_view(gate_value_weight, 0, query_rows);
        const Weight v_tail = detail::ternary_row_view(gate_value_weight, query_rows, kv_rows);
        const detail::TernarySegDesc segs[4] = {{&q_head, &q, query_rows},
                                                {&k_tail, &k, kv_rows},
                                                {&g_head, &gate, query_rows},
                                                {&v_tail, &v, kv_rows}};
        detail::ternary_dispatch_basis_segn(activation, segs, 4, LinearPolicy::A16Only, stream,
                                            scratch);
    } else {
        // 双段合并(M2c, NINFER_SEG2_FUSE=0 回退): q|k 与 gate|v 各一段发射, 4 launch → 2。
        detail::ternary_dispatch_basis_seg2(activation, query_key_weight, q, query_rows, query_rows,
                                            k, kv_rows, LinearPolicy::A16Only, stream, scratch);
        detail::ternary_dispatch_basis_seg2(activation, gate_value_weight, gate, query_rows,
                                            query_rows, v, kv_rows, LinearPolicy::A16Only, stream,
                                            scratch);
    }
    if (qhoist) { detail::ternary_pq2_dp4a_hoist_end(); }
}

} // namespace

std::size_t attn_input_proj_workspace_capacity_bytes(QType parent_qtype, std::int32_t parent_rows,
                                                     std::int32_t input_rows, LinearPolicy policy,
                                                     std::int32_t min_tokens,
                                                     std::int32_t max_tokens) {
    validate_policy(policy);
    if (min_tokens <= 0 || max_tokens < min_tokens) {
        throw std::invalid_argument("attn_input_proj workspace: invalid token interval");
    }

    switch (parent_qtype) {
    case QType::BF16_CTRL:
        if (parent_rows != 14336 || input_rows != 5120 || policy != LinearPolicy::A16Only) {
            throw std::invalid_argument("attn_input_proj workspace: unsupported BF16 profile");
        }
        return 0;
    case QType::NVFP4:
        if (parent_rows != detail::Nvfp4AttnInputGeometry::kOutputRows ||
            input_rows != detail::Nvfp4AttnInputGeometry::kInputRows ||
            (policy != LinearPolicy::A16Only && policy != LinearPolicy::AllowA4)) {
            throw std::invalid_argument("attn_input_proj workspace: unsupported NVFP4 profile");
        }
        return detail::nvfp4_attn_input_workspace_capacity_bytes(policy, min_tokens, max_tokens);
    case QType::FP8_E4M3FN_ROW_BF16S:
        if (parent_rows != detail::Fp8AttnInputGeometry::kOutputRows ||
            input_rows != detail::Fp8AttnInputGeometry::kInputRows) {
            throw std::invalid_argument("attn_input_proj workspace: unsupported FP8 profile");
        }
        return detail::fp8_attn_input_workspace_capacity_bytes(policy, min_tokens, max_tokens);
    case QType::W8G32_F16S:
        if (parent_rows != 9216 || input_rows != 2048 || policy != LinearPolicy::A16Only) {
            throw std::invalid_argument("attn_input_proj workspace: unsupported W8 profile");
        }
        (void)detail::w8_attn_input_resolve_plan(
            {input_rows, 4096, 512, parent_rows, input_rows, min_tokens});
        (void)detail::w8_attn_input_resolve_plan(
            {input_rows, 4096, 512, parent_rows, input_rows, max_tokens});
        return 0;
    case QType::Q4G64_F16S:
    case QType::Q5G64_F16S:
    case QType::Q6G64_F16S:
    case QType::FP32_CTRL:
    case QType::I32_CTRL:
        break;
    }
    throw std::invalid_argument("attn_input_proj workspace: unsupported parent qtype");
}

void attn_input_proj(const Tensor& x, const Weight& query_key_weight,
                     const Weight& gate_value_weight, Tensor& q, Tensor& gate, Tensor& k, Tensor& v,
                     cudaStream_t stream) {
    constexpr std::int32_t kHidden = 5120;
    constexpr std::int32_t kQRows  = 6144;
    constexpr std::int32_t kKvRows = 1024;
    const std::int32_t cols        = x.ne[1];
    require_matrix(x, kHidden, cols, "x");
    require_matrix(q, kQRows, cols, "q");
    require_matrix(gate, kQRows, cols, "gate");
    require_matrix(k, kKvRows, cols, "k");
    require_matrix(v, kKvRows, cols, "v");

    if (is_ternary_attn_parent(query_key_weight.qtype)) {
        require_ternary_attn_parents(query_key_weight, gate_value_weight, kQRows + kKvRows, kHidden);
        if (detail::ternary_rotation_enabled()) {
            // Folded parents need a [5120, T] rotation buffer, and this entry point has no
            // workspace to take it from. Fail loudly instead of running untransformed weights.
            throw std::invalid_argument(
                "attn_input_proj: folded ternary parents need the workspace overload "
                "(or NINFER_TERNARY_HADAMARD=0 to measure without the rotation)");
        }
        launch_ternary_attn(x, query_key_weight, gate_value_weight, q, gate, k, v, kQRows, kKvRows,
                            detail::TernaryS8Scratch{}, stream);
        return;
    }

    require_rowsplit(query_key_weight, QType::Q4G64_F16S, kQRows + kKvRows, "query/key weight");
    require_rowsplit(gate_value_weight, QType::Q5G64_F16S, kQRows + kKvRows, "gate/value weight");

    detail::q4_q5_attn_input_dispatch(x, query_key_weight, gate_value_weight, q, gate, k, v,
                                      stream);
}

void attn_input_proj(const Tensor& x, const Weight& query_key_weight,
                     const Weight& gate_value_weight, Tensor& q, Tensor& gate, Tensor& k, Tensor& v,
                     WorkspaceArena& workspace, cudaStream_t stream) {
    constexpr std::int32_t kHidden = 5120;
    constexpr std::int32_t kQRows  = 6144;
    constexpr std::int32_t kKvRows = 1024;
    const std::int32_t cols        = x.ne[1];
    require_matrix(x, kHidden, cols, "x");
    require_matrix(q, kQRows, cols, "q");
    require_matrix(gate, kQRows, cols, "gate");
    require_matrix(k, kKvRows, cols, "k");
    require_matrix(v, kKvRows, cols, "v");

    if (!is_ternary_attn_parent(query_key_weight.qtype)) {
        attn_input_proj(x, query_key_weight, gate_value_weight, q, gate, k, v, stream);
        return;
    }
    require_ternary_attn_parents(query_key_weight, gate_value_weight, kQRows + kKvRows, kHidden);
    // Scoped: the rotation scratch is handed back when this call returns, so it does not
    // accumulate across the (many) graph constructions of one load. The int8 scratch rides in the
    // same scope, and the arena capacity already counts its bytes (ternary_rotation_workspace_bytes
    // is the capacity statement for this op).
    auto scope = workspace.scope();
    // scratch 先分配: folded_activation_or_defer 要看它决定旋转走向(M2 延迟融合)。
    // 延迟时返回未旋转的 x, 旋转由 dp4a 量化核里的融合核一次性完成(省一个 rotate 节点)。
    detail::TernaryS8Scratch s8_scratch =
        detail::allocate_ternary_s8_scratch(workspace, kHidden, cols);
    const Tensor activation =
        detail::folded_activation_or_defer(x, query_key_weight, workspace, stream, s8_scratch);
    launch_ternary_attn(activation, query_key_weight, gate_value_weight, q, gate, k, v, kQRows,
                        kKvRows, s8_scratch, stream);
}

void attn_input_proj(const Tensor& x, const Weight& query_key_gate_value_weight, Tensor& q,
                     Tensor& gate, Tensor& k, Tensor& v, LinearPolicy policy,
                     WorkspaceArena& workspace, cudaStream_t stream) {
    dispatch_single_parent(x, query_key_gate_value_weight, q, gate, k, v, policy, &workspace,
                           stream);
}

void attn_input_proj(const Tensor& x, const Weight& query_key_gate_value_weight, Tensor& q,
                     Tensor& gate, Tensor& k, Tensor& v, cudaStream_t stream) {
    dispatch_single_parent(x, query_key_gate_value_weight, q, gate, k, v, LinearPolicy::A16Only,
                           nullptr, stream);
}

void attn_input_proj(const Tensor& x, const Weight& query_key_value_weight, Tensor& q, Tensor& k,
                     Tensor& v, cudaStream_t stream) {
    constexpr std::int32_t kHidden = 2048;
    constexpr std::int32_t kQRows  = 4096;
    constexpr std::int32_t kKvRows = 1024;
    constexpr std::int32_t kRows   = 6144;
    const std::int32_t cols        = x.ne[1];
    if (cols <= 0) { throw std::invalid_argument("attn_input_proj: T must be positive"); }
    require_matrix(x, kHidden, cols, "x");
    require_matrix(q, kQRows, cols, "q");
    require_matrix(k, kKvRows, cols, "k");
    require_matrix(v, kKvRows, cols, "v");
    require_w8_rowsplit(query_key_value_weight, kRows, "query/key/value weight");

    detail::w8_attn_input_dispatch(x, query_key_value_weight, q, k, v, stream);
}

} // namespace ninfer::ops

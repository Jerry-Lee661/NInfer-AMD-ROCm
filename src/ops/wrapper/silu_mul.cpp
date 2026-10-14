// ninfer::ops — silu_mul wrapper: implements the public api, validates parameters, and
// dispatches to the launcher. Host-compiled; never includes the kernel header.
// See docs/op-development.md §2.
#include "ninfer/ops/silu_mul.h"

#include "ops/launcher/silu_and_mul.h" // detail::silu_and_mul_launch
#include "ops/linear/ternary/ternary_rotation.h"

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops {

void silu_mul_rotate(const Tensor& gate, const Tensor& up, Tensor& out,
                     const Weight& rotation_source, cudaStream_t stream) {
    // 校验:同形 [K,T] BF16;lane 维连续(nb[0]==2B);K 为 1024 倍数;权重带 sign block。
    // token 步长显式传核(gate/up 是 [2K,T] 父张量的 ne[0] 切片,保留父步长)。
    if (gate.dtype != DType::BF16 || up.dtype != DType::BF16 || out.dtype != DType::BF16 ||
        gate.ne[0] != up.ne[0] || gate.ne[1] != up.ne[1] || out.ne[0] != gate.ne[0] ||
        out.ne[1] != gate.ne[1] || gate.data == nullptr || up.data == nullptr ||
        out.data == nullptr || gate.nb[0] != 2 || up.nb[0] != 2 || out.nb[0] != 2) {
        throw std::invalid_argument(
            "silu_mul_rotate: expected same-shaped lane-contiguous BF16 [K,T]");
    }
    const std::int32_t k = gate.ne[0];
    if ((k % 1024) != 0) {
        throw std::invalid_argument("silu_mul_rotate: K must be a multiple of 1024");
    }
    if (!detail::ternary_rotation_enabled() || !detail::ternary_weight_is_folded(rotation_source)) {
        throw std::invalid_argument("silu_mul_rotate: rotation source is not a folded weight");
    }
    if (rotation_source.hadamard_n_blk != k / 1024) {
        throw std::invalid_argument("silu_mul_rotate: sign rows do not match K");
    }
    const detail::TernaryRotationGeometry geometry = detail::ternary_rotation_geometry(rotation_source);
    detail::silu_and_mul_rotate_launch(gate, up, out, rotation_source.hadamard_signs,
                                       rotation_source.hadamard_n_blk, geometry.perm_hd,
                                       geometry.perm_nk, geometry.perm_rep, gate.nb[1] / 2,
                                       up.nb[1] / 2, out.nb[1] / 2, stream);
}

void silu_mul(const Tensor& gate, const Tensor& up, Tensor& out, cudaStream_t stream) {
    if (gate.dtype != DType::BF16 || up.dtype != DType::BF16 || out.dtype != DType::BF16) {
        throw std::invalid_argument("silu_mul: gate/up/out must be BF16");
    }
    for (int d = 0; d < 4; ++d) {
        if (gate.ne[d] != up.ne[d] || gate.ne[d] != out.ne[d]) {
            throw std::invalid_argument("silu_mul: gate/up/out shapes must match");
        }
    }
    if (!out.is_contiguous()) { throw std::invalid_argument("silu_mul: out must be contiguous"); }
    if (out.numel() == 0) { return; }
    if (gate.data == nullptr || up.data == nullptr || out.data == nullptr) {
        throw std::invalid_argument("silu_mul: gate/up/out data must be non-null");
    }

    detail::silu_and_mul_launch(gate, up, out, stream); // single variant -> direct dispatch
}

} // namespace ninfer::ops

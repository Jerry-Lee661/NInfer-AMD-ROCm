#pragma once

// megakernel M1b 刀一 launcher(qk_norm_rope.cuh): rmsnorm(q)+rmsnorm(k)+rope 三合一。
// 融合核仅覆盖 Text1D / QHeads=24 / KHeads=4 / HeadDim=256 / Half=32(27B decode 形态);
// admits 不匹配时返回 false 由调用方回退三发射路径。

#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail {

// Text1D + QHeads=24 + KHeads=4 + HeadDim=256 的 27B decode 形态;ne[2] = tokens。
[[nodiscard]] inline bool qk_norm_rope_admits(const Tensor& q, const Tensor& k,
                                              const Tensor& positions) {
    // positions 允许长于 tokens(= 全长 rope_pos 的前 T 视角语义一致);只查下限与对齐。
    return q.ne[0] == 256 && k.ne[0] == 256 && q.ne[1] == 24 && k.ne[1] == 4 &&
           q.ne[2] == k.ne[2] && q.ne[2] >= 1 && positions.ne[0] >= q.ne[2] &&
           (reinterpret_cast<std::uintptr_t>(q.data) & 15u) == 0u &&
           (reinterpret_cast<std::uintptr_t>(k.data) & 15u) == 0u;
}

void qk_norm_rope_launch(const Tensor& q, const Tensor& k, const Tensor& q_norm_w,
                         const Tensor& k_norm_w, float eps, const Tensor& positions, Tensor& qn,
                         Tensor& kn, cudaStream_t stream);

} // namespace ninfer::ops::detail

#pragma once

// megakernel M1b 刀一(2026-10-08): attention prologue 三合一 —— rmsnorm(q) +
// rmsnorm(k) + rope(qn, kn) 单发射。覆盖形态:Text1D 单序列、n_q=24、n_kv=4、
// head_dim=256(rotary 64);不匹配形状返回 false,调用方回退三发射路径。
// 数值契约:与三发射路径逐位一致(归一化舍入点/rope 交错半旋/角度来源全部同源)。

#include "core/tensor.h"

#include <cuda_runtime.h> // cudaStream_t

namespace ninfer::ops {

// 返回 true = 融合核已执行(qn/kn 已写);false = 形状不匹配,调用方回退三发射。
[[nodiscard]] bool qk_norm_rope(const Tensor& q, const Tensor& k, const Tensor& q_norm_w,
                                const Tensor& k_norm_w, float eps, const Tensor& positions,
                                Tensor& qn, Tensor& kn, cudaStream_t stream);

} // namespace ninfer::ops

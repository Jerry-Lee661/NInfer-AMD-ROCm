#pragma once

// ninfer::ops::detail — private launch prototype for silu_mul. Included by the wrapper
// (host) and defined by the launcher (.cu). Not part of the public api.
// See docs/op-development.md §2.

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

// Host entry; assumes inputs already validated by the wrapper.
void silu_and_mul_launch(const Tensor& gate, const Tensor& up, Tensor& out, cudaStream_t stream);

// silu_mul + Hadamard 旋转融合(M2e): out = H·(s·P·(silu(gate)∘up))。warp = (1024-block,
// token)。signs/n_blk/perm 取自折叠权重的 sign block(调用方校验 folded 与值域)。
void silu_and_mul_rotate_launch(const Tensor& gate, const Tensor& up, Tensor& out,
                                const float* signs, std::int32_t n_blk, std::int32_t perm_hd,
                                std::int32_t perm_nk, std::int32_t perm_rep,
                                std::int64_t gate_token_stride, std::int64_t up_token_stride,
                                std::int64_t out_token_stride, cudaStream_t stream);

} // namespace ninfer::ops::detail

#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h> // cudaStream_t

namespace ninfer::ops {

/**
 * Elementwise SwiGLU activation:
 *
 *   ideal[i] = (gate[i] / (1 + exp(-gate[i]))) * up[i].
 *
 * `gate`, `up`, and `out` are same-shaped BF16 tensors. out is contiguous; gate and up may use
 * arbitrary valid Tensor strides. out must not overlap either input (the two read-only inputs may
 * overlap one another). The oracle evaluates `ideal` in FP64 from the represented inputs. The BF16
 * output is promoted and compared directly with that result; output storage rounding belongs to
 * the Op's numerical criterion, not the oracle. Private kernel arithmetic is
 * implementation-defined. The Op writes all of out and uses no workspace or persistent state.
 */
void silu_mul(const Tensor& gate, const Tensor& up, Tensor& out, cudaStream_t stream);

// silu_mul + Hadamard 旋转融合(M2e, 2026-10-09): out = H·(s·P·(silu(gate)∘up))。
// `rotation_source` 是下游折叠权重的 sign block 来源(如 MLP 的 down 权重);
// 非折叠权重或 K 非 1024 倍数时抛错(调用方先判 ternary_weight_is_folded)。
// 数值契约:与 "silu_mul 落 bf16 → ternary_rotate_bf16" 两核逐位一致。
void silu_mul_rotate(const Tensor& gate, const Tensor& up, Tensor& out, const Weight& rotation_source,
                     cudaStream_t stream);

} // namespace ninfer::ops

// megakernel M1b 刀一 launcher(qk_norm_rope.cuh 的宿主侧)。
#include "ops/launcher/qk_norm_rope.h"

#include "core/device.h"
#include "ops/kernel/qk_norm_rope.cuh"
#include "ops/launcher/qk_norm_rope.h"

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops {

bool qk_norm_rope(const Tensor& q, const Tensor& k, const Tensor& q_norm_w,
                  const Tensor& k_norm_w, float eps, const Tensor& positions, Tensor& qn,
                  Tensor& kn, cudaStream_t stream) {
    if (!detail::qk_norm_rope_admits(q, k, positions)) {
        if (std::getenv("NINFER_GRAPH_NODES") != nullptr) {
            std::fprintf(stderr, "[qk-norm-rope refuse] q=%dx%dx%d k=%dx%dx%d pos=%d\n",
                         q.ne[0], q.ne[1], q.ne[2], k.ne[0], k.ne[1], k.ne[2], positions.ne[0]);
            std::fflush(stderr);
        }
        return false;
    }
    detail::qk_norm_rope_launch(q, k, q_norm_w, k_norm_w, eps, positions, qn, kn, stream);
    return true;
}

} // namespace ninfer::ops

namespace ninfer::ops::detail {

void qk_norm_rope_launch(const Tensor& q, const Tensor& k, const Tensor& q_norm_w,
                         const Tensor& k_norm_w, float eps, const Tensor& positions, Tensor& qn,
                         Tensor& kn, cudaStream_t stream) {
    const auto tokens = static_cast<std::int32_t>(positions.ne[0]);
    qk_norm_rope_kernel<<<tokens, 1024, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(q.data), static_cast<const __nv_bfloat16*>(k.data),
        static_cast<const __nv_bfloat16*>(q_norm_w.data),
        static_cast<const __nv_bfloat16*>(k_norm_w.data), eps,
        static_cast<const std::int32_t*>(positions.data),
        static_cast<__nv_bfloat16*>(qn.data), static_cast<__nv_bfloat16*>(kn.data), tokens);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail

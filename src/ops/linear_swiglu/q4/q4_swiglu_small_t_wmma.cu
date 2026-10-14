#include "ops/linear_swiglu/q4/q4_swiglu_small_t_wmma.cuh"

#include "core/device.h"
#include "core/tensor.h"

#include <cstdlib>

namespace ninfer::ops::detail {

void q4_swiglu_small_t_wmma_launch(const Tensor& x, const Weight& w, Tensor& out,
                                   cudaStream_t stream) {
    // 只跟踪第一次调用（首层新鲜状态，污染尚未扩散）。
    static const int trace = [] {
        const char* value = std::getenv("NINFER_SWIGLU_NARROW_TRACE");
        return value != nullptr && std::atoi(value) != 0;
    }();
    static int calls = 0;
    const int trace_value = (trace && calls++ == 0) ? 1 : 0;
    CUDA_CHECK(cudaMemcpyToSymbol(q4_swiglu_trace, &trace_value, sizeof(int)));
    const std::int32_t k        = static_cast<std::int32_t>(x.ne[0]);
    const std::int32_t t        = static_cast<std::int32_t>(x.ne[1]);
    const std::int32_t out_rows = static_cast<std::int32_t>(out.ne[0]);
    const dim3 grid(static_cast<unsigned>(out_rows / 32));
    q4_swiglu_small_t_wmma_kernel<<<grid, q4_swiglu_small_t_cfg::kThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data),
        static_cast<const std::uint8_t*>(w.qdata),
        static_cast<const std::uint8_t*>(w.scales),
        static_cast<__nv_bfloat16*>(out.data), k, t, static_cast<std::int32_t>(w.padded_shape[1]),
        out_rows);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail

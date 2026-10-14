#include "ops/linear_swiglu/q4/q4_linear_swiglu_kernels.h"

#include "ops/linear_swiglu/q4/q4_linear_swiglu_gemm_mma.cuh"
#include "ops/linear_swiglu/q4/q4_linear_swiglu_gemm_wmma.cuh"

#include "core/device.h"
#include "ops/common/math.h"
#include "ops/common/token_slices.h"

#include <cstdint>
#include <cstdlib>

namespace ninfer::ops::detail {
namespace {

using GateUpC40Cfg  = GemmCfg<64, 40, 64, 64, 8, 2, 1, false, true, true>;
using GateUpC48Cfg  = GemmCfg<64, 48, 64, 64, 8, 2, 1, false, true, true>;
using GateUpC128Cfg = GemmCfg<64, 128, 64, 64, 16, 2, 1, false, true, true>;

template <class Cfg, bool Full>
void launch_folded(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    constexpr int PM = Cfg::BM / 2;
    const int t      = x.ne[1];
    const dim3 grid(static_cast<unsigned>(div_up(out.ne[0], PM)),
                    static_cast<unsigned>(div_up(t, Cfg::BN)));
    if constexpr (Full) {
        q4_linear_swiglu_mma_split_half_pair_kernel<Cfg, true><<<grid, Cfg::THREADS, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const std::uint8_t*>(weight.qdata),
            static_cast<const std::uint8_t*>(weight.scales), static_cast<__nv_bfloat16*>(out.data),
            out.ne[0], x.ne[0], t, weight.padded_shape[1]);
    } else {
        q4_linear_swiglu_mma_split_half_pair_kernel<Cfg, false><<<grid, Cfg::THREADS, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const std::uint8_t*>(weight.qdata),
            static_cast<const std::uint8_t*>(weight.scales), static_cast<__nv_bfloat16*>(out.data),
            out.ne[0], x.ne[0], t, weight.padded_shape[1]);
    }
    CUDA_CHECK(cudaGetLastError());
}

template <class Cfg>
void launch_route(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    const bool full = (x.ne[1] % Cfg::BN) == 0;
    for_each_token_slice(x.ne[1], Cfg::BN, [&](std::int32_t offset, std::int32_t count) {
        const Tensor x_slice = x.slice(1, offset, count);
        Tensor out_slice     = out.slice(1, offset, count);
        if (full) {
            launch_folded<Cfg, true>(x_slice, weight, out_slice, stream);
        } else {
            launch_folded<Cfg, false>(x_slice, weight, out_slice, stream);
        }
    });
}

#if defined(NINFER_HIP)

// WMMA replacement for the C128 route's emulated inner loop. On by default: the numeric gate
// (bench/numeric-gate.sh) and a byte-identical greedy A/B against the emulated kernel
// (bench/wmma-ab.sh) both pass, at 1.8x the prefill rate. NINFER_SWIGLU_WMMA=0 falls back.
bool swiglu_wmma_enabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("NINFER_SWIGLU_WMMA");
        return value == nullptr || std::atoi(value) != 0;
    }();
    return enabled;
}

bool swiglu_wmma_covers(const Tensor& x, const Weight& weight) {
    return x.ne[0] % 16 == 0 && weight.padded_shape[1] % swiglu_wmma_cfg::kBK == 0;
}

void launch_wmma(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    constexpr int PM = swiglu_wmma_cfg::kPM;
    constexpr int BN = swiglu_wmma_cfg::kBN;
    constexpr int THREADS = swiglu_wmma_cfg::kThreads;
    const dim3 grid(static_cast<unsigned>(div_up(out.ne[0], PM)));
    static const bool debug = std::getenv("NINFER_SWIGLU_WMMA_DEBUG") != nullptr;
    static int calls        = 0;
    // The kernel covers one 128-column tile per launch, so the Op's whole T domain is walked here.
    // for_each_token_slice is NOT this: it only guards CUDA's grid.y ceiling and would hand the
    // kernel the full column count.
    for (std::int32_t col0 = 0; col0 < x.ne[1]; col0 += BN) {
        const std::int32_t count = (x.ne[1] - col0 < BN) ? (x.ne[1] - col0) : BN;
        const Tensor x_slice     = x.slice(1, col0, count);
        Tensor out_slice         = out.slice(1, col0, count);
        std::uint16_t before     = 0;
        if (debug && calls < 4) {
            std::fprintf(stderr,
                         "[swiglu-wmma] #%d x=[%d,%d] out=[%d,%d] col0=%d cnt=%d padded_k=%d "
                         "xdelta=%lld outdelta=%lld grid=%u\n",
                         calls, x.ne[0], x.ne[1], out.ne[0], out.ne[1], col0, count,
                         weight.padded_shape[1],
                         static_cast<long long>(static_cast<const char*>(x_slice.data) -
                                                static_cast<const char*>(x.data)),
                         static_cast<long long>(static_cast<const char*>(out_slice.data) -
                                                static_cast<const char*>(out.data)),
                         grid.x);
            CUDA_CHECK(cudaMemcpy(&before,
                                  static_cast<const std::uint16_t*>(out_slice.data) + 1000,
                                  sizeof(std::uint16_t), cudaMemcpyDeviceToHost));
        }
        q4_linear_swiglu_wmma_kernel<2><<<grid, THREADS, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x_slice.data),
            static_cast<const std::uint8_t*>(weight.qdata),
            static_cast<const std::uint8_t*>(weight.scales),
            static_cast<__nv_bfloat16*>(out_slice.data), out.ne[0], x.ne[0], count,
            weight.padded_shape[1]);
        CUDA_CHECK(cudaGetLastError());
        if (debug && calls < 4) {
            std::uint16_t after = 0;
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaMemcpy(&after, static_cast<const std::uint16_t*>(out_slice.data) + 1000,
                                  sizeof(std::uint16_t), cudaMemcpyDeviceToHost));
            std::fprintf(stderr, "[swiglu-wmma] #%d out[1000] before=0x%04x after=0x%04x\n", calls,
                         before, after);
            ++calls;
        }
    }
}

#endif // NINFER_HIP

} // namespace

void q4_linear_swiglu_mma_split_half_pair_r32_c128_launch(const Tensor& x, const Weight& weight,
                                                          Tensor& out, cudaStream_t stream) {
#if defined(NINFER_HIP)
    if (swiglu_wmma_enabled() && swiglu_wmma_covers(x, weight)) {
        launch_wmma(x, weight, out, stream);
        return;
    }
#endif
    launch_route<GateUpC128Cfg>(x, weight, out, stream);
}

void q4_linear_swiglu_mma_split_half_pair_r32_c40_launch(const Tensor& x, const Weight& weight,
                                                         Tensor& out, cudaStream_t stream) {
    launch_route<GateUpC40Cfg>(x, weight, out, stream);
}

void q4_linear_swiglu_mma_split_half_pair_r32_c48_launch(const Tensor& x, const Weight& weight,
                                                         Tensor& out, cudaStream_t stream) {
    launch_route<GateUpC48Cfg>(x, weight, out, stream);
}

} // namespace ninfer::ops::detail

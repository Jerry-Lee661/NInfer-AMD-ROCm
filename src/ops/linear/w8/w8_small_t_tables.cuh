#pragma once

// Shared templates for the per-geometry W8 small-T launcher tables. Each geometry's TU only
// states which table it owns and over which active-token range; the kernel instantiation and
// its static_asserts live here.

#include "ops/linear/w8/w8_launch.h"

#include "core/device.h"
#include "ops/linear/w8/w8_config.h"
#include "ops/linear/w8/w8_small_t_mma.cuh"

#include <array>
#include <cstddef>
#include <utility>

namespace ninfer::ops::detail {

template <class Geometry, int ActiveTokens>
void launch_exact(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    using Schedule = typename W8LinearSmallTProductionSchedule<Geometry, ActiveTokens>::Type;
    static_assert((Geometry::kOutputRows % Schedule::kRowsPerCta) == 0);
    static_assert((Geometry::kInputRows % Schedule::kGroupK) == 0);

    const W8ContiguousOutput output{static_cast<__nv_bfloat16*>(out.data), Geometry::kOutputRows};
    constexpr int kBlocks = Geometry::kOutputRows / Schedule::kRowsPerCta;
    w8_small_t_mma_kernel<Geometry, ActiveTokens, Schedule>
        <<<kBlocks, Schedule::kThreads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const std::uint8_t*>(weight.qdata),
            static_cast<const std::uint8_t*>(weight.scales), output);
    CUDA_CHECK(cudaGetLastError());
}

template <class Geometry, int First, std::size_t... Offsets>
constexpr auto make_launchers(std::index_sequence<Offsets...>) {
    return std::array<W8Launch, sizeof...(Offsets)>{
        &launch_exact<Geometry, First + static_cast<int>(Offsets)>...};
}

template <class Geometry, int First, int Last>
constexpr auto make_small_t_launchers() {
    return make_launchers<Geometry, First>(
        std::make_index_sequence<static_cast<std::size_t>(Last - First + 1)>{});
}

} // namespace ninfer::ops::detail

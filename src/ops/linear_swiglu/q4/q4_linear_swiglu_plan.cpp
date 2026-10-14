#include "ops/linear_swiglu/q4/q4_linear_swiglu_plan.h"

#include "ninfer/ops/linear.h"
#include "ninfer/ops/silu_mul.h"
#include "core/device.h"
#include "core/layout.h"
#include "ops/linear_swiglu/q4/q4_linear_swiglu_kernels.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

#include <cuda_runtime.h>

namespace ninfer::ops::detail {
namespace {

constexpr std::int32_t kAnyCols = std::numeric_limits<std::int32_t>::max();

struct ColsSet {
    std::int32_t first;
    std::int32_t last;

    constexpr bool contains(std::int32_t cols) const noexcept {
        return cols >= first && cols <= last;
    }
};

struct RouteSpec {
    ColsSet cols;
    Q4LinearSwiGluScheduleId schedule;
};

constexpr Q4LinearSwiGluProblem kShape{34816, 17408, 5120, 5120, 1};

constexpr std::array<RouteSpec, 10> kRoutes{{
    {{1, 1}, Q4LinearSwiGluScheduleId::GemvPair},
    {{2, 32}, Q4LinearSwiGluScheduleId::SmallTExact},
    {{33, 40}, Q4LinearSwiGluScheduleId::MmaSplitHalfPairR32C40},
    {{41, 48}, Q4LinearSwiGluScheduleId::MmaSplitHalfPairR32C48},
    {{49, 128}, Q4LinearSwiGluScheduleId::Materialized},
    {{129, 256}, Q4LinearSwiGluScheduleId::MmaSplitHalfPairR32C128},
    {{257, 384}, Q4LinearSwiGluScheduleId::Materialized},
    {{385, 512}, Q4LinearSwiGluScheduleId::MmaSplitHalfPairR32C128},
    {{513, 640}, Q4LinearSwiGluScheduleId::Materialized},
    {{641, kAnyCols}, Q4LinearSwiGluScheduleId::MmaSplitHalfPairR32C128},
}};

constexpr bool catalog_is_closed() noexcept {
    std::int64_t expected = 1;
    for (const RouteSpec& route : kRoutes) {
        if (route.cols.first != expected || route.cols.last < route.cols.first) { return false; }
        expected = static_cast<std::int64_t>(route.cols.last) + 1;
    }
    return kRoutes.back().cols.last == kAnyCols &&
           expected == static_cast<std::int64_t>(kAnyCols) + 1;
}

static_assert(catalog_is_closed(), "Q4 LinearSwiGLU routes must be exact, contiguous, and closed");

bool supported_shape(const Q4LinearSwiGluProblem& problem) noexcept {
    return problem.gate_up_rows == kShape.gate_up_rows &&
           problem.output_rows == kShape.output_rows && problem.k == kShape.k &&
           problem.padded_k == kShape.padded_k;
}

template <class Allocator>
Tensor allocate_materialized_workspace(Allocator& allocator, std::int32_t rows, std::int32_t cols) {
    return allocator.alloc(DType::BF16, {rows, cols});
}

std::size_t materialized_workspace_bytes(std::int32_t rows, std::int32_t cols) {
    WorkspaceLayoutBuilder layout;
    (void)allocate_materialized_workspace(layout, rows, cols);
    return layout.peak_bytes(1);
}

} // namespace

const char* q4_linear_swiglu_schedule_name(Q4LinearSwiGluScheduleId schedule) noexcept {
    switch (schedule) {
    case Q4LinearSwiGluScheduleId::GemvPair:
        return "linear_swiglu.q4.gemv.paired_rows";
    case Q4LinearSwiGluScheduleId::SmallTExact:
        return "linear_swiglu.q4.mma.small_t.exact";
    case Q4LinearSwiGluScheduleId::MmaSplitHalfPairR32C40:
        return "linear_swiglu.q4.mma.split_half_pair.r32.c40";
    case Q4LinearSwiGluScheduleId::MmaSplitHalfPairR32C48:
        return "linear_swiglu.q4.mma.split_half_pair.r32.c48";
    case Q4LinearSwiGluScheduleId::Materialized:
        return "linear_swiglu.q4.materialized";
    case Q4LinearSwiGluScheduleId::MmaSplitHalfPairR32C128:
        return "linear_swiglu.q4.mma.split_half_pair.r32.c128";
    case Q4LinearSwiGluScheduleId::SwigluNarrowR16:
        return "linear_swiglu.q4.wmma.narrow.r16";
    }
    return "linear_swiglu.q4.unknown";
}

bool q4_linear_swiglu_admits(const Q4LinearSwiGluProblem& problem) noexcept {
    return supported_shape(problem) && problem.cols >= 1;
}

Q4LinearSwiGluPlan q4_linear_swiglu_resolve_plan(const Q4LinearSwiGluProblem& problem) {
    if (!q4_linear_swiglu_admits(problem)) {
        throw std::invalid_argument(
            "q4 linear_swiglu: exact problem or column count is not admitted");
    }

    // 窄核门（NINFER_SWIGLU_NARROW=1，实验）：小 T（2..16 列）走 swiglu 专用窄核
    // （q4_swiglu_small_t_wmma_kernel，gdn v5-final 模式 + silu*up 融合收尾）。
    static const int swiglu_narrow = [] {
        const char* value = std::getenv("NINFER_SWIGLU_NARROW");
        return value != nullptr && std::atoi(value) != 0;
    }();
    if (swiglu_narrow && problem.cols >= 2 && problem.cols <= 16 &&
        problem.padded_k % 64 == 0) {
        return {Q4LinearSwiGluScheduleId::SwigluNarrowR16};
    }

    for (const RouteSpec& route : kRoutes) {
        if (!route.cols.contains(problem.cols)) { continue; }
        Q4LinearSwiGluPlan plan{
            route.schedule,
            0,
        };
        switch (route.schedule) {
        case Q4LinearSwiGluScheduleId::GemvPair:
        case Q4LinearSwiGluScheduleId::SmallTExact:
        case Q4LinearSwiGluScheduleId::MmaSplitHalfPairR32C40:
        case Q4LinearSwiGluScheduleId::MmaSplitHalfPairR32C48:
            return plan;
        case Q4LinearSwiGluScheduleId::Materialized:
            plan.workspace_bytes = materialized_workspace_bytes(problem.gate_up_rows, problem.cols);
            return plan;
        case Q4LinearSwiGluScheduleId::MmaSplitHalfPairR32C128:
            return plan;
        }
    }
    throw std::logic_error("q4 linear_swiglu: admitted problem has no covering route");
}

std::size_t q4_linear_swiglu_capacity_workspace_bytes(std::int32_t gate_up_rows,
                                                      std::int32_t output_rows, std::int32_t k,
                                                      std::int32_t padded_k, std::int32_t min_cols,
                                                      std::int32_t max_cols) {
    if (min_cols <= 0 || max_cols < min_cols) {
        throw std::invalid_argument("q4 linear_swiglu: invalid column interval");
    }
    (void)q4_linear_swiglu_resolve_plan({gate_up_rows, output_rows, k, padded_k, min_cols});
    (void)q4_linear_swiglu_resolve_plan({gate_up_rows, output_rows, k, padded_k, max_cols});

    std::size_t maximum = 0;
    for (const RouteSpec& route : kRoutes) {
        if (route.cols.last < min_cols || route.cols.first > max_cols) { continue; }
        const std::int32_t endpoint = std::min(route.cols.last, max_cols);
        maximum                     = std::max(maximum, q4_linear_swiglu_resolve_plan(
                                        {gate_up_rows, output_rows, k, padded_k, endpoint})
                                                            .workspace_bytes);
    }
    return maximum;
}

void q4_linear_swiglu_execute_plan(const Q4LinearSwiGluPlan& plan, const Tensor& x, const Weight& w,
                                   Tensor& out, WorkspaceArena& ws, cudaStream_t stream) {
    const Q4LinearSwiGluProblem problem{w.n, out.ne[0], x.ne[0], w.padded_shape[1], x.ne[1]};
    const Q4LinearSwiGluPlan resolved = q4_linear_swiglu_resolve_plan(problem);
    if (resolved.schedule != plan.schedule || resolved.workspace_bytes != plan.workspace_bytes) {
        throw std::invalid_argument("q4 linear_swiglu: plan does not match the exact problem");
    }

    switch (plan.schedule) {
    case Q4LinearSwiGluScheduleId::GemvPair:
        q4_linear_swiglu_gemv_pair_launch(x, w, out, stream);
        return;
    case Q4LinearSwiGluScheduleId::SmallTExact:
        q4_linear_swiglu_small_t_exact_launch(x, w, out, stream);
        return;
    case Q4LinearSwiGluScheduleId::MmaSplitHalfPairR32C40:
        q4_linear_swiglu_mma_split_half_pair_r32_c40_launch(x, w, out, stream);
        return;
    case Q4LinearSwiGluScheduleId::MmaSplitHalfPairR32C48:
        q4_linear_swiglu_mma_split_half_pair_r32_c48_launch(x, w, out, stream);
        return;
    case Q4LinearSwiGluScheduleId::Materialized: {
        auto scratch_scope = ws.scope();
        Tensor gate_up = allocate_materialized_workspace(ws, problem.gate_up_rows, problem.cols);
        linear(x, w, gate_up, stream);
        silu_mul(gate_up.slice(0, 0, problem.output_rows),
                 gate_up.slice(0, problem.output_rows, problem.output_rows), out, stream);
        return;
    }
    case Q4LinearSwiGluScheduleId::MmaSplitHalfPairR32C128:
        q4_linear_swiglu_mma_split_half_pair_r32_c128_launch(x, w, out, stream);
        return;
    case Q4LinearSwiGluScheduleId::SwigluNarrowR16: {
        // 就地对照（NINFER_SWIGLU_NARROW_DEBUG=1）：同一调用里窄核写 out、SIMT 基线写副本，
        // 前 3 次调用打印逐元素 diff 统计与样本，用来切开"内核算错"与"引擎别处污染"。
        static const int narrow_debug = [] {
            const char* value = std::getenv("NINFER_SWIGLU_NARROW_DEBUG");
            return value != nullptr && std::atoi(value) != 0;
        }();
        if (!narrow_debug) {
            q4_swiglu_small_t_wmma_launch(x, w, out, stream);
            return;
        }
        static int calls = 0;
        const bool report = calls < 3;
        auto scratch_scope = ws.scope();
        Tensor baseline = allocate_materialized_workspace(ws, problem.output_rows, problem.cols);
        q4_swiglu_small_t_wmma_launch(x, w, out, stream);
        q4_linear_swiglu_small_t_exact_launch(x, w, baseline, stream);
        if (report) {
            const std::size_t n = static_cast<std::size_t>(problem.cols) * problem.output_rows;
            std::vector<__nv_bfloat16> a(n), b(n);
            CUDA_CHECK(cudaMemcpy(a.data(), out.data, n * 2, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(b.data(), baseline.data, n * 2, cudaMemcpyDeviceToHost));
            std::size_t bad = 0;
            double worst = 0.0;
            std::size_t first_bad = n;
            for (std::size_t i = 0; i < n; ++i) {
                const double da = __bfloat162float(a[i]);
                const double db = __bfloat162float(b[i]);
                const double diff = std::fabs(da - db);
                if (diff > 0.05 + 0.01 * std::fabs(db)) {
                    if (bad < 4) {
                        const std::size_t col = i / problem.output_rows;
                        const std::size_t row = i % problem.output_rows;
                        std::fprintf(stderr,
                                     "narrow-debug call=%d layer-call[%zu] col=%zu row=%zu "
                                     "narrow=%.4f simt=%.4f\n",
                                     calls, i, col, row, da, db);
                    }
                    ++bad;
                    worst = std::max(worst, diff);
                    first_bad = std::min(first_bad, i);
                }
            }
            std::fprintf(stderr,
                         "narrow-debug call=%d n=%zu bad=%zu worst=%.4f first_bad=%zu "
                         "(row=%zu col=%zu) out_bits=%08x simt_bits=%08x\n",
                         calls, n, bad, worst, first_bad, first_bad % problem.output_rows,
                         first_bad / problem.output_rows,
                         *reinterpret_cast<const std::uint32_t*>(&a[0]),
                         *reinterpret_cast<const std::uint32_t*>(&b[0]));
            ++calls;
        }
        return;
    }
    }
    throw std::logic_error("q4 linear_swiglu: unknown schedule");
}

void q4_linear_swiglu_dispatch(const Tensor& x, const Weight& w, Tensor& out, WorkspaceArena& ws,
                               cudaStream_t stream) {
    const Q4LinearSwiGluProblem problem{w.n, out.ne[0], x.ne[0], w.padded_shape[1], x.ne[1]};
    const Q4LinearSwiGluPlan plan = q4_linear_swiglu_resolve_plan(problem);
    q4_linear_swiglu_execute_plan(plan, x, w, out, ws, stream);
}

} // namespace ninfer::ops::detail

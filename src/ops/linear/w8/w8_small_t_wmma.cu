#include "ops/linear/w8/w8_launch.h"

#include "core/device.h"
#include "ops/linear/w8/w8_config.h"
#include "ops/linear/w8/w8_small_t_wmma.cuh"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <utility>

namespace ninfer::ops::detail {
namespace {

// 16-token N-tile + Direct scale access：T ≤ 16 时一个 WMMA N-tile 装得下全部列。
// T > 16（列数超一个 tile）不走这条路，回退到 launch_w8_small_t 的仿真路径。
using WmmaSchedule = W8SmallTMmaSchedule<8, 16, 2, W8SmallTMmaScaleAccess::Direct>;

template <class Geometry, int ActiveTokens>
void launch_wmma_exact(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    using Schedule = WmmaSchedule;
    static_assert((Geometry::kOutputRows % Schedule::kRowsPerCta) == 0);
    static_assert((Geometry::kInputRows % Schedule::kGroupK) == 0);
    static_assert(ActiveTokens <= Schedule::kTileTokens);

    // 激活全局直读（NINFER_W8_XDIRECT，默认开）：v4 配方，A 操作数绕过 LDS 暂存。
    // 已默认开（2026-09-29）：同二进制 C8 134.0→135.4（+1.0%，两轮复现）、mtp3 CLI
    // 39.04→40.69（+4.2%）、C1 mtp0/mtp3 32.7/62.0、T3 PPL 4.975734 逐位一致、
    // mtp3 输出与基线逐字相同（值恒等改动：装载源变、值与序不变）；=0 回退 LDS 暂存版。
    static const bool xdirect = [] {
        const char* value = std::getenv("NINFER_W8_XDIRECT");
        return value == nullptr || std::atoi(value) != 0;
    }();

    const W8ContiguousOutput output{static_cast<__nv_bfloat16*>(out.data),
                                    Geometry::kOutputRows};
    constexpr int kBlocks = Geometry::kOutputRows / Schedule::kRowsPerCta;
    if (xdirect) {
        w8_small_t_wmma_kernel<Geometry, ActiveTokens, Schedule, W8ContiguousOutput, true>
            <<<kBlocks, Schedule::kThreads, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(x.data),
                static_cast<const std::uint8_t*>(weight.qdata),
                static_cast<const std::uint8_t*>(weight.scales), output);
    } else {
        w8_small_t_wmma_kernel<Geometry, ActiveTokens, Schedule, W8ContiguousOutput, false>
            <<<kBlocks, Schedule::kThreads, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(x.data),
                static_cast<const std::uint8_t*>(weight.qdata),
                static_cast<const std::uint8_t*>(weight.scales), output);
    }
    CUDA_CHECK(cudaGetLastError());
}

template <class Geometry, int First, std::size_t... Offsets>
constexpr auto make_wmma_launchers(std::index_sequence<Offsets...>) {
    return std::array<W8Launch, sizeof...(Offsets)>{
        &launch_wmma_exact<Geometry, First + static_cast<int>(Offsets)>...};
}

template <class Geometry>
constexpr auto make_wmma_table() {
    return make_wmma_launchers<Geometry, 1>(
        std::make_index_sequence<kW8SmallTWmmaMaxTokens>{});
}

// 七个走 small_t 的几何各一张 T=1..16 的表。MTP 的验证趟是 T = draft+1（窗口 3 时 T=4），
// 落在覆盖内；这六个几何原本全走仿真 mma，是 MTP 之前净亏的主因。
constexpr auto kWmmaVocabularyLaunchers   = make_wmma_table<W8VocabularyProjectionGeometry>();
constexpr auto kWmmaMtpInputLaunchers     = make_wmma_table<W8MtpInputProjectionGeometry>();
constexpr auto kWmmaMtpAttentionLaunchers = make_wmma_table<W8MtpAttentionProjectionGeometry>();
constexpr auto kWmmaMtpAttentionOutputLaunchers =
    make_wmma_table<W8MtpAttentionOutputGeometry>();
constexpr auto kWmmaMtpGateUpLaunchers = make_wmma_table<W8MtpGateUpProjectionGeometry>();
constexpr auto kWmmaMtpDownLaunchers   = make_wmma_table<W8MtpDownProjectionGeometry>();
constexpr auto kWmma35bLaunchers       = make_wmma_table<W835bMtpProjectionGeometry>();

// 形状 → 表。padded_shape[1] 必须等于输入行数（与 launch_w8_small_t 的判据一致）。
struct WmmaTable {
    std::int32_t output_rows;
    std::int32_t input_rows;
    const W8Launch* launchers;
};

constexpr std::array<WmmaTable, 7> kWmmaTables{{
    {W8VocabularyProjectionGeometry::kOutputRows, W8VocabularyProjectionGeometry::kInputRows,
     kWmmaVocabularyLaunchers.data()},
    {W8MtpInputProjectionGeometry::kOutputRows, W8MtpInputProjectionGeometry::kInputRows,
     kWmmaMtpInputLaunchers.data()},
    {W8MtpAttentionProjectionGeometry::kOutputRows, W8MtpAttentionProjectionGeometry::kInputRows,
     kWmmaMtpAttentionLaunchers.data()},
    {W8MtpAttentionOutputGeometry::kOutputRows, W8MtpAttentionOutputGeometry::kInputRows,
     kWmmaMtpAttentionOutputLaunchers.data()},
    {W8MtpGateUpProjectionGeometry::kOutputRows, W8MtpGateUpProjectionGeometry::kInputRows,
     kWmmaMtpGateUpLaunchers.data()},
    {W8MtpDownProjectionGeometry::kOutputRows, W8MtpDownProjectionGeometry::kInputRows,
     kWmmaMtpDownLaunchers.data()},
    {W835bMtpProjectionGeometry::kOutputRows, W835bMtpProjectionGeometry::kInputRows,
     kWmma35bLaunchers.data()},
}};

} // namespace

bool w8_small_t_wmma_covers(std::int32_t n, std::int32_t k) {
    for (const WmmaTable& table : kWmmaTables) {
        if (n == table.output_rows && k == table.input_rows) { return true; }
    }
    return false;
}

void launch_w8_small_t_wmma(const Tensor& x, const Weight& weight, Tensor& out,
                            cudaStream_t stream) {
    const std::int32_t tokens = static_cast<std::int32_t>(x.ne[1]);
    if (tokens >= 1 && tokens <= kW8SmallTWmmaMaxTokens) {
        const std::size_t index = static_cast<std::size_t>(tokens - 1);
        for (const WmmaTable& table : kWmmaTables) {
            if (weight.n == table.output_rows && weight.k == table.input_rows &&
                weight.padded_shape[1] == table.input_rows) {
                table.launchers[index](x, weight, out, stream);
                return;
            }
        }
    }
    launch_w8_small_t(x, weight, out, stream);  // fallback: emulated path
}

} // namespace ninfer::ops::detail

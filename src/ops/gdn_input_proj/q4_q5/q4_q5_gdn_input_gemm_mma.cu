#include "ops/gdn_input_proj/q4_q5/q4_q5_gdn_input_kernels.h"

#include "core/device.h"
#include "ops/common/math.h"
#include "ops/common/rowsplit_grouped_mma.cuh"
#include "ops/common/q4_q5_grouped_wmma.cuh"
#include "ops/common/q4_q5_small_t_wmma.cuh"
#include "ops/common/token_slices.h"

#include <cstdint>
#include <cstdlib>

namespace ninfer::ops::detail {
namespace {

RowSplitGroupedMmaJob make_job(const Weight& weight, std::int32_t weight_row_offset,
                               std::int32_t rows, Tensor& out, std::int32_t output_row_offset) {
    const std::int64_t groups = weight.padded_shape[1] / weight.group;
    const auto* codes         = static_cast<const std::uint8_t*>(weight.qdata) +
                        static_cast<std::int64_t>(weight_row_offset) * groups * 32;
    const auto* high   = weight.qtype == QType::Q5G64_F16S
                             ? static_cast<const std::uint8_t*>(weight.qhigh) +
                                 static_cast<std::int64_t>(weight_row_offset) * groups * 8
                             : nullptr;
    const auto* scales = static_cast<const std::uint8_t*>(weight.scales) +
                         static_cast<std::int64_t>(weight_row_offset) * groups * 2;
    return RowSplitGroupedMmaJob{
        codes,
        high,
        scales,
        static_cast<__nv_bfloat16*>(out.data),
        rows,
        out.ne[0],
        output_row_offset,
        weight.qtype == QType::Q5G64_F16S,
    };
}

void launch_slice(bool full, const Tensor& x, const Weight& qk_weight, const Weight& value_z_weight,
                  Tensor& qkv, Tensor& z, cudaStream_t stream) {
    constexpr std::int32_t kValueRows = 6144;
    using Schedule                    = GemmCfg<64, 128, 64, 64, 16, 2, 1, false, true, true>;
    const RowSplitGroupedMmaJob qk    = make_job(qk_weight, 0, qk_weight.n, qkv, 0);
    const RowSplitGroupedMmaJob value = make_job(value_z_weight, 0, kValueRows, qkv, qk_weight.n);
    const RowSplitGroupedMmaJob output_gate =
        make_job(value_z_weight, kValueRows, kValueRows, z, 0);
    RowSplitGroupedMmaJob empty{};
    const int tiles = div_up(qk.n, Schedule::BM) + div_up(value.n, Schedule::BM) +
                      div_up(output_gate.n, Schedule::BM);
    const int cols = x.ne[1];
    const dim3 grid(static_cast<unsigned>(tiles),
                    static_cast<unsigned>(div_up(cols, Schedule::BN)));

    if (full) {
        rowsplit_grouped_mma_kernel<Schedule, true, RowSplitGroupedMmaCodec::Mixed, 4>
            <<<grid, Schedule::THREADS, 0, stream>>>(static_cast<const __nv_bfloat16*>(x.data), qk,
                                                     value, output_gate, empty, x.ne[0], cols,
                                                     x.ne[0]);
    } else {
        rowsplit_grouped_mma_kernel<Schedule, false, RowSplitGroupedMmaCodec::Mixed, 4>
            <<<grid, Schedule::THREADS, 0, stream>>>(static_cast<const __nv_bfloat16*>(x.data), qk,
                                                     value, output_gate, empty, x.ne[0], cols,
                                                     x.ne[0]);
    }
    CUDA_CHECK(cudaGetLastError());
}

// WMMA variant of the grouped Mixed-codec kernel (the GDN input projection is the prefill's single
// biggest op: amd-research.md §3.7.5). On by default: the numeric gate passes (T3 PPL 4.969047) and
// the arm is 1.78x the prefill rate. NINFER_GDN_INPUT_WMMA=0 falls back to the emulated kernel.
bool gdn_input_wmma_enabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("NINFER_GDN_INPUT_WMMA");
        return value == nullptr || std::atoi(value) != 0;
    }();
    return enabled;
}

void launch_slice_wmma(const Tensor& x, const Weight& qk_weight, const Weight& value_z_weight,
                       Tensor& qkv, Tensor& z, cudaStream_t stream) {
    constexpr std::int32_t kValueRows = 6144;
    constexpr int BM                  = grouped_wmma_cfg::kBM;
    constexpr int BN                  = grouped_wmma_cfg::kBN;
    constexpr int THREADS             = grouped_wmma_cfg::kThreads;
    const RowSplitGroupedMmaJob qk    = make_job(qk_weight, 0, qk_weight.n, qkv, 0);
    const RowSplitGroupedMmaJob value = make_job(value_z_weight, 0, kValueRows, qkv, qk_weight.n);
    const RowSplitGroupedMmaJob output_gate =
        make_job(value_z_weight, kValueRows, kValueRows, z, 0);
    RowSplitGroupedMmaJob empty{};
    const int tiles = div_up(qk.n, BM) + div_up(value.n, BM) + div_up(output_gate.n, BM);
    const int cols  = x.ne[1];
    const dim3 grid(static_cast<unsigned>(tiles), static_cast<unsigned>(div_up(cols, BN)));
    q4_q5_grouped_wmma_kernel<2><<<grid, THREADS, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), qk, value, output_gate, empty, x.ne[0], cols,
        x.ne[0]);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

// Narrow small-T arm (NINFER_GDN_WMMA_NARROW): the same three jobs, one 16-column WMMA
// N-tile, row tiles of 16 flattened into blockIdx.x. Decode shapes only (T in [2,16]).
void q4_q5_gdn_input_wmma_narrow_launch(const Tensor& x, const Weight& qk_weight,
                                        const Weight& value_z_weight, Tensor& qkv, Tensor& z,
                                        cudaStream_t stream) {
    constexpr std::int32_t kValueRows = 6144;
    constexpr int ROWS                = q4_q5_small_t_cfg::kRows;
    const RowSplitGroupedMmaJob qk    = make_job(qk_weight, 0, qk_weight.n, qkv, 0);
    const RowSplitGroupedMmaJob value = make_job(value_z_weight, 0, kValueRows, qkv, qk_weight.n);
    const RowSplitGroupedMmaJob output_gate =
        make_job(value_z_weight, kValueRows, kValueRows, z, 0);
    RowSplitGroupedMmaJob empty{};
    const int tiles = div_up(qk.n, ROWS) + div_up(value.n, ROWS) + div_up(output_gate.n, ROWS);
    q4_q5_small_t_wmma_kernel<<<tiles, q4_q5_small_t_cfg::kThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), qk, value, output_gate, empty, x.ne[0],
        x.ne[1], x.ne[0]);
    CUDA_CHECK(cudaGetLastError());
}

void q4_q5_gdn_input_grouped_mma_launch(const Tensor& x, const Weight& qk_weight,
                                        const Weight& value_z_weight, Tensor& qkv, Tensor& z,
                                        cudaStream_t stream) {
    constexpr std::int32_t kTileCols = 128;
    const bool full                  = (x.ne[1] % kTileCols) == 0;
    for_each_token_slice(x.ne[1], kTileCols, [&](std::int32_t offset, std::int32_t count) {
        const Tensor x_slice = x.slice(1, offset, count);
        Tensor qkv_slice     = qkv.slice(1, offset, count);
        Tensor z_slice       = z.slice(1, offset, count);
        if (gdn_input_wmma_enabled() && full && qk_weight.k == qk_weight.padded_shape[1] &&
            value_z_weight.k == value_z_weight.padded_shape[1] && (qk_weight.k % 64) == 0) {
            launch_slice_wmma(x_slice, qk_weight, value_z_weight, qkv_slice, z_slice, stream);
            return;
        }
        launch_slice(full, x_slice, qk_weight, value_z_weight, qkv_slice, z_slice, stream);
    });
}

} // namespace ninfer::ops::detail

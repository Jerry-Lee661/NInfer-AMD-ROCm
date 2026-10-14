#include "ops/linear/w8/w8_launch.h"

#include "ops/linear/w8/w8_config.h"
#include "ops/linear/w8/w8_small_t_tables.h"

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {

// Dispatch only: each geometry's launcher table lives in its own TU (see
// w8_small_t_tables.h for why the tables are split). Keeping the device templates out of
// this TU means the dispatcher itself has no kernel code to compile.
void launch_w8_small_t(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    if (weight.n == W8VocabularyProjectionGeometry::kOutputRows &&
        weight.k == W8VocabularyProjectionGeometry::kInputRows &&
        weight.padded_shape[1] == W8VocabularyProjectionGeometry::kInputRows &&
        x.ne[1] >= kW8VocabularyFirstSmallT && x.ne[1] <= kW8VocabularyLastSmallT) {
        w8_small_t_table_vocab(static_cast<std::int32_t>(x.ne[1]))(x, weight, out, stream);
        return;
    }
    if (weight.n == W8MtpInputProjectionGeometry::kOutputRows &&
        weight.k == W8MtpInputProjectionGeometry::kInputRows &&
        weight.padded_shape[1] == W8MtpInputProjectionGeometry::kInputRows &&
        x.ne[1] >= kW8MtpInputFirstSmallT && x.ne[1] <= kW8MtpInputLastSmallT) {
        w8_small_t_table_mtp_input(static_cast<std::int32_t>(x.ne[1]))(x, weight, out, stream);
        return;
    }
    if (weight.n == W8MtpAttentionProjectionGeometry::kOutputRows &&
        weight.k == W8MtpAttentionProjectionGeometry::kInputRows &&
        weight.padded_shape[1] == W8MtpAttentionProjectionGeometry::kInputRows &&
        x.ne[1] >= kW8MtpAttentionFirstSmallT && x.ne[1] <= kW8MtpAttentionLastSmallT) {
        w8_small_t_table_mtp_attention(static_cast<std::int32_t>(x.ne[1]))(x, weight, out, stream);
        return;
    }
    if (weight.n == W8MtpAttentionOutputGeometry::kOutputRows &&
        weight.k == W8MtpAttentionOutputGeometry::kInputRows &&
        weight.padded_shape[1] == W8MtpAttentionOutputGeometry::kInputRows &&
        x.ne[1] >= kW8MtpAttentionOutputFirstSmallT && x.ne[1] <= kW8MtpAttentionOutputLastSmallT) {
        w8_small_t_table_mtp_attention_output(static_cast<std::int32_t>(x.ne[1]))(x, weight, out,
                                                                                  stream);
        return;
    }
    if (weight.n == W8MtpGateUpProjectionGeometry::kOutputRows &&
        weight.k == W8MtpGateUpProjectionGeometry::kInputRows &&
        weight.padded_shape[1] == W8MtpGateUpProjectionGeometry::kInputRows &&
        x.ne[1] >= kW8MtpGateUpFirstSmallT && x.ne[1] <= kW8MtpGateUpLastSmallT) {
        w8_small_t_table_mtp_gate_up(static_cast<std::int32_t>(x.ne[1]))(x, weight, out, stream);
        return;
    }
    if (weight.n == W8MtpDownProjectionGeometry::kOutputRows &&
        weight.k == W8MtpDownProjectionGeometry::kInputRows &&
        weight.padded_shape[1] == W8MtpDownProjectionGeometry::kInputRows &&
        x.ne[1] >= kW8MtpDownFirstSmallT && x.ne[1] <= kW8MtpDownLastSmallT) {
        w8_small_t_table_mtp_down(static_cast<std::int32_t>(x.ne[1]))(x, weight, out, stream);
        return;
    }
    if (weight.n == W835bMtpProjectionGeometry::kOutputRows &&
        weight.k == W835bMtpProjectionGeometry::kInputRows &&
        weight.padded_shape[1] == W835bMtpProjectionGeometry::kInputRows &&
        x.ne[1] >= kW835bMtpProjectionFirstSmallT && x.ne[1] <= kW835bMtpProjectionLastSmallT) {
        w8_small_t_table_35b_mtp_projection(static_cast<std::int32_t>(x.ne[1]))(x, weight, out,
                                                                                stream);
        return;
    }
    throw std::invalid_argument("W8 Linear small-T: unsupported exact problem");
}

} // namespace ninfer::ops::detail

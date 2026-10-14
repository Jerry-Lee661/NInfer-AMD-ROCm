#include "ops/linear/w8/w8_small_t_tables.h"

#include "ops/linear/w8/w8_small_t_tables.cuh"

namespace ninfer::ops::detail {
namespace {

constexpr auto kTable = make_small_t_launchers<W8MtpAttentionOutputGeometry,
                                               kW8MtpAttentionOutputFirstSmallT,
                                               kW8MtpAttentionOutputLastSmallT / 2>();

} // namespace

W8Launch w8_small_t_table_mtp_attention_output_lo(std::int32_t t) {
    return kTable[static_cast<std::size_t>(t - kW8MtpAttentionOutputFirstSmallT)];
}

} // namespace ninfer::ops::detail

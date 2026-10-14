#include "ops/linear/w8/w8_small_t_tables.h"

#include "ops/linear/w8/w8_small_t_tables.cuh"

namespace ninfer::ops::detail {
namespace {

constexpr std::int32_t kFirst = kW8MtpAttentionLastSmallT / 2 + 1;
constexpr auto kTable          = make_small_t_launchers<W8MtpAttentionProjectionGeometry, kFirst,
                                                        kW8MtpAttentionLastSmallT>();

} // namespace

W8Launch w8_small_t_table_mtp_attention_hi(std::int32_t t) {
    return kTable[static_cast<std::size_t>(t - kFirst)];
}

} // namespace ninfer::ops::detail

#include "ops/linear/w8/w8_small_t_tables.h"

#include "ops/linear/w8/w8_small_t_tables.cuh"

namespace ninfer::ops::detail {
namespace {

constexpr auto kTable = make_small_t_launchers<W8MtpDownProjectionGeometry,
                                               kW8MtpDownFirstSmallT,
                                               kW8MtpDownLastSmallT>();

} // namespace

W8Launch w8_small_t_table_mtp_down(std::int32_t t) {
    return kTable[static_cast<std::size_t>(t - kW8MtpDownFirstSmallT)];
}

} // namespace ninfer::ops::detail

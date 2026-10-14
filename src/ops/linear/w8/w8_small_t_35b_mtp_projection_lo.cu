#include "ops/linear/w8/w8_small_t_tables.h"

#include "ops/linear/w8/w8_small_t_tables.cuh"

namespace ninfer::ops::detail {
namespace {

constexpr auto kTable = make_small_t_launchers<W835bMtpProjectionGeometry,
                                               kW835bMtpProjectionFirstSmallT,
                                               kW835bMtpProjectionLastSmallT / 2>();

} // namespace

W8Launch w8_small_t_table_35b_mtp_projection_lo(std::int32_t t) {
    return kTable[static_cast<std::size_t>(t - kW835bMtpProjectionFirstSmallT)];
}

} // namespace ninfer::ops::detail

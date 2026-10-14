#include "ops/linear/w8/w8_small_t_tables.h"

#include "ops/linear/w8/w8_small_t_tables.cuh"

namespace ninfer::ops::detail {
namespace {

constexpr std::int32_t kFirst = kW835bMtpProjectionLastSmallT / 2 + 1;
constexpr auto kTable          = make_small_t_launchers<W835bMtpProjectionGeometry, kFirst,
                                                        kW835bMtpProjectionLastSmallT>();

} // namespace

W8Launch w8_small_t_table_35b_mtp_projection_hi(std::int32_t t) {
    return kTable[static_cast<std::size_t>(t - kFirst)];
}

} // namespace ninfer::ops::detail

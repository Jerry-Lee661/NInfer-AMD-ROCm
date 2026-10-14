#include "ops/linear/w8/w8_small_t_tables.h"

#include "ops/linear/w8/w8_small_t_tables.cuh"

namespace ninfer::ops::detail {
namespace {

constexpr auto kTable = make_small_t_launchers<W8MtpGateUpProjectionGeometry,
                                               kW8MtpGateUpFirstSmallT,
                                               kW8MtpGateUpLastSmallT / 2>();

} // namespace

W8Launch w8_small_t_table_mtp_gate_up_lo(std::int32_t t) {
    return kTable[static_cast<std::size_t>(t - kW8MtpGateUpFirstSmallT)];
}

} // namespace ninfer::ops::detail

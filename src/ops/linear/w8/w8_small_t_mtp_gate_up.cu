#include "ops/linear/w8/w8_small_t_tables.h"

#include "ops/linear/w8/w8_config.h"

namespace ninfer::ops::detail {
namespace {

// 1..20 | 21..40。
constexpr std::int32_t kMid = kW8MtpGateUpLastSmallT / 2;

} // namespace

W8Launch w8_small_t_table_mtp_gate_up(std::int32_t t) {
    return t <= kMid ? w8_small_t_table_mtp_gate_up_lo(t) : w8_small_t_table_mtp_gate_up_hi(t);
}

} // namespace ninfer::ops::detail

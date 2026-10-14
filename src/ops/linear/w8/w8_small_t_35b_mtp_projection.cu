#include "ops/linear/w8/w8_small_t_tables.h"

#include "ops/linear/w8/w8_config.h"

namespace ninfer::ops::detail {
namespace {

// 1..24 | 25..48。
constexpr std::int32_t kMid = kW835bMtpProjectionLastSmallT / 2;

} // namespace

W8Launch w8_small_t_table_35b_mtp_projection(std::int32_t t) {
    return t <= kMid ? w8_small_t_table_35b_mtp_projection_lo(t)
                     : w8_small_t_table_35b_mtp_projection_hi(t);
}

} // namespace ninfer::ops::detail

#include "ops/linear/w8/w8_small_t_tables.h"

#include "ops/linear/w8/w8_config.h"

namespace ninfer::ops::detail {
namespace {

// 1..24 | 25..48。
constexpr std::int32_t kMid = kW8MtpAttentionOutputLastSmallT / 2;

} // namespace

W8Launch w8_small_t_table_mtp_attention_output(std::int32_t t) {
    return t <= kMid ? w8_small_t_table_mtp_attention_output_lo(t)
                     : w8_small_t_table_mtp_attention_output_hi(t);
}

} // namespace ninfer::ops::detail

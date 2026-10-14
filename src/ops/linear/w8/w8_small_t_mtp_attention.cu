#include "ops/linear/w8/w8_small_t_tables.h"

#include "ops/linear/w8/w8_config.h"

namespace ninfer::ops::detail {
namespace {

// 1..24 | 25..48，两个半表各占一个重 TU（实测 48 个实例化的这个几何要 63 min）。
constexpr std::int32_t kMid = kW8MtpAttentionLastSmallT / 2;

} // namespace

W8Launch w8_small_t_table_mtp_attention(std::int32_t t) {
    return t <= kMid ? w8_small_t_table_mtp_attention_lo(t)
                     : w8_small_t_table_mtp_attention_hi(t);
}

} // namespace ninfer::ops::detail

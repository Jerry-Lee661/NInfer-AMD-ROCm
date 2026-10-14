#pragma once

// Host-side declarations for the per-geometry W8 small-T launcher tables.
//
// Why one TU per geometry: the production kernel selects a *different schedule* per
// active-token count, so the 313 instantiations spread over the seven geometries produce
// highly diverse device code. Held in a single TU that was ~3h of CPU and >10 GiB RSS, and
// losing it to a kill discarded every hour of progress. Split by geometry each piece is a
// separately restartable ~30 min unit that ninja can compile in parallel.
//
// Each accessor indexes a table built for T in [FirstSmallT, LastSmallT] of that geometry;
// the caller must have range-checked t (launch_w8_small_t does).

#include "ops/linear/w8/w8_launch.h"

#include <cstdint>

namespace ninfer::ops::detail {

W8Launch w8_small_t_table_vocab(std::int32_t t);
W8Launch w8_small_t_table_mtp_input(std::int32_t t);
W8Launch w8_small_t_table_mtp_attention(std::int32_t t);
W8Launch w8_small_t_table_mtp_attention_output(std::int32_t t);
W8Launch w8_small_t_table_mtp_gate_up(std::int32_t t);
W8Launch w8_small_t_table_mtp_down(std::int32_t t);
W8Launch w8_small_t_table_35b_mtp_projection(std::int32_t t);

// 四个"贵"几何（实测单 TU 46-63 min，而 mtp_input/mtp_down 同样 48 个实例化只要 1.6 min——
// 成本由 schedule 决定，不由实例化个数决定）再按 T 区间对半拆，各占一个编译单元：
// lo = [First, Last/2]，hi = [Last/2+1, Last]。上面那个同名 accessor 退化成极薄的路由。
W8Launch w8_small_t_table_mtp_attention_lo(std::int32_t t);
W8Launch w8_small_t_table_mtp_attention_hi(std::int32_t t);
W8Launch w8_small_t_table_mtp_attention_output_lo(std::int32_t t);
W8Launch w8_small_t_table_mtp_attention_output_hi(std::int32_t t);
W8Launch w8_small_t_table_mtp_gate_up_lo(std::int32_t t);
W8Launch w8_small_t_table_mtp_gate_up_hi(std::int32_t t);
W8Launch w8_small_t_table_35b_mtp_projection_lo(std::int32_t t);
W8Launch w8_small_t_table_35b_mtp_projection_hi(std::int32_t t);

} // namespace ninfer::ops::detail

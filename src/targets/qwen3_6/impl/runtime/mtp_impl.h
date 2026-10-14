#include "targets/qwen3_6/impl/runtime/instance.h"
#include "targets/qwen3_6/impl/runtime/schedule.h"

#include "decode_profile.h"

#include "ninfer/ops/mtp_round.h"
#include "ninfer/ops/scatter.h"
#include "ninfer/ops/scalar.h"

#include <cuda_runtime.h>

#include <chrono>
#include <cstdio>
#include <stdexcept>
#include <thread>

namespace ninfer::targets::qwen3_6::detail {
// NINFER_PROFILE_DECODE + MTP 轮分段(eager 模式专用): hipEvent 对在轮 lambda 内
// record, 事件随轮执行触发; 后台线程每 5s 读 elapsed 分账。
// 图捕获模式下事件不重触发 + host 剖面失明, 该模式无解(见 pc2 记档), 本结构只服务
// --no-cuda-graph 诊断。clock64 跨启动不可比(per-SIMD 计数), 已弃。
struct MtpRoundProfile {
    cudaEvent_t events[4]{};
    double      acc_ms[3] = {};
    int         rounds    = 0;
    bool        on        = false;

    static MtpRoundProfile& instance() {
        static MtpRoundProfile profile;
        return profile;
    }

    void ensure() {
        if (events[0] != nullptr) { return; }
        for (auto& event : events) { CUDA_CHECK(cudaEventCreate(&event)); }
        on = true;
        std::thread([this]() {
            while (on) {
                std::this_thread::sleep_for(std::chrono::seconds(5));
                settle();
            }
        }).detach();
    }

    void record(int slot, cudaStream_t stream) {
        CUDA_CHECK(cudaEventRecord(events[slot], stream));
    }

    void settle() {
        if (!on) { return; }
        float ms = 0.0f;
        bool sampled = false;
        for (int i = 0; i < 3; ++i) {
            if (cudaEventElapsedTime(&ms, events[i], events[i + 1]) == cudaSuccess &&
                ms >= 0.0f) {
                acc_ms[i] += ms;
                sampled = true;
            }
        }
        if (!sampled) { return; }
        ++rounds;
        const double verify = acc_ms[0] / rounds, align = acc_ms[1] / rounds,
                     draft = acc_ms[2] / rounds;
        std::fprintf(stderr,
                     "[mtp-round] %d 轮均值: verify %.2f ms | align %.2f ms | draft %.2f ms | "
                     "sum %.2f ms\n",
                     rounds, verify, align, draft, verify + align + draft);
        for (double& slot : acc_ms) { slot = 0.0; }
        rounds = 0;
    }
};
}  // namespace ninfer::targets::qwen3_6::detail

namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS::schedule {
void mtp_bridge_and_propose(PrefillContext& state, const Tensor& next_token,
                            const Tensor& previous_hidden, std::int32_t position,
                            std::span<const std::int32_t> rope_position, bool build_proposal,
                            const Tensor* next_embedding) {
    if (!state.mtp_kv.valid() || !state.execution.io.mtp) {
        throw std::logic_error("MTP bridge requires MTP storage");
    }
    if (rope_position.size() != 3) {
        throw std::invalid_argument("MTP bridge requires one three-axis rope position");
    }
    state.execution.work.reset();
    TextContext card(state.execution.device, state.execution.model, state.execution.work,
                     state.text_kv, state.execution.linear_attention, state.execution.io,
                     state.execution.prefill_hidden, state.execution.prefill_chunk,
                     state.text_kv_base, state.mtp_kv, &state.text_cache, state.mtp_cache);
    configure_text_card(card, state.execution, state.sampling, state.state_source_slot,
                        state.state_destination_slot, state.mtp_proposal_extent);

    Tensor position_view = state.execution.io.mtp->target_positions.slice(0, 0, 1);
    ops::set_i32_scalar(position_view, position, state.execution.device.stream);
    Tensor mtp_hidden         = state.execution.io.mtp->ar_hidden;
    Tensor logits             = state.execution.io.logits.slice(1, 0, 1);
    Tensor draft0             = state.execution.io.mtp->draft_tokens.slice(0, 0, 1);
    Tensor rope_position_view = state.execution.work.alloc(DType::I32, {1, 3});
    CUDA_CHECK(cudaMemcpyAsync(rope_position_view.data, rope_position.data(),
                               rope_position.size_bytes(), cudaMemcpyHostToDevice,
                               state.execution.device.stream));
    const auto bridge_visible = static_cast<std::uint32_t>(position + 1);
    const ops::CausalAttentionExecutionEnvelope bridge_envelope{bridge_visible, bridge_visible};
    card.mtp_forward_batch(next_token, previous_hidden, position_view, bridge_envelope, mtp_hidden,
                           build_proposal ? 0 : -1, build_proposal ? &logits : nullptr,
                           build_proposal ? &draft0 : nullptr, &rope_position_view, next_embedding);
    if (!build_proposal) { return; }

    if (state.mtp_proposal_extent == 0 ||
        state.mtp_proposal_extent >
            static_cast<std::uint32_t>(state.execution.io.mtp->draft_tokens.ne[0])) {
        throw std::logic_error("MTP bridge proposal extent is outside the configured window");
    }

    Tensor ar_position = state.execution.io.mtp->position.slice(0, 0, 1);
    ops::set_i32_scalar(ar_position, position + 1, state.execution.device.stream);
    for (int i = 1; i < static_cast<int>(state.mtp_proposal_extent); ++i) {
        Tensor previous_token = state.execution.io.mtp->draft_tokens.slice(0, i - 1, 1);
        Tensor next_draft     = state.execution.io.mtp->draft_tokens.slice(0, i, 1);
        Tensor next_hidden    = state.execution.prefill_hidden.slice(1, i, 1);
        const auto visible    = static_cast<std::uint32_t>(position + i + 1);
        const ops::CausalAttentionExecutionEnvelope envelope{visible, visible};
        card.mtp_forward_ar_step(previous_token, state.execution.io.mtp->ar_hidden, ar_position,
                                 envelope, next_hidden, logits, next_draft);
        CUDA_CHECK(cudaMemcpyAsync(state.execution.io.mtp->ar_hidden.data, next_hidden.data,
                                   state.execution.io.mtp->ar_hidden.bytes(),
                                   cudaMemcpyDeviceToDevice, state.execution.device.stream));
        ops::increment_i32_scalar(ar_position, state.execution.device.stream);
    }
}

auto mtp_decode_batch_body(MtpBatchContext& state, std::int32_t batch_size, std::uint32_t k,
                           MtpCausalAttentionEnvelopes envelopes) {
    return [&state, batch_size, k, envelopes] {
        if (batch_size <= 0 || batch_size > static_cast<std::int32_t>(kMaximumConcurrency) ||
            k == 0 || k > kMtpDecodeMaximumDrafts) {
            throw std::logic_error("MTP decode batch state is incomplete");
        }

        qwen3_6::MtpDecodeState& frame = state.frame;
        const std::int32_t width       = static_cast<std::int32_t>(k) + 1;
        CUDA_CHECK(cudaMemcpyAsync(frame.ingress.data, &state.host_ingress,
                                   sizeof(qwen3_6::MtpDecodeIngress), cudaMemcpyHostToDevice,
                                   state.execution.device.stream));

        TextContext card(state.execution.device, state.execution.model, state.execution.work, {},
                         state.execution.linear_attention, state.execution.io,
                         state.execution.prefill_hidden, state.execution.prefill_chunk, 0, {},
                         &state.text_cache, &state.mtp_cache);
        Tensor anchors            = frame.anchors.slice(0, 0, batch_size);
        Tensor frontiers          = frame.base_frontiers.slice(0, 0, batch_size);
        Tensor budgets            = frame.remaining_budgets.slice(0, 0, batch_size);
        Tensor current_extents    = frame.current_extents.slice(0, 0, batch_size);
        Tensor target_valid       = frame.target_valid_columns.slice(0, 0, batch_size);
        Tensor current_drafts     = frame.current_drafts.slice(1, 0, batch_size);
        Tensor target_rope        = frame.target_rope_positions.slice(1, 0, batch_size);
        Tensor text_rows          = frame.text_kv_table_rows.slice(0, 0, batch_size);
        Tensor mtp_rows           = frame.mtp_kv_table_rows.slice(0, 0, batch_size);
        Tensor state_sources      = frame.state_source_slots.slice(0, 0, batch_size);
        Tensor state_destinations = frame.state_destination_slots.slice(0, 0, batch_size);
        Tensor rope_deltas        = frame.rope_deltas.slice(0, 0, batch_size);
        Tensor verify_ids         = frame.verify_ids.slice(1, 0, batch_size);
        Tensor target_positions   = frame.target_positions.slice(1, 0, batch_size);
        Tensor target_tokens      = frame.target_argmax.slice(1, 0, batch_size);
        Tensor target_logits      = frame.target_logits.slice(2, 0, batch_size);
        Tensor target_hidden      = frame.target_hidden.slice(2, 0, batch_size);
        Tensor selected_hidden    = frame.target_continuation_hidden.slice(1, 0, batch_size);
        Tensor licensed_tokens    = frame.licensed_tokens.slice(1, 0, batch_size);
        Tensor licensed_counts    = frame.licensed_counts.slice(0, 0, batch_size);
        Tensor accepted           = frame.accepted_drafts.slice(0, 0, batch_size);
        Tensor next_extents       = frame.next_extents.slice(0, 0, batch_size);
        Tensor alignment_ids      = frame.alignment_ids.slice(1, 0, batch_size);
        Tensor alignment_hidden   = frame.alignment_hidden.slice(2, 0, batch_size);
        Tensor ar_hidden          = frame.ar_hidden.slice(1, 0, batch_size);
        Tensor next_hidden        = frame.next_hidden.slice(1, 0, batch_size);
        Tensor ar_positions       = frame.ar_positions.slice(0, 0, batch_size);
        Tensor ar_rope_positions  = frame.ar_rope_positions.slice(0, 0, batch_size);
        Tensor ar_valid_columns   = frame.ar_valid_columns.slice(0, 0, batch_size);
        Tensor next_drafts        = frame.next_drafts.slice(0, 0, batch_size);

        // 轮内分段事件(NINFER_PROFILE_DECODE 时 bake 进图, 宿主 settle 读)。
        MtpRoundProfile& round_profile = MtpRoundProfile::instance();
        const bool round_timing        = DecodeRegionProfile::instance().on();
        const cudaStream_t body_stream = state.execution.device.stream;
        if (round_timing) {
            round_profile.ensure();
            round_profile.record(0, body_stream);
        }

        ops::speculative_prepare_verify_inputs(anchors, current_drafts, frontiers, current_extents,
                                               verify_ids, target_positions,
                                               state.execution.device.stream);
        target_verify_accept(state.execution, state.continuation_hidden_store, card,
                             TargetVerifyFrameView{
                                 .ids                     = verify_ids,
                                 .cache_positions         = target_positions,
                                 .rope_positions          = target_rope,
                                 .valid_columns           = target_valid,
                                 .kv_table_rows           = text_rows,
                                 .state_source_slots      = state_sources,
                                 .state_destination_slots = state_destinations,
                                 .target_hidden           = target_hidden,
                                 .target_logits           = target_logits,
                                 .target_tokens           = target_tokens,
                                 .drafts                  = current_drafts,
                                 .current_extents         = current_extents,
                                 .frontiers               = frontiers,
                                 .anchors                 = anchors,
                                 .licensed_tokens         = licensed_tokens,
                                 .licensed_counts         = licensed_counts,
                                 .accepted_drafts         = accepted,
                                 .selected_hidden         = selected_hidden,
                                 .replay_records          = state.execution.replay_records,
                                 .sampling                = frame.sampling,
                             },
                             envelopes.target_verify);

        if (round_timing) { round_profile.record(1, body_stream); }
        ops::mtp_prepare_next_round(verify_ids, anchors, accepted, frontiers, budgets,
                                    licensed_counts, rope_deltas, alignment_ids, next_extents,
                                    ar_positions, ar_rope_positions, ar_valid_columns,
                                    static_cast<std::int32_t>(state.text_cache.max_context()),
                                    state.execution.device.stream);
        card.mtp_forward_decode_batch(alignment_ids, target_hidden, target_positions, target_rope,
                                      licensed_counts, mtp_rows, envelopes.batch, alignment_hidden);
        ops::speculative_select_accepted_hidden(alignment_hidden, accepted, ar_hidden,
                                                state.execution.device.stream);
        if (round_timing) { round_profile.record(2, body_stream); }

        Tensor proposal_logits = frame.proposal_logits.slice(1, 0, batch_size);
        Tensor draft0          = next_drafts.slice(1, 0, 1).view({batch_size});
        card.mtp_propose_batch(ar_hidden, proposal_logits, draft0);
        for (std::uint32_t step = 0; step + 1 < k; ++step) {
            Tensor previous =
                next_drafts.slice(1, static_cast<std::int32_t>(step), 1).view({batch_size});
            Tensor next =
                next_drafts.slice(1, static_cast<std::int32_t>(step + 1), 1).view({batch_size});
            Tensor position =
                ar_positions.slice(1, static_cast<std::int32_t>(step), 1).view({1, batch_size});
            Tensor rope = ar_rope_positions.slice(1, static_cast<std::int32_t>(step), 1)
                              .view({1, batch_size});
            Tensor valid =
                ar_valid_columns.slice(1, static_cast<std::int32_t>(step), 1).view({batch_size});
            Tensor previous_batch    = previous.view({1, batch_size});
            Tensor hidden_batch      = ar_hidden.view({TextConfig::hidden, 1, batch_size});
            Tensor next_hidden_batch = next_hidden.view({TextConfig::hidden, 1, batch_size});
            card.mtp_forward_decode_batch(previous_batch, hidden_batch, position, rope, valid,
                                          mtp_rows, envelopes.ar[step], next_hidden_batch);
            card.mtp_propose_batch(next_hidden, proposal_logits, next);
            CUDA_CHECK(cudaMemcpyAsync(ar_hidden.data, next_hidden.data, ar_hidden.bytes(),
                                       cudaMemcpyDeviceToDevice, state.execution.device.stream));
        }
        if (round_timing) { round_profile.record(3, body_stream); }

        CUDA_CHECK(cudaMemcpyAsync(&state.host_egress, frame.egress.data,
                                   sizeof(qwen3_6::MtpDecodeEgress), cudaMemcpyDeviceToHost,
                                   state.execution.device.stream));
        // B1 分账:MTP 轮从不走 T=1 路径的 step_done,剖面永远不出报告 —— 轮末补一拍,
        // 让 NINFER_PROFILE_DECODE 在 MTP 下也能出区域分账(每 32 轮一报,acc 与 T=1 路径共用)。
        DecodeRegionProfile::instance().step_done();
    };
}

void capture_mtp_decode_batch(MtpBatchContext& state, std::int32_t batch_size, std::uint32_t k,
                              MtpCausalAttentionEnvelopes envelopes,
                              DecodeGraphDefinition& definition) {
    auto body = mtp_decode_batch_body(state, batch_size, k, envelopes);
    capture_graph(state, definition, body);
}

void mtp_decode_batch(MtpBatchContext& state, std::int32_t batch_size, std::uint32_t k,
                      MtpCausalAttentionEnvelopes envelopes, DecodeGraphExecutable* executable) {
    auto body = mtp_decode_batch_body(state, batch_size, k, envelopes);
    run_prepared(state, executable, body);
}

} // namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS::schedule

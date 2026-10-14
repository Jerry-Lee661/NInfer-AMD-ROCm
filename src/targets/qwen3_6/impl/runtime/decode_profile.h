#pragma once

// Decode 预算剖析：`NINFER_PROFILE_DECODE=1` 打开，逐区同步计时，每 kReportEvery 步把各区
// 累计毫秒打到 stderr。`NINFER_PROFILE_DECODE=host` 则**不做 stream 同步**，量的是宿主侧
// 入队时间——两者相减即可把"GPU 执行"与"宿主开销"分开。
//
// 为什么要分：实测 attention 那段的开销与上下文长度几乎无关（17 token 10.28ms/步 vs
// 612 token 12.95ms/步，36 倍上下文只涨 26%），而 decode 在宿主 CPU 被 12 个 clang 抢占时
// 从 25.4 掉到 16-19 t/s——两条都指向"小 T 下宿主入队/校验开销占大头"，同步计时看不出这个。
//
// 同步模式会让绝对耗时变大（丢掉了跨区重叠），看的是**相对占比**。关闭时只读一个 static bool。

#include "core/device.h"

#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace ninfer::targets::qwen3_6::detail {

class DecodeRegionProfile {
public:
    // 粗分区：整步的四个大段。
    enum Slot {
        kEmbedding = 0,
        kLayers,
        kFinalNorm,
        kLmHead,
        // 层内细分：run_layers 里两个 nvtx 区（mixer / mlp_tail）按层类型各记一份。
        // 目的：36.9ms 的"64 层"到底落在 attention、GDN 还是 MLP 上——这个分辨不出来，
        // 下一刀就只能照 perf-survey 的旧账猜。
        kAttnMix,
        kGdnMix,
        kMlpFull,
        kMlpGdn,
        // mixer 内的逐 op 细分（层内分解把杠杆指到了这两个 mixer 上）：
        // attention = 输入 norm + QKV/gate 融合投影 → q/k norm + rope + attention 核 + sigmoid_mul
        //             → o 投影；GDN = norm+control 投影 → 其余（GDN 投影 + recurrence + 输出投影）。
        kAttnProj,
        kAttnCore,
        kAttnOutProj,
        kGdnControl,
        kGdnCtlStart,     // P-C④: gdn_control 劈两段——本 mark 前为层间空隙(host/dispatch),
                          // 本 mark→kGdnControl 为 norm+control 投影本体。
        kGdnRecur,
        // 再下一层（attn_core 12.6ms / gdn_rest 8.65ms 是实测最大的两项，要看到底是哪个核）。
        // attn_core 与 gdn_rest 这两个 slot 本身不再打点，改为下面子项求和（内层 mark 会重置
        // last_，直接打点只会量到尾巴）。
        kAttnNorm,        // q_norm + k_norm 两次 rmsnorm
        kAttnRope,
        kAttnKernel,      // causal_softmax_attention
        kAttnSigmoid,     // sigmoid_mul(gate, a)
        kGdnProj,         // GDN 输入投影（+conv1d，Verify 趟是融合核）
        kGdnRecurKernel,  // gated_delta_net（batch_update / replay / prefill 三选一）
        kGdnOutProj,      // gated_rmsnorm + 输出投影
        kSlotCount
    };

    static constexpr int kReportEvery = 32;

    static DecodeRegionProfile& instance() {
        static DecodeRegionProfile profiler;
        return profiler;
    }

    bool on() const { return on_; }

    // 在一个区的最后一个 kernel 之后调用：同步模式下先同步，把这段记到 slot 上；
    // host 模式只读时钟，量的是宿主入队耗时。
    void mark(Slot slot, cudaStream_t stream) {
        if (!on_) { return; }
        if (!host_only_) {
            // 图捕获期任何同步都非法且会作废整段图(2026-10-03: 曾直接炸捕获,
            // 吞错误码也不行——错误已发生)。捕获期整段跳过, 运行期 mark 照常计时。
            // 图捕获期任何同步都非法且会作废整段图(2026-10-03: 曾直接炸捕获,
            // 吞错误码也不行——错误已发生)。捕获期整段跳过, 运行期 mark 照常计时。
#if defined(NINFER_HIP)
            hipStreamCaptureStatus capture_status = hipStreamCaptureStatusNone;
            CUDA_CHECK(hipStreamIsCapturing(stream, &capture_status));
            if (capture_status == hipStreamCaptureStatusActive) { return; }
#else
            cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
            CUDA_CHECK(cudaStreamIsCapturing(stream, &capture_status));
            if (capture_status == cudaStreamCaptureStatusActive) { return; }
#endif
            CUDA_CHECK(cudaStreamSynchronize(stream));
        }
        const auto now = std::chrono::steady_clock::now();
        if (have_last_) {
            acc_[slot] += std::chrono::duration<double, std::milli>(now - last_).count();
        }
        last_      = now;
        have_last_ = true;
    }

    void step_done() {
        if (!on_ || ++steps_ < kReportEvery) { return; }
        steps_ = 0;
        // 层内/mixer 内的 mark 都会重置 last_，所以 kAttnMix/kGdnMix 这两个外层 slot 只剩
        // "尾巴"（最后一个内层 mark 到 mixer 结束）。层的总量必须用两个 total 相加，
        // 否则会少报一大截（踩过：曾打印 15.5ms/步，真实是 42ms/步）。
        const double attn_core = acc_[kAttnNorm] + acc_[kAttnRope] + acc_[kAttnKernel] +
                                 acc_[kAttnSigmoid];
        const double gdn_rest =
            acc_[kGdnProj] + acc_[kGdnRecurKernel] + acc_[kGdnOutProj];
        const double attn_mix_total = acc_[kAttnProj] + attn_core + acc_[kAttnOutProj];
        const double gdn_mix_total  = acc_[kGdnControl] + acc_[kGdnCtlStart] + gdn_rest;
        const double layers = attn_mix_total + gdn_mix_total + acc_[kMlpFull] + acc_[kMlpGdn];
        const char* mode            = host_only_ ? "host" : "dev";
        std::fprintf(stderr,
                     "profile[%s] decode (%d steps): embedding %.2f ms | layers %.2f ms | "
                     "final_norm %.2f ms | lm_head %.2f ms\n",
                     mode, kReportEvery, acc_[kEmbedding], layers, acc_[kFinalNorm],
                     acc_[kLmHead]);
        std::fprintf(stderr,
                     "profile[%s] layers (%d steps): attn_mix %.2f ms | gdn_mix %.2f ms | "
                     "mlp_full %.2f ms | mlp_gdn %.2f ms\n",
                     mode, kReportEvery, attn_mix_total, gdn_mix_total, acc_[kMlpFull],
                     acc_[kMlpGdn]);
        std::fprintf(stderr,
                     "profile[%s] mixer (%d steps): attn_proj %.2f ms | attn_core %.2f ms | "
                     "attn_out_proj %.2f ms | gdn_control %.2f ms | gdn_rest %.2f ms\n",
                     mode, kReportEvery, acc_[kAttnProj], attn_core, acc_[kAttnOutProj],
                     acc_[kGdnControl], gdn_rest);
        std::fprintf(stderr,
                     "profile[%s] op (%d steps): qk_norm %.2f ms | rope %.2f ms | attn_kernel "
                     "%.2f ms | sigmoid_mul %.2f ms | gdn_proj %.2f ms | gdn_recur %.2f ms | "
                     "gdn_out_proj %.2f ms\n",
                     mode, kReportEvery, acc_[kAttnNorm], acc_[kAttnRope], acc_[kAttnKernel],
                     acc_[kAttnSigmoid], acc_[kGdnProj], acc_[kGdnRecurKernel],
                     acc_[kGdnOutProj]);
        std::fflush(stderr);
        acc_.fill(0.0);
    }

private:
    DecodeRegionProfile() {
        const char* mode = std::getenv("NINFER_PROFILE_DECODE");
        on_              = mode != nullptr;
        host_only_       = mode != nullptr && std::strcmp(mode, "host") == 0;
    }

    bool on_;
    bool host_only_ = false;
    bool have_last_ = false;
    int steps_      = 0;
    std::chrono::steady_clock::time_point last_{};
    std::array<double, kSlotCount> acc_{};
};

} // namespace ninfer::targets::qwen3_6::detail

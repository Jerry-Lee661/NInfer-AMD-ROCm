#include "ops/linear/ternary/ternary_rowsplit_gemm.cuh"

#include "core/device.h"
#include "ops/common/math.h"
#include "ops/linear/ternary/ternary_launch.h"
#include "ops/linear/ternary/ternary_rotation.h"
#include "ops/linear/ternary/ternary_rowsplit_gemv.cuh"
#include "ops/linear/ternary/ternary_rowsplit_mma_s8.cuh"
#include "ops/linear/ternary/ternary_rowsplit_mma_small_t.cuh"
#include "ops/linear/ternary/ternary_rowsplit_mma_wide_t.cuh"
#include "ops/linear/ternary/ternary_pq2_iu8_wmma.cuh"
#include "ops/linear/ternary/ternary_t2v2.cuh"

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

namespace ninfer::ops::detail {
namespace {

// Decode (T == 1) takes the warp-per-row GEMV for PQ2_0. K is a whole number of 128-groups for
// every width in this model, so that kernel needs no column guard.
void launch_pq2_gemv(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    if ((w.k % 128) != 0 || x.ne[1] != 1) {
        throw std::invalid_argument("ternary gemv: expected one token and a whole-group K");
    }
    static const bool xlds_enabled = [] {
        const char* value = std::getenv("NINFER_GEMV_XLDS");
        return value != nullptr && std::strcmp(value, "1") == 0;
    }();
    if (xlds_enabled && w.k <= 5120) {
        // x-LDS 共享臂(P-D, 探针 +17% 码流): 32 warp/CTA + x 装 shared。k>5120 的
        // 层(LM head 17408 方向除外——它走 w8 家族)不做, shared 预算与装载成本按
        // 探针形状先验。
        const std::int32_t groups_per_row = w.k / 128;
        const unsigned grid = static_cast<unsigned>(div_up(w.n, kGemvXLdsWarpsPerBlock));
        ternary_pq2_gemv_xlds_kernel<<<grid, kGemvXLdsWarpsPerBlock * 32, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(w.qdata),
            static_cast<const std::uint8_t*>(w.scales), static_cast<__nv_bfloat16*>(out.data), w.n,
            groups_per_row, w.k);
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    const std::int32_t groups_per_row = w.k / 128;
    const unsigned grid               = static_cast<unsigned>(div_up(w.n, kGemvWarpsPerBlock));
    ternary_pq2_gemv_kernel<<<grid, kGemvWarpsPerBlock * 32, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(w.qdata),
        static_cast<const std::uint8_t*>(w.scales), static_cast<__nv_bfloat16*>(out.data), w.n,
        groups_per_row);
    CUDA_CHECK(cudaGetLastError());
}

// Small-token-tile GEMV: weights are read once for up to 4 tokens, which is what makes the
// speculative verify pass (T = draft + 1) cheap. Falls back to the reference tiled kernel beyond
// that, and for PTQ1_0 / padded-K weights.
void launch_pq2_gemv_tile(const Tensor& x, const Weight& w, Tensor& out,
                          std::int32_t out_row_stride, std::int32_t tokens,
                          cudaStream_t stream) {
    if ((w.k % 128) != 0) {
        throw std::invalid_argument("ternary gemv: K must be a whole number of 128-groups");
    }
    const std::int32_t groups_per_row = w.k / 128;
    const unsigned grid               = static_cast<unsigned>(div_up(w.n, kGemvWarpsPerBlock));
    const dim3 block(kGemvWarpsPerBlock * 32, 1u, 1u);
    if (tokens <= 1) {
        ternary_pq2_gemv_kernel<<<grid, block, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(w.qdata),
            static_cast<const std::uint8_t*>(w.scales), static_cast<__nv_bfloat16*>(out.data), w.n,
            groups_per_row);
    } else {
        ternary_pq2_gemv_tile_kernel<4><<<grid, block, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(w.qdata),
            static_cast<const std::uint8_t*>(w.scales), static_cast<__nv_bfloat16*>(out.data), w.n,
            groups_per_row, tokens, out_row_stride);
    }
    CUDA_CHECK(cudaGetLastError());
}

// PTQ1_0 twin of the GEMV above: same grid/block policy and the same token-tile size, but the weight
// fetch needs the high plane (w.qhigh) and decodes base-3 trits per column instead of 2-bit slots.
void launch_ptq1_gemv_tile(const Tensor& x, const Weight& w, Tensor& out,
                           std::int32_t out_row_stride, std::int32_t tokens,
                           cudaStream_t stream) {
    if ((w.k % 128) != 0) {
        throw std::invalid_argument("ternary gemv: K must be a whole number of 128-groups");
    }
    if (w.qhigh == nullptr) {
        throw std::invalid_argument("ternary gemv: PTQ1_0 needs its high plane");
    }
    const std::int32_t groups_per_row = w.k / 128;
    const unsigned grid               = static_cast<unsigned>(div_up(w.n, kGemvWarpsPerBlock));
    const dim3 block(kGemvWarpsPerBlock * 32, 1u, 1u);
    if (tokens <= 1) {
        ternary_ptq1_gemv_kernel<<<grid, block, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(w.qdata),
            static_cast<const std::uint8_t*>(w.qhigh), static_cast<const std::uint8_t*>(w.scales),
            static_cast<__nv_bfloat16*>(out.data), w.n, groups_per_row);
    } else {
        ternary_ptq1_gemv_tile_kernel<4><<<grid, block, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(w.qdata),
            static_cast<const std::uint8_t*>(w.qhigh), static_cast<const std::uint8_t*>(w.scales),
            static_cast<__nv_bfloat16*>(out.data), w.n, groups_per_row, tokens, out_row_stride);
    }
    CUDA_CHECK(cudaGetLastError());
}

// One call site for both ternary formats. The kernels key on w.qhigh (PQ2_0 has no high plane), so
// callers stay format-agnostic beyond this dispatch -- which is the whole point of the change.
void launch_gemv_tile(const Tensor& x, const Weight& w, Tensor& out,
                      std::int32_t out_row_stride, std::int32_t tokens, cudaStream_t stream) {
    if (w.qtype == QType::PTQ1_0_G128) {
        launch_ptq1_gemv_tile(x, w, out, out_row_stride, tokens, stream);
        return;
    }
    launch_pq2_gemv_tile(x, w, out, out_row_stride, tokens, stream);
}

template <class Storage, class Atom, int kTileT>
void launch_gemm(const Tensor& x, const Weight& w, Tensor& out, std::int32_t out_row_stride,
                 cudaStream_t stream) {
    const std::int32_t rows = w.n;
    const std::int32_t k    = w.k;
    const std::int32_t t    = x.ne[1];
    if (k % Storage::kGroupK != 0) {
        throw std::invalid_argument("ternary linear: K must be a multiple of the group size");
    }
    if (out_row_stride < rows) {
        throw std::invalid_argument("ternary linear: output row stride is smaller than the tile");
    }
    const std::int32_t groups_per_row = k / Storage::kGroupK;

    const dim3 grid(static_cast<unsigned>(rows), static_cast<unsigned>(div_up(t, kTileT)), 1u);
    dim3 block(Storage::kGroupK, 1u, 1u);

    ternary_rowsplit_gemm_kernel<Storage, Atom, kTileT><<<grid, block, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(w.qdata),
        static_cast<const std::uint8_t*>(w.qhigh), static_cast<const std::uint8_t*>(w.scales),
        static_cast<__nv_bfloat16*>(out.data), rows, k, t, groups_per_row, out_row_stride);
    CUDA_CHECK(cudaGetLastError());
}

template <int kTileT>
void launch_by_qtype(const Tensor& x, const Weight& w, Tensor& out, std::int32_t out_row_stride,
                     cudaStream_t stream) {
    switch (w.qtype) {
    case QType::PTQ1_0_G128:
        launch_gemm<PTQ1RowSplitStorage, PTQ1SimtDecodeAtom, kTileT>(x, w, out, out_row_stride,
                                                                    stream);
        return;
    case QType::PQ2_0_G128:
        launch_gemm<PQ2RowSplitStorage, PQ2SimtDecodeAtom, kTileT>(x, w, out, out_row_stride,
                                                                  stream);
        return;
    default:
        break;
    }
    throw std::invalid_argument("ternary linear: unsupported weight qtype");
}

} // namespace

// NINFER_TERNARY_PTQ1_FAST=0 puts PTQ1_0 back on the reference path -- the A/B arm, same pattern as
// the mma/s8 switches. Read once: the engine treats env as process-level (ternary_s8_scratch.h).
bool ternary_ptq1_fast_enabled() {
    static const bool enabled = [] {
        const char* env = std::getenv("NINFER_TERNARY_PTQ1_FAST");
        return env == nullptr || std::strcmp(env, "0") != 0;
    }();
    return enabled;
}

// A PQ2_0 **or PTQ1_0** weight can take the GEMV family whenever the layout did not pad K past the
// real width -- true for every width in this model (5120/6144/10240/17408 are all whole 128-groups),
// and checked here rather than assumed, because the GEMV reads whole groups without a column guard.
// The two formats are told apart by the high plane: row_split_geometry gives PQ2_0 zero high bytes
// (qhigh == nullptr) and PTQ1_0 two (qhigh != nullptr), which is also what the kernels key on.
bool gemv_admits(const Tensor& x, const Weight& w, std::int32_t max_tokens) {
    const bool pq2  = w.qtype == QType::PQ2_0_G128 && w.qhigh == nullptr;
    const bool ptq1 = w.qtype == QType::PTQ1_0_G128 && w.qhigh != nullptr &&
                      ternary_ptq1_fast_enabled();
    return (pq2 || ptq1) && w.padded_shape[1] == w.k && (w.k % 128) == 0 && x.ne[1] >= 1 &&
           x.ne[1] <= max_tokens;
}

void launch_pq2_mma_wide_t(const Tensor& x, const Weight& w, Tensor& out,
                           std::int32_t out_row_stride, std::int32_t tokens,
                           cudaStream_t stream) {
    // Static shared memory again (42.25 KB at BN=64), so no dynamic size and no opt-in.
    const unsigned grid = static_cast<unsigned>(div_up(w.n, kTernaryWideRowsPerCta));
    ternary_wide_t_kernel<false><<<grid, kTernaryWideThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(w.qdata),
        nullptr, static_cast<const std::uint8_t*>(w.scales),
        static_cast<__nv_bfloat16*>(out.data), w.n, w.k, tokens, out_row_stride);
    CUDA_CHECK(cudaGetLastError());
}

// PTQ1_0 twin: same kernel, same grid, same shared budget -- the only difference is that its staging
// step repacks the 24+2-byte rows into PQ2 code bytes (see the kernel's kPtq1 note).
void launch_ptq1_mma_wide_t(const Tensor& x, const Weight& w, Tensor& out,
                            std::int32_t out_row_stride, std::int32_t tokens,
                            cudaStream_t stream) {
    if (w.qhigh == nullptr) {
        throw std::invalid_argument("ternary wide-t mma: PTQ1_0 needs its high plane");
    }
    const unsigned grid = static_cast<unsigned>(div_up(w.n, kTernaryWideRowsPerCta));
    ternary_wide_t_kernel<true><<<grid, kTernaryWideThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(w.qdata),
        static_cast<const std::uint8_t*>(w.qhigh), static_cast<const std::uint8_t*>(w.scales),
        static_cast<__nv_bfloat16*>(out.data), w.n, w.k, tokens, out_row_stride);
    CUDA_CHECK(cudaGetLastError());
}

// One call site for both ternary formats, exactly like the GEMV dispatcher.
void launch_wide_t(const Tensor& x, const Weight& w, Tensor& out,
                   std::int32_t out_row_stride, std::int32_t tokens, cudaStream_t stream) {
    if (w.qtype == QType::PTQ1_0_G128) {
        launch_ptq1_mma_wide_t(x, w, out, out_row_stride, tokens, stream);
        return;
    }
    launch_pq2_mma_wide_t(x, w, out, out_row_stride, tokens, stream);
}

// Admission for every rung that READS PTQ1_0 and repacks it into the PQ2_0 code layout in shared
// (wide_t, small_t, s8). All three stage the raw 24-byte base rows with 8-byte cp.async copies and
// move a whole chunk's 2 high bytes per row as one more 8-byte copy, so the high plane's BASE has to
// be 8-byte aligned -- the row pitch is aligned by construction, because K is a whole number of
// 512-wide chunks (groups_per_row a multiple of 4). The check is explicit rather than assumed: an
// artifact that violates it falls back to a correct-but-slower rung instead of mis-decoding.
bool ptq1_repack_admits(const Weight& w) {
    return w.qtype == QType::PTQ1_0_G128 && w.qhigh != nullptr && ternary_ptq1_fast_enabled() &&
           (reinterpret_cast<std::uintptr_t>(w.qhigh) & 7u) == 0u;
}

// The wide-token-tile kernel only needs K to be a whole number of 128-wide chunks (one PQ2_0
// group), which is strictly weaker than the small-t path's 512. PTQ1_0 is admitted too: its staging
// step repacks the 24+2-byte rows into the PQ2 code layout in shared (verified bit-exact against
// real weights), so everything downstream of the staging is the same code.
bool mma_wide_admits(const Weight& w, std::int32_t out_row_stride) {
    const bool pq2  = w.qtype == QType::PQ2_0_G128 && w.qhigh == nullptr;
    const bool ptq1 = ptq1_repack_admits(w);
    return (pq2 || ptq1) && w.padded_shape[1] == w.k && (w.k % kTernaryWideChunkK) == 0 &&
           out_row_stride >= w.n;
}


// Decode (T == 1): the warp-per-row GEMV.
//
// MEASURED AND REJECTED ALTERNATIVE -- routing T = 1 through the tensor-core kernel. Recorded here
// so it is not re-tried: the synthetic harness (E:/infer-build/tscale_bench.cu) ranks mma at
// 14.56 ms/forward against 19.44 ms for this GEMV at T=1 over the artifact's real shape mix (+34%),
// which is the origin of the "+17% at T=1" claim on the small-T path below. On the actual engine
// the sign flips. A 400-token greedy decode A/B with the arms alternating three times measures
// 58.9 t/s with the tensor core against 60.8 t/s with this GEMV (58.3-59.5 against 60.2-61.2, no
// overlap), and a node-level nsys profile of both arms shows 81402 ternary GEMM calls in each --
// so the branch was genuinely taken, the mma kernel just costs ~7% more per call at T=1
// (36.6 us against 34.1 us).
//
// WHY THE HARNESS DISAGREES, MEASURED (E:/infer-build/t1_flush_probe.cu)
// Not L2 residency, and not the flush failing -- a 256 MB memset does evict this part's 64 MB L2
// (an 8 MiB read goes 7.4 us L2-hot to 13.3 us after a memset, against an 11.4 us DRAM floor).
// The harness's T=1 numbers are dominated by its own subtraction: it reports
//   (mean[256 MB memset + kernel]) - flush_constant
// where the constant is ~419 us and its own repeatability across a run spans 415-445 us
// (7.0%). That 29 us band is 1.2-3.2x the kernels it is used to isolate, and the same binary run
// three times reports this shape's T=1 mma as 901.5 / 292.7 / 357.8 GiB/s. A negative control sits
// inside one run: gemv_tile<kT=4> and gemv_tile<kT=8> are the SAME kernel at T=1 (both fall
// through to ternary_pq2_gemv_kernel) and were reported 344.0 and 487.0 GiB/s. The 901 GiB/s
// reading is also below that shape's 11.35 us DRAM floor, so it was never a bandwidth at all.
// Measured without any flush or subtraction (rotating a 128 MiB working set so every launch is
// DRAM-cold), the mma IS the faster kernel in isolation (~440-560 GiB/s against ~330-460 for this
// GEMV), so what the engine loses is not the kernel's inner loop but its interaction with the
// other ~600 launches of a token -- 38144 B of static shared memory puts it at 2 CTAs/SM where
// this GEMV has no shared memory and no barrier at all.
//
// WHAT THE TOKEN ACTUALLY LOOKS LIKE at T=1 (nsys, 200-token greedy run, 403 ternary GEMM calls
// per step): 13.75 ms of GEMV inside a 16.2 ms token = 85%, i.e. 6.70 GiB / 13.75 ms = 487 GiB/s
// = 71% of this card's 685.5 GiB/s spec peak, and 82% of the 595 GiB/s it actually delivers on a
// plain streaming read. The other ~2.4 ms is ~258 rotation launches (~1.1 ms) plus norms, GDN and
// attention. Repro pack: E:/infer-build/run-t1-ab-long.cmd, run-nsys-t1.cmd, t1_flush_probe.cu.
//
// COMMUNITY CROSS-CHECK (2026-09-20) -- this is not a local anomaly, so do not re-litigate it:
//   * The one other ninfer fork that routed a T=1 projection to mma (UDPSendToFailed/ninfer-4090,
//     commit 39a6f20, "route T=1 draft head projections to double-buffered Ada MMA") measured
//     +3.0% on real-world decode but 0.0% on PURE T=1 auto-regression, and its win is credited to
//     grid/wave geometry (32768 blocks = 256 waves -> 8192 blocks), not to the tensor core.
//   * Upstream PR #292 moves T=1 onto a K-split mma rung only on two N=5120 shapes, "where it beats
//     the dedicated GEMV by 8-11%. N=6144 and N=7168 keep that GEMV, which is still 7-12% faster
//     for them", and reports that ordinary (non-speculative) generation "moves about a percent,
//     which is close to this machine's run-to-run noise".
//   * llama.cpp keeps SIMT matvec up to batch 8 on Ada for the same reason this kernel loses here:
//     it says the deciding factor is that dequantization cost cannot be amortised over one column
//     ("k-quants cost more to decode and mvq redoes that per column, so MMQ wins sooner"), and
//     Marlin's own framing is that batch < 64 is memory-bound, so a tensor core at batch 1 is at
//     best "not a loss". Expect 0-3% here, per shape -- never the +17% a micro-benchmark suggests.
// Defined below: the small-t mma rung and its admission check. The T=1 policy prefers them on
// sm_86 -- see the routing note in launch_ternary_gemm_t1.
bool mma_admits(const Weight& w, std::int32_t out_row_stride);
void launch_small_t(const Tensor& x, const Weight& w, Tensor& out,
                    std::int32_t out_row_stride, std::int32_t tokens, cudaStream_t stream);

void launch_ternary_gemm_t1(const Tensor& x, const Weight& w, Tensor& out,
                            std::int32_t out_row_stride, cudaStream_t stream,
                            TernaryS8Scratch scratch) {
    // Decode rung probe: same env switch and budget knob as the T>=2 probe in launch_ternary_gemm_t8.
    // Under CUDA Graphs this host dispatch runs exactly once per linear at capture time, so the
    // first lines ARE the rungs baked into the decode graph -- no per-replay accounting exists.
    static int probe_t1_budget = []() {
        const char* env = std::getenv("NINFER_TERNARY_S8_DEBUG_BUDGET");
        return env != nullptr ? std::atoi(env) : 24;
    }();
    const bool probe_t1 =
        std::getenv("NINFER_TERNARY_S8_DEBUG") != nullptr && probe_t1_budget > 0;

    const char* rung   = "generic";
    bool use_small_t   = false;
    if (gemv_admits(x, w, 1)) {
        rung = "gemv_tile";
    }
#if defined(NINFER_SM86) || defined(NINFER_HIP)
    // sm_86 routing: the SIMT GEMV family underperforms on this target. T0-T5 (bench86/REPORT.md,
    // 2026-09-27) measured the PTQ1 GEMV at ~148 GB/s against 536 GB/s for the mma tile path on
    // the same weights, and the SASS census ties the gap to the PTQ1 lane plan issuing four
    // per-lane byte loads per group where the PQ2 twin issues one. The small-t rung reads the
    // weights once per token tile, so at T=1 it is one weight pass exactly like the GEMV and wins
    // on issue efficiency. The upstream/4090 measurements that keep the GEMV at T=1 were taken on
    // healthy-GEMV builds (PQ2 artifact, sm_89) -- a premise this build does not satisfy -- so the
    // preference is sm_86-only, and NINFER_TERNARY_T1_MMA=0 restores the historical arm without a
    // rebuild.
    // 2026-10-08 (gfx1100): the gate widened to NINFER_HIP for the ROCm port's own A/B -- our
    // GEMV is the dp4a variant (230..280 GB/s measured on gfx1100) and the upstream ninfer-all
    // line now runs T=1 through its tensor-core small-T kernel by default (430..660 GB/s there);
    // NINFER_TERNARY_T1_MMA=1 opts in here, unset keeps the historical HIP arm (dp4a GEMV).
    static const bool t1_mma_enabled = []() {
        const char* env = std::getenv("NINFER_TERNARY_T1_MMA");
#if defined(NINFER_HIP) && !defined(NINFER_SM86)
        return env != nullptr && std::strcmp(env, "0") != 0;  // HIP: 默认关,=1 开
#else
        return env == nullptr || std::strcmp(env, "0") != 0;  // sm_86: 默认开
#endif
    }();
    if (t1_mma_enabled && mma_admits(w, out_row_stride)) {
        use_small_t = true;
        rung        = "small_t_mma";
    }
#endif
    if (probe_t1) {
        --probe_t1_budget;
        std::fprintf(stderr,
                     "[ternary-t1] rung=%s qtype=%d T=%d k=%d n=%d stride=%d gemv_admits=%d "
                     "mma_admits=%d ptq1_fast=%d\n",
                     rung, static_cast<int>(w.qtype), static_cast<int>(x.ne[1]),
                     static_cast<int>(w.k), static_cast<int>(w.n),
                     static_cast<int>(out_row_stride), gemv_admits(x, w, 1) ? 1 : 0,
                     mma_admits(w, out_row_stride) ? 1 : 0,
                     ternary_ptq1_fast_enabled() ? 1 : 0);
    }
    if (use_small_t) {
        launch_small_t(x, w, out, out_row_stride, x.ne[1], stream);
        return;
    }
    if (gemv_admits(x, w, 1)) {
#if defined(NINFER_HIP)
        // dp4a rung(P-A 定案): 激活 q8_1 预量化 + v_perm 解码 + dp4a, 探针码流 435→670 GB/s。
        // scratch 为空(门关/容量端未分配)时保持 bf16 arm, 同二进制 A/B。
        if (ternary_pq2_dp4a_enabled() && scratch.dp4a_xq != nullptr &&
            w.qtype == QType::PQ2_0_G128) {
            launch_pq2_dp4a_tile(x, w, out, out_row_stride, stream, scratch);
            return;
        }
#else
        (void)scratch;
#endif
        launch_gemv_tile(x, w, out, out_row_stride, x.ne[1], stream);
        return;
    }
    launch_by_qtype<1>(x, w, out, out_row_stride, stream);
}

// 量化上提窗口(2026-10-08, NINFER_DP4A_QHOIST=1;详见 ternary_launch.h 的契约注释)。
// 宿主派发与图捕获都是单线程,静态窗口安全;窗口必须与分片组 begin/end 成对。
namespace {
const void*    g_dp4a_hoist_x     = nullptr;
std::int32_t   g_dp4a_hoist_tokens = 0;
bool           g_dp4a_hoist_armed  = false;
} // namespace

void ternary_pq2_dp4a_hoist_begin(const Tensor& x, std::int32_t k, TernaryS8Scratch scratch,
                                  cudaStream_t stream) {
#if defined(NINFER_HIP)
    if (scratch.rot_deferred()) {
        // M2: 激活未旋转, 由融合核就地旋转+量化(省掉一个 rotate 节点)。
        launch_ternary_rotation_quantized(x, k, scratch, stream);
    } else {
        const unsigned qgrid = static_cast<unsigned>(div_up(k / 32, 256));
        const dim3     qgrid3(qgrid, static_cast<unsigned>(x.ne[1]), 1u);
        ternary_pq2_dp4a_quant_kernel<<<qgrid3, 256, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data), scratch.dp4a_xq, scratch.dp4a_ds8, k);
        CUDA_CHECK(cudaGetLastError());
    }
    g_dp4a_hoist_x      = x.data;
    g_dp4a_hoist_tokens = x.ne[1];
    g_dp4a_hoist_armed  = true;
#else
    (void)x; (void)k; (void)scratch; (void)stream;
#endif
}

void ternary_pq2_dp4a_hoist_end() {
    g_dp4a_hoist_armed = false;
    g_dp4a_hoist_x     = nullptr;
    g_dp4a_hoist_tokens = 0;
}

// S1/S1b(2026-10-17, verify 摊销手术): iu8-WMMA verify tile 域开关(dp4a tile 与 seg2 两处
// 共用同一门)。默认开, `=0` 回退 dp4a(opt-out 语义与四链一致)。
bool pq2_iu8_enabled() {
    static const bool enabled = []() {
        const char* v = std::getenv("NINFER_PQ2_IU8");
        return v == nullptr || std::strcmp(v, "0") != 0;
    }();
    return enabled;
}

// dp4a rung 的统一入口(T=1..4, PQ2_0, NINFER_PQ2_DP4A=1): 量化 pass(按 token 行)
// + gemv/tile dp4a 内核。调用方保证 scratch.dp4a_xq 非空(门开时由
// allocate_ternary_s8_scratch / ternary_dispatch 分配)与 PQ2 整组 K; 这里只做形状分发。
void launch_pq2_dp4a_tile(const Tensor& x, const Weight& w, Tensor& out,
                          std::int32_t out_row_stride, cudaStream_t stream,
                          TernaryS8Scratch scratch, const Tensor* residual) {
#if defined(NINFER_HIP)
    // S1(2026-10-17, verify 摊销手术): verify tile(T=2..4)换 iu8-WMMA 核
    // (ternary_pq2_iu8_wmma.cuh)。同一 xq/ds8/权重字节流, 只换 GEMM 核: 16 行解码摊 16 列,
    // 指令流不再随列线性放大。探针实测对 dp4a tile 全形状 ≥2.0×(T=3 2.00-2.16×)。
    // 全门绿后**默认开**(2026-10-17): d2 27.3→39.0(+42%, 验收 1.81 同, md5 3857bdb9 逐字节同),
    // d1 +25%, serve headline 24.2→34.7(+43%)/服务端 25.6→37.5, PPL 1.75673 逐位同。
    const bool iu8_enabled = pq2_iu8_enabled();
    const __nv_bfloat16* residual_ptr =
        residual != nullptr ? static_cast<const __nv_bfloat16*>(residual->data) : nullptr;
    const std::int32_t tokens           = x.ne[1];
    const std::int32_t groups_per_row   = w.k / 128;
    const unsigned     qgrid            = static_cast<unsigned>(div_up(w.k / 32, 256));
    const dim3         qgrid3(qgrid, static_cast<unsigned>(tokens), 1u);
    // 量化上提(NINFER_DP4A_QHOIST=1): 武装窗口内同 (x.data, tokens) 的分片跳过量化
    // (窗口由融合调用点 begin/end 圈定,宿主派发单线程)。
    const bool hoisted = ternary_pq2_dp4a_qhoist_enabled() && g_dp4a_hoist_armed &&
                         g_dp4a_hoist_x == x.data && g_dp4a_hoist_tokens == tokens;
    if (!hoisted) {
        if (scratch.rot_deferred()) {
            // M2: 激活未旋转(调用点已声明), 融合核就地旋转+量化。
            launch_ternary_rotation_quantized(x, w.k, scratch, stream);
        } else {
            ternary_pq2_dp4a_quant_kernel<<<qgrid3, 256, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(x.data), scratch.dp4a_xq, scratch.dp4a_ds8, w.k);
        }
    }
    if (iu8_enabled && tokens >= 2 && (w.n % 16) == 0) {
        launch_pq2_iu8_wmma_tile(scratch.dp4a_xq, scratch.dp4a_ds8,
                                 static_cast<const std::uint8_t*>(w.qdata),
                                 static_cast<const std::uint8_t*>(w.scales),
                                 static_cast<__nv_bfloat16*>(out.data), w.n, groups_per_row,
                                 tokens, out_row_stride, stream, residual_ptr,
                                 scratch.dp4a_ksplit);
        return;
    }
    const unsigned grid = static_cast<unsigned>(div_up(w.n, kGemvWarpsPerBlock));
    const dim3 block(kGemvWarpsPerBlock * 32, 1u, 1u);
    if (tokens <= 1) {
        ternary_pq2_gemv_dp4a_kernel<<<grid, block, 0, stream>>>(
            scratch.dp4a_xq, scratch.dp4a_ds8, static_cast<const std::uint8_t*>(w.qdata),
            static_cast<const std::uint8_t*>(w.scales), static_cast<__nv_bfloat16*>(out.data),
            w.n, groups_per_row, residual_ptr);
    } else if (tokens <= 4) {
        // 4 列实例 = P0b 前的唯一实例, codegen 与旧核逐字同。
        ternary_pq2_gemv_tile_dp4a_kernel<4><<<grid, block, 0, stream>>>(
            scratch.dp4a_xq, scratch.dp4a_ds8, static_cast<const std::uint8_t*>(w.qdata),
            static_cast<const std::uint8_t*>(w.scales), static_cast<__nv_bfloat16*>(out.data),
            w.n, groups_per_row, tokens, out_row_stride, residual_ptr);
    } else {
        ternary_pq2_gemv_tile_dp4a_kernel<kPq2Dp4aMaxTokens><<<grid, block, 0, stream>>>(
            scratch.dp4a_xq, scratch.dp4a_ds8, static_cast<const std::uint8_t*>(w.qdata),
            static_cast<const std::uint8_t*>(w.scales), static_cast<__nv_bfloat16*>(out.data),
            w.n, groups_per_row, tokens, out_row_stride, residual_ptr);
    }
    CUDA_CHECK(cudaGetLastError());
#else
    (void)x; (void)w; (void)out; (void)out_row_stride; (void)stream; (void)scratch; (void)residual;
    throw std::logic_error("pq2 dp4a rung is HIP-only");
#endif
}

// 双段合并发射(M2c): 调用方保证 rows_a + rows_b == w.n。量化/上提逻辑与
// launch_pq2_dp4a_tile 完全同构(同 x 同 scratch; 武装窗口内跳过量化)。
void launch_pq2_dp4a_seg2(const Tensor& x, const Weight& w, Tensor& out_a, std::int32_t rows_a,
                          std::int32_t stride_a, Tensor& out_b, std::int32_t stride_b,
                          cudaStream_t stream, TernaryS8Scratch scratch) {
#if defined(NINFER_HIP)
    const std::int32_t tokens         = x.ne[1];
    const std::int32_t rows_b         = w.n - rows_a;
    const std::int32_t groups_per_row = w.k / 128;
    const unsigned     qgrid          = static_cast<unsigned>(div_up(w.k / 32, 256));
    const dim3         qgrid3(qgrid, static_cast<unsigned>(tokens), 1u);
    const bool hoisted = ternary_pq2_dp4a_qhoist_enabled() && g_dp4a_hoist_armed &&
                         g_dp4a_hoist_x == x.data && g_dp4a_hoist_tokens == tokens;
    if (!hoisted) {
        if (scratch.rot_deferred()) {
            launch_ternary_rotation_quantized(x, w.k, scratch, stream);
        } else {
            ternary_pq2_dp4a_quant_kernel<<<qgrid3, 256, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(x.data), scratch.dp4a_xq, scratch.dp4a_ds8, w.k);
        }
        CUDA_CHECK(cudaGetLastError());
    }
    // S1b(2026-10-17): seg2 的 iu8 分支 —— 同一量化 scratch, 两段各一发 iu8 tile
    // (段 B 基址 = codes/scales + rows_a 行偏移; 段长任意, 核内 clamp/guard 兜界)。
    // T=1 仍走合并 dp4a 专用形态(launch 合并的意义在 T=1)。
    if (pq2_iu8_enabled() && tokens >= 2) {
        const std::uint8_t* codes  = static_cast<const std::uint8_t*>(w.qdata);
        const std::uint8_t* scales = static_cast<const std::uint8_t*>(w.scales);
        launch_pq2_iu8_wmma_tile(scratch.dp4a_xq, scratch.dp4a_ds8, codes, scales,
                                 static_cast<__nv_bfloat16*>(out_a.data), rows_a, groups_per_row,
                                 tokens, stride_a, stream, nullptr, scratch.dp4a_ksplit);
        launch_pq2_iu8_wmma_tile(
            scratch.dp4a_xq, scratch.dp4a_ds8,
            codes + static_cast<std::int64_t>(rows_a) * groups_per_row * kGemvCodeBytesPerGroup,
            scales + static_cast<std::int64_t>(rows_a) * groups_per_row * kGemvScaleBytesPerGroup,
            static_cast<__nv_bfloat16*>(out_b.data), rows_b, groups_per_row, tokens, stride_b,
            stream, nullptr, scratch.dp4a_ksplit);
        return;
    }
    const unsigned grid = static_cast<unsigned>(div_up(w.n, kGemvWarpsPerBlock));
    const dim3 block(kGemvWarpsPerBlock * 32, 1u, 1u);
    if (tokens <= 1) {
        // T=1 必须走专用形态(tile 结构在 T=1 慢 2.7-2.9×, 见核旁注释与探针)。
        ternary_pq2_gemv_dp4a_seg2_t1_kernel<<<grid, block, 0, stream>>>(
            scratch.dp4a_xq, scratch.dp4a_ds8, static_cast<const std::uint8_t*>(w.qdata),
            static_cast<const std::uint8_t*>(w.scales), static_cast<__nv_bfloat16*>(out_a.data),
            rows_a, stride_a, static_cast<__nv_bfloat16*>(out_b.data), stride_b, w.n,
            groups_per_row);
    } else if (tokens <= 4) {
        // 4 列实例 = P0b 前的唯一实例, codegen 与旧核逐字同。
        ternary_pq2_gemv_tile_dp4a_seg2_kernel<4><<<grid, block, 0, stream>>>(
            scratch.dp4a_xq, scratch.dp4a_ds8, static_cast<const std::uint8_t*>(w.qdata),
            static_cast<const std::uint8_t*>(w.scales), static_cast<__nv_bfloat16*>(out_a.data),
            rows_a, stride_a, static_cast<__nv_bfloat16*>(out_b.data), stride_b, w.n,
            groups_per_row, tokens);
    } else {
        ternary_pq2_gemv_tile_dp4a_seg2_kernel<kPq2Dp4aMaxTokens><<<grid, block, 0, stream>>>(
            scratch.dp4a_xq, scratch.dp4a_ds8, static_cast<const std::uint8_t*>(w.qdata),
            static_cast<const std::uint8_t*>(w.scales), static_cast<__nv_bfloat16*>(out_a.data),
            rows_a, stride_a, static_cast<__nv_bfloat16*>(out_b.data), stride_b, w.n,
            groups_per_row, tokens);
    }
    CUDA_CHECK(cudaGetLastError());
#else
    (void)x; (void)w; (void)out_a; (void)rows_a; (void)stride_a; (void)out_b; (void)stride_b;
    (void)stream; (void)scratch;
    throw std::logic_error("pq2 dp4a seg2 rung is HIP-only");
#endif
}

// segn 表驱动发射(M2d): 2..4 段, T=1, 各段可有不同权重基址。量化/上提与单段同构。
void launch_pq2_dp4a_segn(const Tensor& x, const TernarySegDesc* segs, std::int32_t nsegs,
                          cudaStream_t stream, TernaryS8Scratch scratch) {
#if defined(NINFER_HIP)
    const std::int32_t tokens          = x.ne[1];
    const std::int32_t k               = x.ne[0];
    const std::int32_t groups_per_row  = k / 128;
    const unsigned     qgrid           = static_cast<unsigned>(div_up(k / 32, 256));
    const dim3         qgrid3(qgrid, static_cast<unsigned>(tokens), 1u);
    const bool hoisted = ternary_pq2_dp4a_qhoist_enabled() && g_dp4a_hoist_armed &&
                         g_dp4a_hoist_x == x.data && g_dp4a_hoist_tokens == tokens;
    if (!hoisted) {
        if (scratch.rot_deferred()) {
            launch_ternary_rotation_quantized(x, k, scratch, stream);
        } else {
            ternary_pq2_dp4a_quant_kernel<<<qgrid3, 256, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(x.data), scratch.dp4a_xq, scratch.dp4a_ds8, k);
        }
        CUDA_CHECK(cudaGetLastError());
    }
    TernarySegTable table{};
    std::int32_t total = 0;
    for (std::int32_t i = 0; i < nsegs; ++i) {
        table.codes[i]  = static_cast<const std::uint8_t*>(segs[i].w->qdata);
        table.scales[i] = static_cast<const std::uint8_t*>(segs[i].w->scales);
        table.out[i]    = static_cast<__nv_bfloat16*>(segs[i].out->data);
        table.row_begin[i] = total;
        total += segs[i].w->n;
    }
    for (std::int32_t i = nsegs; i < 4; ++i) { table.row_begin[i] = 0x7FFFFFFF; }
    table.nsegs = nsegs;
    const unsigned grid = static_cast<unsigned>(div_up(total, kGemvWarpsPerBlock));
    const dim3 block(kGemvWarpsPerBlock * 32, 1u, 1u);
    ternary_pq2_gemv_dp4a_segn_t1_kernel<<<grid, block, 0, stream>>>(
        scratch.dp4a_xq, scratch.dp4a_ds8, groups_per_row, total, table);
    CUDA_CHECK(cudaGetLastError());
#else
    (void)x; (void)segs; (void)nsegs; (void)stream; (void)scratch;
    throw std::logic_error("pq2 dp4a segn rung is HIP-only");
#endif
}

// Tensor-core path for the small-T regime (T = 2..8) -- the speculative verify pass.
//
// The SIMT tile GEMV reuses each weight across the tile but still pays one FMA per weight PER
// TOKEN, so its cost grows with T while its weight traffic does not: measured on a 322 MB PQ2_0
// payload it takes 0.706 ms at T=1 and 1.533 ms at T=3, i.e. 2.2x the time for byte-identical
// traffic (211 GiB/s against 483 GiB/s). The tensor core removes that per-token bill entirely --
// one mma.m16n8k16 carries 16 rows x 8 tokens x 16 k -- so the whole draft+verify tile rides one
// weight read: measured 0.611 / 0.633 ms at T = 3 / 8 against 1.493 / 1.533 ms for the tile GEMV.
// (The T=1 endpoint of that same measurement, 0.603 against 0.706 ms, did NOT transfer to the real
// engine -- see the note on launch_ternary_gemm_t1. The kernel below is for T >= 2 only.)
//
// BOTH formats are admitted. The PTQ1_0 instantiation cp.async's the raw 24+2-byte row into the
// same 144-byte weight rows and repacks it into the PQ2 code layout before the mma (the same
// repack and the same code_of/stage_codes arithmetic the wide-t rung uses, validated bit-exact
// against the real artifact), so everything downstream of the staging is the same code. Its only
// extra layout requirement is that the high plane be 8-byte aligned, and it is checked HERE rather
// than assumed: a failure falls back to the tile GEMV, which is correct and merely slower (see the
// note at the condition), never a silent mis-decode.
bool mma_admits(const Weight& w, std::int32_t out_row_stride) {
    const bool pq2 = w.qtype == QType::PQ2_0_G128 && w.qhigh == nullptr;
    const bool ptq1 = w.qtype == QType::PTQ1_0_G128 && w.qhigh != nullptr &&
                      ternary_ptq1_fast_enabled() &&
                      // The staging moves a whole chunk's 4 x 2 high bytes per row as ONE 8-byte
                      // cp.async. The row pitch is 8-byte aligned by construction (groups_per_row
                      // is a multiple of 4 whenever K is a whole number of 512-wide chunks), so
                      // only the plane base is not implied by the geometry.
                      (reinterpret_cast<std::uintptr_t>(w.qhigh) & 7u) == 0u;
    return (pq2 || ptq1) && w.padded_shape[1] == w.k &&
           (w.k % kTernaryMmaChunkK) == 0 && out_row_stride >= w.n;
}

// sm_86 decode retune (2026-09-27): ncu shows the 4-warp CTA geometry leaves the SMs starved at
// narrow N (grid 96 = 0.3 waves on 80 SMs, DRAM 33-54%). NINFER_SMALLT_WARPS=2 launches the
// 2-warp instantiation (CTA count doubles, shared halves); the default keeps the shipped 4-warp
// geometry. Outputs are bit-identical either way: every warp owns 16 rows and walks the whole K,
// so only the row-to-CTA packing changes.
// sm_86 decode retune (2026-09-27 rounds 1-2, now DEFAULT): 2 warps per CTA + group-ILP unroll.
// ncu evidence: 4-warp geometry ran 0.3 waves on 80 SMs with "wait" dependency stalls at 45-56%;
// the 2-warp + ILP geometry is bit-identical on greedy (chat/easy/hard, 596/436/553 tokens) and
// lifted the MTP d2 production wall clock 70 -> 81.2 t/s and MTP0 38.5 -> 44.3. The env vars
// still override for A/B: NINFER_SMALLT_WARPS=4|2|1, NINFER_SMALLT_ILP=0|1.
[[nodiscard]] inline int small_t_warps_per_cta() {
    static const int warps = [] {
        const char* value = std::getenv("NINFER_SMALLT_WARPS");
        if (value != nullptr && std::strcmp(value, "4") == 0) { return kTernaryMmaWarpsPerCta; }
        if (value != nullptr && std::strcmp(value, "1") == 0) { return 1; }
        return 2;
    }();
    return warps;
}

// Round 2 (2026-09-27): NINFER_SMALLT_ILP=0 restores the serial group loop. Default (unset or
// "1") unrolls the per-chunk group loop (kGroups == 2 on PTQ1_0) so the two base-3 decode chains
// interleave. Per-group arithmetic is unchanged (bit-identical out).
// Round 18: NINFER_SMALLT_WBUFS=2 drops the PTQ1_0 weight-buffer rotation from 3 to 2. With
// the native operand production (repack off) only two roles are live -- raw(c+1) fill and raw(c)
// mma read -- so the third buffer is vestigial shared. 2 bufs shrink the 2-warp CTA's shared from
// 16,384 B to 13,824 B: 6 -> 7 resident CTAs per SM (12 -> 14 warps).
// Round 29: NINFER_SMALLT_CHUNK128=1 halves the PTQ1_0 staging chunk (256 -> 128 weights),
// shrinking shared per 2-warp CTA and buying resident CTAs. The mma loop, staging and
// scale application are all Geom-parameterized, so the variant is a pure template argument.
[[nodiscard]] inline bool small_t_chunk128() {
    static const bool on = [] {
        const char* value = std::getenv("NINFER_SMALLT_CHUNK128");
        return value != nullptr && std::strcmp(value, "1") == 0;
    }();
    return on;
}

// Round 39 (sm_86): NINFER_SMALLT_BITPLANE=1 runs the PQ2 path's sign-plane decode instead of
// the LUT. Requires a PQ2S artifact (bench86/pq2s_repack.py: codes remapped so bit0 = negative
// plane, bit1 = positive). PTQ1 artifacts must NOT set it. Default off.
[[nodiscard]] inline bool small_t_bitplane() {
    static const bool on = [] {
        const char* value = std::getenv("NINFER_SMALLT_BITPLANE");
        return value != nullptr && std::strcmp(value, "1") == 0;
    }();
    return on;
}

// Round 55 (sm_86 kernel port, user goal A): NINFER_SMALLT_T2V2 runs the ported upstream
// t2_small_t_v2 design (ternary_t2v2.cuh) on the PQ2 path -- L1-direct activations, PRMT
// two-register decode, 8 warps x 16-row tiles with K split over warps, codes+scales staged in a
// ring. Requires rows % 16 == 0, k % 512 == 0 and tokens <= 8; otherwise the caller keeps the
// existing geometry. ROUND 56: DEFAULT ON -- five loads (zh easy/hard/chat, en, code) +40..56%,
// MTP d3/d5 (+50%/+40%, acceptance unchanged), easy output bitwise identical, hard/chat benign
// reordering, PPL 512/256 bit-identical (1.125169058908091 both). Set NINFER_SMALLT_T2V2=0 to
// fall back to the previous geometry.
[[nodiscard]] inline bool small_t_t2v2() {
    static const bool on = [] {
        const char* value = std::getenv("NINFER_SMALLT_T2V2");
        return value == nullptr || std::strcmp(value, "0") != 0;
    }();
    return on;
}

[[nodiscard]] inline int small_t_weight_bufs() {
    static const int bufs = [] {
        const char* value = std::getenv("NINFER_SMALLT_WBUFS");
        if (value != nullptr && std::strcmp(value, "3") == 0) { return 3; }
        return 2; // default: 2-buf rotation (round 18: output 100% match, wall-neutral→positive at 400W)
    }();
    return bufs;
}

[[nodiscard]] inline bool small_t_group_ilp() {
    static const bool ilp = [] {
        const char* value = std::getenv("NINFER_SMALLT_ILP");
        return value == nullptr || std::strcmp(value, "0") != 0;
    }();
    return ilp;
}

// Round 4 (2026-09-27): NINFER_SMALLT_DIRECT=1 recomputes each decode stage from the staged raw
// byte ((raw*3^s)&0xFF)*3 instead of advancing the in-place base-3 chain -- mod-256 arithmetic is
// identical, but the per-step multiplies stop depending on the previous step (ncu "wait" 41-53%).
[[nodiscard]] inline bool small_t_direct_decode() {
    static const bool direct = [] {
        const char* value = std::getenv("NINFER_SMALLT_DIRECT");
        return value != nullptr && std::strcmp(value, "1") == 0;
    }();
    return direct;
}

template <bool kPtq1_>
void launch_small_t_mma(const Tensor& x, const Weight& w, const std::uint8_t* high, Tensor& out,
                        std::int32_t out_row_stride, std::int32_t tokens, cudaStream_t stream) {
    const int warps = small_t_warps_per_cta();
    const bool ilp = small_t_group_ilp();
    const bool direct = kPtq1_ && small_t_direct_decode();
    const unsigned grid = static_cast<unsigned>(div_up(w.n, kTernaryMmaRowsPerWarp * warps));
    const auto* xp = static_cast<const __nv_bfloat16*>(x.data);
    const auto* qp = static_cast<const std::uint8_t*>(w.qdata);
    const auto* sp = static_cast<const std::uint8_t*>(w.scales);
    auto* op = static_cast<__nv_bfloat16*>(out.data);
    if (warps == 2) {
        if (direct) {
            ternary_mma_small_t_kernel<kPtq1_, 2, true, true, 2>
                <<<grid, 64, 0, stream>>>(xp, qp, high, sp, op, w.n, w.k, tokens, out_row_stride);
        } else if (ilp) {
            if (!kPtq1_ && small_t_t2v2() && tokens <= 8 && (w.n % 16) == 0 && (w.k % 512) == 0) {
                const unsigned t2_grid = static_cast<unsigned>(w.n / 16);
                tern_t2v2_kernel<TernaryT2v2Schedule<8, 1, 1, 2, 6>>
                    <<<t2_grid, 256, 0, stream>>>(xp, qp, sp, op, w.n, w.k, tokens,
                                                  out_row_stride);
            } else if (!kPtq1_ && small_t_bitplane()) {
                ternary_mma_small_t_kernel<kPtq1_, 2, true, false, 2, 0, true>
                    <<<grid, 64, 0, stream>>>(xp, qp, high, sp, op, w.n, w.k, tokens,
                                              out_row_stride);
            } else if (small_t_chunk128()) {
                ternary_mma_small_t_kernel<kPtq1_, 2, true, false, 2, 1>
                    <<<grid, 64, 0, stream>>>(xp, qp, high, sp, op, w.n, w.k, tokens,
                                              out_row_stride);
            } else if (small_t_weight_bufs() == 2) {
                ternary_mma_small_t_kernel<kPtq1_, 2, true, false, 2>
                    <<<grid, 64, 0, stream>>>(xp, qp, high, sp, op, w.n, w.k, tokens,
                                              out_row_stride);
            } else {
                ternary_mma_small_t_kernel<kPtq1_, 2, true>
                    <<<grid, 64, 0, stream>>>(xp, qp, high, sp, op, w.n, w.k, tokens,
                                              out_row_stride);
            }
        } else {
            ternary_mma_small_t_kernel<kPtq1_, 2>
                <<<grid, 64, 0, stream>>>(xp, qp, high, sp, op, w.n, w.k, tokens, out_row_stride);
        }
    } else if (warps == 1) {
        if (direct) {
            ternary_mma_small_t_kernel<kPtq1_, 1, true, true>
                <<<grid, 32, 0, stream>>>(xp, qp, high, sp, op, w.n, w.k, tokens, out_row_stride);
        } else if (ilp) {
            ternary_mma_small_t_kernel<kPtq1_, 1, true>
                <<<grid, 32, 0, stream>>>(xp, qp, high, sp, op, w.n, w.k, tokens, out_row_stride);
        } else {
            ternary_mma_small_t_kernel<kPtq1_, 1>
                <<<grid, 32, 0, stream>>>(xp, qp, high, sp, op, w.n, w.k, tokens, out_row_stride);
        }
    } else {
        if (direct) {
            ternary_mma_small_t_kernel<kPtq1_, 4, true, true>
                <<<grid, 128, 0, stream>>>(xp, qp, high, sp, op, w.n, w.k, tokens, out_row_stride);
        } else if (ilp) {
            ternary_mma_small_t_kernel<kPtq1_, 4, true>
                <<<grid, 128, 0, stream>>>(xp, qp, high, sp, op, w.n, w.k, tokens, out_row_stride);
        } else {
            ternary_mma_small_t_kernel<kPtq1_, 4>
                <<<grid, 128, 0, stream>>>(xp, qp, high, sp, op, w.n, w.k, tokens, out_row_stride);
        }
    }
    CUDA_CHECK(cudaGetLastError());
}

void launch_pq2_mma_small_t(const Tensor& x, const Weight& w, Tensor& out,
                            std::int32_t out_row_stride, std::int32_t tokens,
                            cudaStream_t stream) {
    // The staging is static shared memory (a compile-time 32 KB), so there is no dynamic size to
    // pass and no cudaFuncSetAttribute opt-in to make.
    launch_small_t_mma<false>(x, w, nullptr, out, out_row_stride, tokens, stream);
}

// PTQ1_0 twin: same kernel, same grid, same everything below the repack. Its static shared is
// 46.25 KB instead of 32 KB (three rotating weight-row buffers instead of two -- see
// kTernaryMmaPtq1WeightBufs), still under the 48 KB static cap and still two CTAs per SM.
void launch_ptq1_mma_small_t(const Tensor& x, const Weight& w, Tensor& out,
                             std::int32_t out_row_stride, std::int32_t tokens,
                             cudaStream_t stream) {
    if (w.qhigh == nullptr) {
        throw std::invalid_argument("ternary small-t mma: PTQ1_0 needs its high plane");
    }
    launch_small_t_mma<true>(x, w, static_cast<const std::uint8_t*>(w.qhigh), out, out_row_stride,
                             tokens, stream);
}

// One call site for both ternary formats, exactly like the GEMV and wide-t dispatchers.
void launch_small_t(const Tensor& x, const Weight& w, Tensor& out,
                    std::int32_t out_row_stride, std::int32_t tokens, cudaStream_t stream) {
    if (w.qtype == QType::PTQ1_0_G128) {
        launch_ptq1_mma_small_t(x, w, out, out_row_stride, tokens, stream);
        return;
    }
    launch_pq2_mma_small_t(x, w, out, out_row_stride, tokens, stream);
}

// int8 rung (#14): quantize the activation row per token, then run the s8 tensor-core kernel.
// Requires a caller-provided scratch (see TernaryS8Scratch): the quantization pass needs one int8
// code row per token plus one fp32 scale per token.
void launch_pq2_mma_s8(const Tensor& x, const Weight& w, Tensor& out,
                       std::int32_t out_row_stride, std::int32_t tokens, TernaryS8Scratch scratch,
                       cudaStream_t stream) {
    ternary_s8_quantize_kernel<<<tokens, 256, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), scratch.codes, scratch.scales, w.k);
    CUDA_CHECK(cudaGetLastError());

    constexpr int kWarps  = 4;
    // Decode window (T < 33): an 8-token tile keeps the s8 mma count minimal at T where the
    // prefill tile would waste most of its 64-token staging (round 17 negative motivated this).
    if (tokens < kTernaryS8MinTokens) {
        constexpr int kT8 = 8;
        const unsigned grid8 = static_cast<unsigned>(
            div_up(w.n, TernaryS8Storage<kT8, kWarps>::kRowsPerCta));
        ternary_pq2_mma_s8_kernel<kT8, kWarps, 3><<<grid8, kWarps * 32, 0, stream>>>(
            scratch.codes, scratch.scales, static_cast<const std::uint8_t*>(w.qdata),
            static_cast<const std::uint8_t*>(w.scales), static_cast<__nv_bfloat16*>(out.data), w.n,
            w.k, tokens, out_row_stride);
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    constexpr int kTokens = 64;
    const unsigned grid =
        static_cast<unsigned>(div_up(w.n, TernaryS8Storage<kTokens, kWarps>::kRowsPerCta));
    ternary_pq2_mma_s8_kernel<kTokens, kWarps, 3><<<grid, kWarps * 32, 0, stream>>>(
        scratch.codes, scratch.scales, static_cast<const std::uint8_t*>(w.qdata),
        static_cast<const std::uint8_t*>(w.scales), static_cast<__nv_bfloat16*>(out.data), w.n,
        w.k, tokens, out_row_stride);
    CUDA_CHECK(cudaGetLastError());
}

// PTQ1_0 twin of the int8 rung (2026-09-23). Same activation quantization pass, same mma, same
// staging geometry -- the only difference is that the weight staging cp.async's the raw 24+2-byte
// rows and repacks them into the PQ2_0 code layout in shared (the shared helper in
// ternary_ptq1_repack.cuh, also used by wide_t and small_t). Kernel-level accuracy on real shapes:
// rel_l2 ~9.0e-3, which IS the int8 activation quantization error itself (see the s8 header note),
// i.e. the same accuracy class as the PQ2_0 s8 rung -- this rung is not bit-exact by construction,
// and its gate is the E4 criterion the PQ2_0 rung already passed.
void launch_ptq1_mma_s8(const Tensor& x, const Weight& w, Tensor& out,
                        std::int32_t out_row_stride, std::int32_t tokens, TernaryS8Scratch scratch,
                        cudaStream_t stream) {
    if (w.qhigh == nullptr) {
        throw std::invalid_argument("ternary s8 mma: PTQ1_0 needs its high plane");
    }
    ternary_s8_quantize_kernel<<<tokens, 256, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), scratch.codes, scratch.scales, w.k);
    CUDA_CHECK(cudaGetLastError());

    constexpr int kWarps  = 4;
    if (tokens < kTernaryS8MinTokens) {
        constexpr int kT8 = 8;
        const unsigned grid8 = static_cast<unsigned>(
            div_up(w.n, TernaryS8Storage<kT8, kWarps, true>::kRowsPerCta));
        ternary_pq2_mma_s8_kernel<kT8, kWarps, 3, true><<<grid8, kWarps * 32, 0, stream>>>(
            scratch.codes, scratch.scales, static_cast<const std::uint8_t*>(w.qdata),
            static_cast<const std::uint8_t*>(w.scales), static_cast<__nv_bfloat16*>(out.data), w.n,
            w.k, tokens, out_row_stride, static_cast<const std::uint8_t*>(w.qhigh));
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    constexpr int kTokens = 64;
    const unsigned grid = static_cast<unsigned>(
        div_up(w.n, TernaryS8Storage<kTokens, kWarps, true>::kRowsPerCta));
    ternary_pq2_mma_s8_kernel<kTokens, kWarps, 3, true><<<grid, kWarps * 32, 0, stream>>>(
        scratch.codes, scratch.scales, static_cast<const std::uint8_t*>(w.qdata),
        static_cast<const std::uint8_t*>(w.scales), static_cast<__nv_bfloat16*>(out.data), w.n,
        w.k, tokens, out_row_stride, static_cast<const std::uint8_t*>(w.qhigh));
    CUDA_CHECK(cudaGetLastError());
}

// One call site for both ternary formats, like the GEMV / wide-t / small-t dispatchers.
void launch_s8(const Tensor& x, const Weight& w, Tensor& out, std::int32_t out_row_stride,
               std::int32_t tokens, TernaryS8Scratch scratch, cudaStream_t stream) {
    if (w.qtype == QType::PTQ1_0_G128) {
        launch_ptq1_mma_s8(x, w, out, out_row_stride, tokens, scratch, stream);
        return;
    }
    launch_pq2_mma_s8(x, w, out, out_row_stride, tokens, scratch, stream);
}

void launch_ternary_gemm_t8(const Tensor& x, const Weight& w, Tensor& out,
                            std::int32_t out_row_stride, cudaStream_t stream,
                            TernaryS8Scratch scratch) {
    // The speculative verify pass runs T = draft + 1 (2..4 here). The reference tiled kernel wastes
    // five of its eight token slots at that size and needs a 128-thread CTA plus seven barriers per
    // output row, which cost more than the whole decode step it was verifying. The tensor-core path
    // is preferred because its cost is flat in T; the small-tile GEMV is the fallback for shapes it
    // cannot take (PTQ1_0, padded K, or K not a whole number of 256-wide chunks).
    //
    // The tensor-core kernel walks the sequence in 8-token tiles, so it now covers PREFILL too, not
    // just the verify pass. Above 8 tokens the old code fell through to the reference kernel, which
    // is where prefill's 55 tok/s (against 412-437 t/s for the non-ternary engine) came from.
    //
    // But 8 tokens per weight pass is also the reason prefill is 3-4x behind the official runtime on
    // this card: this kernel restages the weight window for EVERY tile, so one forward reads the
    // weights ceil(T/8) times (measured: T=62 -> 131 ms = 8 passes x 6.70 GiB at 438 GiB/s, fitted
    // as t ~= 12.31 ms * ceil(T/8) + 44 ms over a 14x range of T). The wide-token-tile kernel keeps
    // the weight window staged across all tokens of its tile, dropping that to ceil(T/64). Rungs:
    //   T <= 8  -> small_t  (one pass already; also the verified MTP-verify path, left untouched)
    //   T >= 9  -> wide_t   (one pass up to 64 tokens: prefill, and batched multi-sequence passes,
    //                        where 11 tokens in one tile is what the 5-lane 25% efficiency costs)
    //   T >= 33 -> s8       (int8 operands: 1.20-1.30x the bf16 wide rung from T=40 up, and the
    //                        first rung that attacks the measured instruction/pipe oversubscription.
    //                        Only when the caller supplied the quantization scratch -- the *_basis
    //                        entry points have no workspace and stay on the bf16 rungs.)
    // Diagnostic for the rung dispatch: which path each call ACTUALLY takes, and why. Gated on an
    // env var and capped, so it costs nothing in normal runs. (The first version of this probe only
    // printed the CONDITIONS, which is how the PPL gate came back bit-identical with the int8 rung
    // enabled and still left the question open -- so this one is called from the branch that was
    // taken, after the decision, and names the rung.)   NINFER_TERNARY_S8_DEBUG=1 -> stderr lines
    //
    // 2026-09-23: the budget is env-tunable and a min-T filter was added, because ONE forward runs
    // ~320 ternary calls, so the fixed 24-line budget was entirely consumed by the very first
    // forward -- which left "which rung does the T=6 verify step of --draft-tokens 5 take"
    // unanswerable, and that gap was already recorded (stale-assets S-23: "make the print budget env
    // adjustable, default 24").  NINFER_TERNARY_S8_DEBUG_BUDGET (default 24);
    // NINFER_TERNARY_S8_DEBUG_MIN_T (default 0) prints only calls with at least that many tokens.
    static const int probe_min_t = []() {
        const char* env = std::getenv("NINFER_TERNARY_S8_DEBUG_MIN_T");
        return env != nullptr ? std::atoi(env) : 0;
    }();
    static int probe_budget = []() {
        const char* env = std::getenv("NINFER_TERNARY_S8_DEBUG_BUDGET");
        return env != nullptr ? std::atoi(env) : 24;
    }();
    const char* const probe_fmt = (w.qtype == QType::PTQ1_0_G128) ? "PTQ1"
                                : (w.qtype == QType::PQ2_0_G128)  ? "PQ2"
                                                                  : "other";
    const auto note_rung = [&](const char* rung) {
        if (probe_budget <= 0 || std::getenv("NINFER_TERNARY_S8_DEBUG") == nullptr) { return; }
        if (static_cast<int>(x.ne[1]) < probe_min_t) { return; }
        --probe_budget;
        std::fprintf(stderr,
                     "[ternary] rung=%s fmt=%s qtype=%d T=%d k=%d n=%d stride=%d wide_admits=%d "
                     "small_admits=%d gemv_admits=%d s8_scratch=%d s8_switch=%d s8_min=%d mma=%d "
                     "ptq1_fast=%d\n",
                     rung, probe_fmt, static_cast<int>(w.qtype), static_cast<int>(x.ne[1]),
                     static_cast<int>(w.k), static_cast<int>(w.n),
                     static_cast<int>(out_row_stride), mma_wide_admits(w, out_row_stride) ? 1 : 0,
                     mma_admits(w, out_row_stride) ? 1 : 0, gemv_admits(x, w, 4) ? 1 : 0,
                     scratch.codes != nullptr ? 1 : 0, ternary_s8_enabled() ? 1 : 0,
                     kTernaryS8MinTokens, ternary_mma_enabled() ? 1 : 0,
                     ternary_ptq1_fast_enabled() ? 1 : 0);
    };
    if (scratch.codes != nullptr && scratch.scales != nullptr && ternary_s8_enabled() &&
        ternary_mma_enabled() &&
        // 2026-09-23: PTQ1_0 is admitted here on the kPtq1 instantiation, which repacks its raw rows
        // into the PQ2_0 code layout in shared first. The old PQ2_0-only guard was correct while that
        // instantiation did not exist: the s8 kernel reads the 2-bit code plane DIRECTLY, so feeding
        // it base-3 bytes would have mis-decoded silently. With the repack in the staging the hazard
        // is gone, and the accuracy class is unchanged (rel_l2 ~9.0e-3 == the int8 activation error
        // itself, same as PQ2_0's s8). This rung is what PQ2_0 prefill has been running all along --
        // PTQ1_0 was simply missing it, which was the whole 2.36x prefill gap (measured on a
        // 4,044-token prompt: 635.7 against 1,500.0 tok/s).
        //
        // 2026-09-28 (round 17): NINFER_S8_DECODE=1 also admits the decode window (T = 2..32),
        // trading per-weight decode instructions for int8 mma -- the round-15 architecture research
        // tied the decode round cost to that chain.
        (w.qtype == QType::PQ2_0_G128 || ptq1_repack_admits(w)) &&
        mma_wide_admits(w, out_row_stride) &&
        (x.ne[1] >= kTernaryS8MinTokens || ternary_s8_decode_admits(x.ne[1]))) {
        note_rung("s8");
        launch_s8(x, w, out, out_row_stride, x.ne[1], scratch, stream);
        return;
    }
    if (ternary_mma_enabled() && mma_wide_admits(w, out_row_stride) &&
        x.ne[1] >= ternary_wide_min_tokens()) {
        note_rung("wide_t");
        launch_wide_t(x, w, out, out_row_stride, x.ne[1], stream);
        return;
    }
    if (ternary_mma_enabled() && mma_admits(w, out_row_stride) && x.ne[1] >= 2) {
        note_rung("small_t");
        launch_small_t(x, w, out, out_row_stride, x.ne[1], stream);
        return;
    }
    if (gemv_admits(x, w, 4)) {
#if defined(NINFER_HIP)
        if (ternary_pq2_dp4a_enabled() && scratch.dp4a_xq != nullptr &&
            w.qtype == QType::PQ2_0_G128) {
            note_rung("gemv_tile_dp4a");
            launch_pq2_dp4a_tile(x, w, out, out_row_stride, stream, scratch);
            return;
        }
#endif
        note_rung("gemv_tile");
        launch_gemv_tile(x, w, out, out_row_stride, x.ne[1], stream);
        return;
    }
    // REFERENCE-PATH TILE WIDTH -- 8, and a wider tile is a MEASURED REGRESSION, not an oversight.
    // 2026-09-23: this fallback was what PTQ1_0 prefill ran on when a wider tile was tried to
    // amortise the seven CTA-wide reductions per tile (the MMA rungs were PQ2-only then; PTQ1_0 has
    // since been admitted to wide_t and small_t, so it is now reached only for shapes the rungs
    // reject -- padded K, a non-512-multiple K, or the gate switches off).
    // kTileT = 32 (threshold 33 tokens) measured **45.2 -> 19.0 t/s** on a 2,536-token prompt
    // (5,036 tokens: 19.0 t/s too), decode unchanged. The reason is occupancy, not overhead: 32
    // accumulators per thread plus a 16 KB partials tile cost resident CTAs, and this kernel is
    // latency-bound (one FMA per weight per token, weight decoded once per group per tile). Do NOT
    // raise kTileT again -- the way to make PTQ1_0 prefill fast is the tensor-core rung, not this.
    note_rung("reference");
    launch_by_qtype<8>(x, w, out, out_row_stride, stream);
}

} // namespace ninfer::ops::detail

#include "ops/linear/ternary/ternary_dispatch.h"

#include "ops/linear/ternary/ternary_rotation.h"
#include "ops/linear/ternary/ternary_row_view.h"
#include "ops/linear/ternary/ternary_rowsplit_storage.cuh"
#include "ops/linear/ternary/ternary_s8_scratch.h"
#include "ops/linear/ternary/ternary_small_t_wmma.cuh"
#include "ops/linear/ternary/ternary_big_t_wmma.cuh"
#include "ops/linear/ternary/ternary_t2v2.cuh"

#include "core/device.h"  // CUDA_CHECK

#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

namespace ninfer::ops::detail {

namespace {

// NINFER_TERNARY_SMALLT_WMMA: route small-T (2..16 col) PQ2 GEMMs through the WMMA kernel
// (ternary_small_t_wmma.cuh). Default ON since 2026-09-30 (probe bit-exact; mtp2 A/B
// 17.47 vs 5.31 tok/s = 3.3x, acceptance 43.5% -> 47.5%); =0 falls back to the SIMT path.
bool smallt_wmma_enabled() {
    static const bool enabled = [] {
        const char* v = std::getenv("NINFER_TERNARY_SMALLT_WMMA");
        return v == nullptr || std::atoi(v) != 0;
    }();
    return enabled;
}

bool smallt_wmma_admits(const Weight& w, std::int32_t t) {
    return smallt_wmma_enabled() && t >= 2 && t <= 16 && w.qtype == QType::PQ2_0_G128 &&
           w.k % 128 == 0 && w.k >= 512 && w.padded_shape[1] == w.k;
}

void launch_smallt_wmma(const Tensor& x, const Weight& w, Tensor& out, std::int32_t out_row_stride,
                        cudaStream_t stream) {
    TernarySmallTJob job{static_cast<const __nv_bfloat16*>(x.data),
                         static_cast<const std::uint8_t*>(w.qdata),
                         static_cast<const std::uint8_t*>(w.scales),
                         static_cast<__nv_bfloat16*>(out.data), w.n, w.k, w.padded_shape[1],
                         out_row_stride};
    const int grid = (w.n + 31) / 32;
    ternary_small_t_wmma_kernel<<<grid, 256, 0, stream>>>(job, x.ne[1], nullptr);
    CUDA_CHECK(cudaGetLastError());
}

// NINFER_TERNARY_BIGT_WMMA: route prefill-size (t >= 17) PQ2 GEMMs through the WMMA kernel
// (ternary_big_t_wmma.cuh) instead of the s8/wide_t rungs (FMA-emulated mma chain). Default ON
// since 2026-09-30 (probe 8/8 bit-exact; greedy 48-tok identical to the s8 rung; mtp3 coherent;
// prefill 14.1 -> 88.2 tok/s, serve TTFT 6.75 s -> 0.53 s); =0 falls back to the s8 rung.
bool bigt_wmma_enabled() {
    static const bool enabled = [] {
        const char* v = std::getenv("NINFER_TERNARY_BIGT_WMMA");
        return v == nullptr || std::atoi(v) != 0;
    }();
    return enabled;
}

bool bigt_wmma_admits(const Weight& w, std::int32_t t) {
    return bigt_wmma_enabled() && t >= 17 && w.qtype == QType::PQ2_0_G128 &&
           w.k % 128 == 0 && w.padded_shape[1] == w.k;
}

void launch_bigt_wmma(const Tensor& x, const Weight& w, Tensor& out, std::int32_t out_row_stride,
                      cudaStream_t stream) {
    TernaryBigTJob job{static_cast<const __nv_bfloat16*>(x.data),
                       static_cast<const std::uint8_t*>(w.qdata),
                       static_cast<const std::uint8_t*>(w.scales),
                       static_cast<__nv_bfloat16*>(out.data), w.n, w.k, w.padded_shape[1],
                       out_row_stride};
    const dim3 grid((w.n + 63) / 64, (x.ne[1] + 127) / 128);
    ternary_big_t_wmma_kernel<128, 256><<<grid, 256, 0, stream>>>(job, x.ne[1]);
    CUDA_CHECK(cudaGetLastError());
}

}  // namespace

TernaryLaunch select_ternary_launch(std::int32_t n, std::int32_t k, std::int32_t t,
                                    LinearPolicy policy) {
    if (n <= 0 || k <= 0 || t <= 0) {
        throw std::invalid_argument("ternary linear: unsupported shape or T");
    }
    // The reference kernel decodes whole 128-weight groups, so K has to be a whole number
    // of them. Every width the model uses (5120, 6144, 10240, 17408) satisfies this, and
    // row_split_geometry() pads to the same 128 boundary, so no padding groups exist.
    if (k % PTQ1RowSplitStorage::kGroupK != 0) {
        throw std::invalid_argument("ternary linear: K must be a multiple of the group size");
    }
    switch (policy) {
    case LinearPolicy::A16Only:
    case LinearPolicy::AllowA8:
    case LinearPolicy::AllowA4:
        break;
    }
    // Both ternary formats take the same schedule; the atom is chosen from w.qtype inside
    // the launch. Decode (T = 1) and small prefill use different token tiles.
    return t == 1 ? launch_ternary_gemm_t1 : launch_ternary_gemm_t8;
}

TernaryS8Scratch allocate_ternary_s8_scratch(WorkspaceArena& workspace, std::int32_t k,
                                             std::int32_t tokens) {
    TernaryS8Scratch scratch{};
    // dp4a rung(PQ2 decode GEMV/tile, T≤4): 门开即从同一 arena 分配。这里看不到 qtype,
    // PTQ1/BF16 权重会白拿几 KB scratch, launch 侧的 PQ2 条件会安全落回 bf16 arm。
    // 容量端: 通用路径 = ternary_rotation_workspace_bytes 的 dp4a 片(按 T=4 计)✓;
    // wrapper 叶池 = cut3 的容量补丁(见 gdn_input_proj/attn_input_proj 容量函数),
    // 2026-10-03 首版因叶池未计片而越界(arena 现抛带实数的 overflow)。
    if (tokens >= 1 && tokens <= 4 && ternary_pq2_dp4a_enabled() && (k % 128) == 0) {
        const DeviceSpan xq  = workspace.alloc_bytes(ternary_pq2_dp4a_xq_bytes(k, tokens));
        const DeviceSpan ds8 = workspace.alloc_bytes(ternary_pq2_dp4a_ds8_bytes(k, tokens));
        scratch.dp4a_xq      = static_cast<std::int8_t*>(xq.data);
        scratch.dp4a_ds8     = static_cast<std::uint16_t*>(ds8.data);
    }
    if (tokens < kTernaryS8MinTokens) {
        // Below the int8 rung's threshold the rung is not taken, so the scratch would be wasted
        // bytes -- allocate nothing and let the launch fall through to the bf16 rungs.
        // dp4a 字段(如有)已在上面填好, 随 scratch 带出。
        return scratch;
    }
    const DeviceSpan codes  = workspace.alloc_bytes(ternary_s8_codes_bytes(k, tokens));
    const DeviceSpan scales = workspace.alloc_bytes(ternary_s8_scales_bytes(tokens));
    scratch.codes           = static_cast<std::int8_t*>(codes.data);
    scratch.scales          = static_cast<float*>(scales.data);
    return scratch;
}

void ternary_dispatch_basis_strided(const Tensor& x_folded, const Weight& w, Tensor& out,
                                    std::int32_t out_row_stride, LinearPolicy policy,
                                    cudaStream_t stream, TernaryS8Scratch scratch) {
    // dp4a rung(NINFER_PQ2_DP4A=1, T=1..4): 置于 small-T WMMA 之前, 与 WMMA 正面 A/B。
    // 门关/scratch 空(容量端没分)时条件不成立, 落回原路由。判据与"旋转延迟"共用一份(M2)。
    if (ternary_pq2_dp4a_admits(x_folded, w, scratch)) {
        launch_pq2_dp4a_tile(x_folded, w, out, out_row_stride, stream, scratch);
        return;
    }
    // 安全网(M2): 调用点声明过"激活未旋转、旋转由融合核负责", 但这次派发没走 dp4a 分支 ——
    // 继续往下就是在未旋转的激活上乘折叠权重(数值垃圾)。宁可炸。
    if (scratch.rot_deferred()) {
        throw std::logic_error(
            "ternary linear: deferred rotation reached a non-dp4a rung (activation is not rotated)");
    }
    // out_row_stride may differ from w.n (verify-fused qkv buffers, e.g. 10240-stride gdn/attn
    // projections): the kernel writes [col * stride + row], so the strided layouts stay on WMMA
    // instead of falling back to the FMA-sim rungs (measured 6.8x gdn_proj round cost, B1).
    // 2026-10-08 (转GEMM 线2): NINFER_VERIFY_T2V2=1 时 t 2..8 的 strided 投影改走 t2v2
    // (上游 t2_small_t_v2 移植件, T=1 A/B 与 dp4a GEMV 逐位一致同速; verify 摊销假设的
    // 对照臂)。默认关; =0 恢复 WMMA。
    static const bool verify_t2v2 = []() {
        const char* v = std::getenv("NINFER_VERIFY_T2V2");
        return v != nullptr && std::strcmp(v, "0") != 0;
    }();
    if (verify_t2v2 && x_folded.ne[1] >= 2 && x_folded.ne[1] <= 8 &&
        w.qtype == QType::PQ2_0_G128 && (w.n % 16) == 0 && (w.k % 512) == 0 &&
        w.padded_shape[1] == w.k) {
        const unsigned t2_grid = static_cast<unsigned>(w.n / 16);
        tern_t2v2_kernel<TernaryT2v2Schedule<8, 1, 1, 2, 6>>
            <<<t2_grid, 256, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(x_folded.data),
                static_cast<const std::uint8_t*>(w.qdata),
                static_cast<const std::uint8_t*>(w.scales),
                static_cast<__nv_bfloat16*>(out.data), w.n, w.k, x_folded.ne[1],
                out_row_stride);
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    if (smallt_wmma_admits(w, x_folded.ne[1])) {
        launch_smallt_wmma(x_folded, w, out, out_row_stride, stream);
        return;
    }
    if (bigt_wmma_admits(w, x_folded.ne[1])) {
        launch_bigt_wmma(x_folded, w, out, out_row_stride, stream);
        return;
    }
    const TernaryLaunch launch = select_ternary_launch(w.n, w.k, x_folded.ne[1], policy);
    // No workspace on this entry point: the CALLER already folded the activation, and may hand in
    // the int8 scratch it allocated from its own arena (see allocate_ternary_s8_scratch). With an
    // empty scratch these calls stay on the bf16 rungs, which is what they did before 2026-09-23.
    launch(x_folded, w, out, out_row_stride, stream, scratch);
}

void ternary_dispatch_basis(const Tensor& x_folded, const Weight& w, Tensor& out,
                            LinearPolicy policy, cudaStream_t stream, TernaryS8Scratch scratch) {
    ternary_dispatch_basis_strided(x_folded, w, out, w.n, policy, stream, scratch);
}

void ternary_dispatch_basis_seg2(const Tensor& x_folded, const Weight& w, Tensor& out_a,
                                 std::int32_t rows_a, std::int32_t stride_a, Tensor& out_b,
                                 std::int32_t stride_b, LinearPolicy policy, cudaStream_t stream,
                                 TernaryS8Scratch scratch) {
    const std::int32_t rows_b = w.n - rows_a;
    if (rows_a <= 0 || rows_b <= 0) {
        throw std::invalid_argument("ternary seg2: both row ranges must be positive");
    }
    // dp4a 准入时一段发射(M2c);否则回退为两次行切片派发(旧行为, 逐位同)。
    // 回退时激活必须已旋转 —— 延迟旋转只由 dp4a 融合核消费, 到不了这里就该炸。
    if (ternary_seg2_fuse_enabled() && ternary_pq2_dp4a_admits(x_folded, w, scratch)) {
        launch_pq2_dp4a_seg2(x_folded, w, out_a, rows_a, stride_a, out_b, stride_b, stream,
                             scratch);
        return;
    }
    if (scratch.rot_deferred()) {
        throw std::logic_error(
            "ternary linear: deferred rotation reached a non-dp4a rung (activation is not rotated)");
    }
    const Weight head = ternary_row_view(w, 0, rows_a);
    const Weight tail = ternary_row_view(w, rows_a, rows_b);
    ternary_dispatch_basis_strided(x_folded, head, out_a, stride_a, policy, stream, scratch);
    ternary_dispatch_basis_strided(x_folded, tail, out_b, stride_b, policy, stream, scratch);
}

void ternary_dispatch_basis_segn(const Tensor& x_folded, const TernarySegDesc* segs,
                                 std::int32_t nsegs, LinearPolicy policy, cudaStream_t stream,
                                 TernaryS8Scratch scratch) {
    if (nsegs < 2 || nsegs > 4) {
        throw std::invalid_argument("ternary segn: segment count must be 2..4");
    }
    const std::int32_t k = x_folded.ne[0];
    bool all_admit = true;
    for (std::int32_t i = 0; i < nsegs; ++i) {
        const Weight& w = *segs[i].w;
        if (w.k != k || w.qtype != QType::PQ2_0_G128 || w.padded_shape[1] != w.k ||
            (w.k % 128) != 0) {
            all_admit = false;
            break;
        }
    }
    // T=1 + 全段准入时一段发射(M2d);否则回退逐段派发(旧行为, 逐位同)。
    // 延迟旋转只由 dp4a 融合核消费 —— 回退路径(经 strided 派发)自己有守卫。
    if (ternary_segn_fuse_enabled() && x_folded.ne[1] == 1 && all_admit &&
        ternary_pq2_dp4a_admits(x_folded, *segs[0].w, scratch)) {
        launch_pq2_dp4a_segn(x_folded, segs, nsegs, stream, scratch);
        return;
    }
    if (scratch.rot_deferred()) {
        throw std::logic_error(
            "ternary linear: deferred rotation reached a non-dp4a rung (activation is not rotated)");
    }
    for (std::int32_t i = 0; i < nsegs; ++i) {
        ternary_dispatch_basis_strided(x_folded, *segs[i].w, *segs[i].out, segs[i].stride, policy,
                                       stream, scratch);
    }
}

bool ternary_dispatch_add(const Tensor& x, const Weight& w, Tensor& residual_out,
                          LinearPolicy policy, WorkspaceArena* workspace, cudaStream_t stream,
                          bool pre_rotated) {
    (void)policy;
    // 外层条件 = dp4a 分支的判据减去 scratch 测试(scratch 在下面才分配), 与
    // ternary_dispatch 的 dp4a 分支同形。不满足就返回 false, 由调用方 compose。
    if (!ternary_res_fuse_enabled() || workspace == nullptr || x.ne[1] > 4 ||
        !ternary_pq2_dp4a_enabled() || w.qtype != QType::PQ2_0_G128 ||
        w.padded_shape[1] != w.k || (w.k % 128) != 0) {
        return false;
    }
    auto scope = workspace->scope();
    TernaryS8Scratch scratch{};
    const DeviceSpan xq  = workspace->alloc_bytes(ternary_pq2_dp4a_xq_bytes(w.k, x.ne[1]));
    const DeviceSpan ds8 = workspace->alloc_bytes(ternary_pq2_dp4a_ds8_bytes(w.k, x.ne[1]));
    scratch.dp4a_xq      = static_cast<std::int8_t*>(xq.data);
    scratch.dp4a_ds8     = static_cast<std::uint16_t*>(ds8.data);
    // pre_rotated(M2e): x 已是折叠基(silu_mul_rotate 产物), 跳过旋转;量化直接读它。
    // dp4a 准入不满足时必须炸(回落 compose 会在已旋转激活上再旋转 = 数值垃圾)。
    if (pre_rotated) {
        launch_pq2_dp4a_tile(x, w, residual_out, w.n, stream, scratch, &residual_out);
        return true;
    }
    const Tensor activation = folded_activation_or_defer(x, w, *workspace, stream, scratch);
    // 原地: residual_out 既是残差输入也是投影输出(GEMV epilogue 先读后写)。
    launch_pq2_dp4a_tile(activation, w, residual_out, w.n, stream, scratch, &residual_out);
    return true;
}

void ternary_dispatch(const Tensor& x, const Weight& w, Tensor& out, LinearPolicy policy,
                      WorkspaceArena* workspace, cudaStream_t stream) {
    // dp4a verify-tile rung(NINFER_PQ2_DP4A=1, T=1..4): 同 strided 入口, WMMA 之前 A/B。
    if (x.ne[1] <= 4 && ternary_pq2_dp4a_enabled() && w.qtype == QType::PQ2_0_G128 &&
        w.padded_shape[1] == w.k && (w.k % 128) == 0 && workspace != nullptr) {
        auto scope = workspace->scope();
        TernaryS8Scratch scratch{};
        const DeviceSpan xq  = workspace->alloc_bytes(ternary_pq2_dp4a_xq_bytes(w.k, x.ne[1]));
        const DeviceSpan ds8 = workspace->alloc_bytes(ternary_pq2_dp4a_ds8_bytes(w.k, x.ne[1]));
        scratch.dp4a_xq      = static_cast<std::int8_t*>(xq.data);
        scratch.dp4a_ds8     = static_cast<std::uint16_t*>(ds8.data);
        // M2: scratch 先分配再决定旋转走向(延迟时不需要旋转缓冲)。
        const Tensor activation = folded_activation_or_defer(x, w, *workspace, stream, scratch);
        launch_pq2_dp4a_tile(activation, w, out, w.n, stream, scratch);
        return;
    }
    if (workspace != nullptr &&
        (smallt_wmma_admits(w, x.ne[1]) || bigt_wmma_admits(w, x.ne[1]))) {
        auto scope = workspace->scope();
        const Tensor activation = folded_activation(x, w, *workspace, stream);
        if (smallt_wmma_admits(w, x.ne[1])) {
            launch_smallt_wmma(activation, w, out, w.n, stream);
        } else {
            launch_bigt_wmma(activation, w, out, w.n, stream);
        }
        return;
    }
    const TernaryLaunch launch = select_ternary_launch(w.n, w.k, x.ne[1], policy);

    if (!ternary_rotation_enabled()) {
        // NINFER_TERNARY_HADAMARD=0: run the GEMM against the raw activation so the full forward
        // pass (and therefore the M4 decode speed) is measurable. The result is numerically
        // meaningless -- the activation is in the wrong basis -- which is the point: it separates
        // "the ternary decode is broken" from "the rotation is broken" without a rebuild.
        launch(x, w, out, w.n, stream, TernaryS8Scratch{});
        return;
    }
    if (workspace == nullptr) {
        // Name the shape in the message: a folded ternary weight reaching the workspace-free
        // entry point is a call-site wiring gap, and the shape is what identifies the weight.
        throw std::invalid_argument(
            "ternary linear: the folded basis needs a rotation workspace [N=" +
            std::to_string(w.n) + ", K=" + std::to_string(w.k) + ", T=" +
            std::to_string(x.ne[1]) + ", qtype=" + std::to_string(static_cast<int>(w.qtype)) + "]");
    }

    // Scoped: the scratch is handed back when this op returns, so it does not accumulate across
    // the (many) graph constructions of one load. Sequential reuse on one stream is safe.
    // folded_activation() rejects a ternary weight with no sign block, so an unfolded artifact
    // fails loudly here instead of silently multiplying by unfolded weights.
    auto scope              = workspace->scope();
    const Tensor activation = folded_activation(x, w, *workspace, stream);

    // int8 rung scratch (#14): one int8 code row per token plus one fp32 scale per token. Taken from
    // the same arena the rotation just used, and counted by ternary_rotation_workspace_bytes() so
    // the planner sizes the arena for it -- allocating it lazily inside the launch would be illegal,
    // because this op runs inside captured CUDA graphs.
    TernaryS8Scratch scratch{};
    if (x.ne[1] >= kTernaryS8MinTokens || ternary_s8_decode_admits(x.ne[1])) {
        const DeviceSpan codes  = workspace->alloc_bytes(ternary_s8_codes_bytes(w.k, x.ne[1]));
        const DeviceSpan scales = workspace->alloc_bytes(ternary_s8_scales_bytes(x.ne[1]));
        scratch.codes           = static_cast<std::int8_t*>(codes.data);
        scratch.scales          = static_cast<float*>(scales.data);
    }
    launch(activation, w, out, w.n, stream, scratch);
}

} // namespace ninfer::ops::detail

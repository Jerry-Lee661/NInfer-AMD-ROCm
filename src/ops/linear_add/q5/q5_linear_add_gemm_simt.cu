#include "ops/linear_add/q5/q5_linear_add_kernels.h"

#include "core/device.h"
#include "ops/linear/q5/q5_rowsplit_gemm_simt.cuh"

#include <cuda_bf16.h>

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

// 保序行分块核（2026-09-29）：down（3.6 GB）+ 两个输出投影（1.32 GB）在 T=2..16 全走
// Split2ExactResidual，实测 310-450 GB/s。split2 每行 2 warp 各做半 k（x 每行全量重读一遍，
// C8 下 x 流量 ≈ 1.43 GB L2 vs 权重 58.5 MB DRAM）。本核每 warp 吃 RowsPerWarp 行、x 每份喂
// RowsPerWarp 行的 fma 链（x 流量 ÷RowsPerWarp），并**逐位复刻 split2 的累加顺序**：
// 每 (行,列) 两个部分和 accA（偶 chunk：s 内 chunk 0,2 → 下一 slab 8,10 ...）/ accB（奇 chunk），
// 各自 warp 归约后相加再加残差，fma 序与 split2 的 part0/part1 逐项一致 → 输出逐位相同。
// NINFER_Q5LAD_RB=2/4 进路由（Cols>8 时 RW 钳到 2 控寄存器），默认关。
constexpr int kLadRbWarps = 8;

template <int Cols, int FullSlabs, int Stride, int RowsPerWarp>
__global__ __launch_bounds__(kLadRbWarps * 32, 1) void q5_linear_add_rb_kernel(
    const __nv_bfloat16* __restrict__ x, const std::uint8_t* __restrict__ codes,
    const std::uint8_t* __restrict__ high, const std::uint8_t* __restrict__ scales,
    __nv_bfloat16* __restrict__ out, std::int32_t n, std::int32_t padded_k) {
    const int lane     = static_cast<int>(threadIdx.x) & 31;
    const int warp     = static_cast<int>(threadIdx.x) >> 5;
    const int row_base =
        static_cast<int>(blockIdx.x) * (kLadRbWarps * RowsPerWarp) + warp * RowsPerWarp;

    const int kg_padded = padded_k / Q5RowSplitStorage::kGroupK;
    const std::uint8_t* code_row[RowsPerWarp];
    const std::uint8_t* high_row[RowsPerWarp];
    const std::uint8_t* scale_row[RowsPerWarp];
    bool row_valid[RowsPerWarp];
#pragma unroll
    for (int r = 0; r < RowsPerWarp; ++r) {
        const int row = row_base + r;
        row_valid[r]  = row < n;
        code_row[r]   = codes + static_cast<std::int64_t>(row) * kg_padded * 32;
        high_row[r]   = high + static_cast<std::int64_t>(row) * kg_padded *
                                   Q5RowSplitSimtSchedule::kHighBytesPerGroup;
        scale_row[r] = scales + static_cast<std::int64_t>(row) * kg_padded * 2;
    }

    float acc_a[RowsPerWarp][Cols];
    float acc_b[RowsPerWarp][Cols];
#pragma unroll
    for (int r = 0; r < RowsPerWarp; ++r) {
#pragma unroll
        for (int tt = 0; tt < Cols; ++tt) {
            acc_a[r][tt] = 0.0f;
            acc_b[r][tt] = 0.0f;
        }
    }

    constexpr int kHalfChunksPerSlab = 2;  // 每 warp 每 slab 处理的 chunk 数（=2，与 split2 同）
    const __half2 bias               = __half2half2(__ushort_as_half(0x6410));  // 1040.0

// unroll 1： RW×Cols×2 的累加器 + RW×8 瞬态权重不容相位展开（q4 RB 的 240 VGPR 教训）。
#pragma unroll 1
    for (int s = 0; s < FullSlabs; ++s) {
#pragma unroll
        for (int local = 0; local < kHalfChunksPerSlab; ++local) {
#pragma unroll
            for (int half = 0; half < 2; ++half) {
                // half=0 → 偶 chunk（split2 part0 的序），half=1 → 奇 chunk（part1 的序）
                const int chunk = local * 2 + half;
                std::uint32_t word[RowsPerWarp];
                std::uint32_t hc[RowsPerWarp];
                float scale[RowsPerWarp];
#pragma unroll
                for (int r = 0; r < RowsPerWarp; ++r) {
                    if (row_valid[r]) {
                        word[r] = *reinterpret_cast<const std::uint32_t*>(
                            code_row[r] + static_cast<std::int64_t>(s) * 512 + chunk * 128 +
                            lane * 4);
                        hc[r] = static_cast<std::uint32_t>(
                                    high_row[r][static_cast<std::int64_t>(s) * 128 + chunk * 32 +
                                                lane]) ^
                                0xffu;
                        const int group_in_slab = chunk * 4 + (lane >> 3);
                        scale[r] =
                            __half2float(__ushort_as_half(*reinterpret_cast<const std::uint16_t*>(
                                scale_row[r] +
                                static_cast<std::int64_t>(s) * 32 + group_in_slab * 2)));
                    }
                }
                float w[RowsPerWarp][8];
#pragma unroll
                for (int r = 0; r < RowsPerWarp; ++r) {
                    if (!row_valid[r]) { continue; }
#pragma unroll
                    for (int p = 0; p < 4; ++p) {
                        std::uint32_t bits = ((word[r] >> (4 * p)) & 0x000f000fu) | 0x64006400u;
                        bits |= (((hc[r] >> p) & 1u) << 4) | (((hc[r] >> (p + 4)) & 1u) << 20);
                        const __half2 h  = __hsub2(half2_from_bits(bits), bias);
                        const float2 f   = __half22float2(h);
                        w[r][p]          = f.x * scale[r];
                        w[r][p + 4]      = f.y * scale[r];
                    }
                }

                const std::int64_t xoff =
                    static_cast<std::int64_t>(s) * 1024 + chunk * 256 + lane * 8;
#pragma unroll
                for (int tt = 0; tt < Cols; ++tt) {
                    const uint4 xv =
                        load_vec<uint4>(x + static_cast<std::int64_t>(tt) * Stride + xoff);
                    const float2 f0 = bf16x2_bits_to_float2(xv.x);
                    const float2 f1 = bf16x2_bits_to_float2(xv.y);
                    const float2 f2 = bf16x2_bits_to_float2(xv.z);
                    const float2 f3 = bf16x2_bits_to_float2(xv.w);
#pragma unroll
                    for (int r = 0; r < RowsPerWarp; ++r) {
                        if (!row_valid[r]) { continue; }
                        if (half == 0) {
                            acc_a[r][tt] = fmaf(w[r][0], f0.x, acc_a[r][tt]);
                            acc_a[r][tt] = fmaf(w[r][1], f0.y, acc_a[r][tt]);
                            acc_a[r][tt] = fmaf(w[r][2], f1.x, acc_a[r][tt]);
                            acc_a[r][tt] = fmaf(w[r][3], f1.y, acc_a[r][tt]);
                            acc_a[r][tt] = fmaf(w[r][4], f2.x, acc_a[r][tt]);
                            acc_a[r][tt] = fmaf(w[r][5], f2.y, acc_a[r][tt]);
                            acc_a[r][tt] = fmaf(w[r][6], f3.x, acc_a[r][tt]);
                            acc_a[r][tt] = fmaf(w[r][7], f3.y, acc_a[r][tt]);
                        } else {
                            acc_b[r][tt] = fmaf(w[r][0], f0.x, acc_b[r][tt]);
                            acc_b[r][tt] = fmaf(w[r][1], f0.y, acc_b[r][tt]);
                            acc_b[r][tt] = fmaf(w[r][2], f1.x, acc_b[r][tt]);
                            acc_b[r][tt] = fmaf(w[r][3], f1.y, acc_b[r][tt]);
                            acc_b[r][tt] = fmaf(w[r][4], f2.x, acc_b[r][tt]);
                            acc_b[r][tt] = fmaf(w[r][5], f2.y, acc_b[r][tt]);
                            acc_b[r][tt] = fmaf(w[r][6], f3.x, acc_b[r][tt]);
                            acc_b[r][tt] = fmaf(w[r][7], f3.y, acc_b[r][tt]);
                        }
                    }
                }
            }
        }
    }

#pragma unroll
    for (int r = 0; r < RowsPerWarp; ++r) {
        if (!row_valid[r]) { continue; }
        const int row = row_base + r;
#pragma unroll
        for (int tt = 0; tt < Cols; ++tt) {
            float sum = warp_reduce_sum(acc_a[r][tt]) + warp_reduce_sum(acc_b[r][tt]);
            if (lane == 0) {
                const std::int64_t index = static_cast<std::int64_t>(tt) * n + row;
                sum += __bfloat162float(out[index]);
                out[index] = __float2bfloat16(sum);
            }
        }
    }
}

template <int Cols, int FullSlabs, int Stride, int RowsPerWarp>
void launch_rb(const Tensor& x, const Weight& w, Tensor& residual_out, cudaStream_t stream) {
    const dim3 grid(
        static_cast<unsigned>(div_up(residual_out.ne[0], kLadRbWarps * RowsPerWarp)), 1u, 1u);
    q5_linear_add_rb_kernel<Cols, FullSlabs, Stride, RowsPerWarp>
        <<<grid, kLadRbWarps * 32, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(w.qdata),
            static_cast<const std::uint8_t*>(w.qhigh), static_cast<const std::uint8_t*>(w.scales),
            static_cast<__nv_bfloat16*>(residual_out.data), residual_out.ne[0], w.padded_shape[1]);
    CUDA_CHECK(cudaGetLastError());
}

int lad_rb_rows() {
    static const int rb_rows = [] {
        const char* value = std::getenv("NINFER_Q5LAD_RB");
        if (value == nullptr) { return 0; }
        const int v = std::atoi(value);
        return (v == 2 || v == 4) ? v : 0;
    }();
    return rb_rows;
}

// split4 残差收尾（2026-09-29）：down/两输出投影（k=17408，cols 2..16）现走 split2
// （每行 2 warp、链长 34 chunk、310-450 GB/s）。行少 k 深的形状 RB 已判负（aaaea34），
// 正确方向 = 加深 k 切分：split4 每行 4 warp、链长 17 chunk、并行翻倍。归约序与 split2
// 不同（四部分和连加 vs 两部分和），属 lean attention 先例的"良性分歧"类：同数学不同
// fp 分组，判据 = T2 连贯 + mtp3 两臂各自连贯 + 接受率持平 + T3 PPL 门。
// NINFER_Q5LAD_SPLIT4=1 进路由，默认关。
struct Q5Split4AddResidualEpilogue {
    template <bool SplitOutput, int SplitRow, int Tokens>
    __device__ __forceinline__ void
    operator()(__nv_bfloat16* out, __nv_bfloat16* out_tail, std::int32_t n, std::int32_t out_ld,
               std::int32_t row, const float (&values)[Tokens]) const {
        (void)out_tail;
        (void)n;
#pragma unroll
        for (int token = 0; token < Tokens; ++token) {
            const std::int64_t index = static_cast<std::int64_t>(token) * out_ld + row;
            out[index] = __float2bfloat16(values[token] + __bfloat162float(out[index]));
        }
    }
};

template <int Cols, int FullSlabs, int Stride>
void launch_split4_residual(const Tensor& x, const Weight& w, Tensor& residual_out,
                            cudaStream_t stream) {
    constexpr int kThreads = 4 * 32;
    const std::int32_t out_ld =
        static_cast<std::int32_t>(residual_out.nb[1] / sizeof(__nv_bfloat16));
    const dim3 grid(static_cast<unsigned>(residual_out.ne[0]), 1u, 1u);
    const Q5Split4AddResidualEpilogue epilogue{};
    q5_rowsplit_gemm_simt_split4_kernel<Q5RowSplitSimtSchedule, Cols, FullSlabs, Stride, false, 0,
                                        Q5Split4AddResidualEpilogue, false, false, 1>
        <<<grid, kThreads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(w.qdata),
            static_cast<const std::uint8_t*>(w.qhigh), static_cast<const std::uint8_t*>(w.scales),
            static_cast<__nv_bfloat16*>(residual_out.data), nullptr, residual_out.ne[0], out_ld,
            x.ne[0], x.ne[1], w.padded_shape[1], FullSlabs, epilogue);
    CUDA_CHECK(cudaGetLastError());
}

// split8 残差变体（2026-09-29）：k 切分继续加深。k=17408 = 68 个 256-k chunk，8 warp/行
// 按 chunk ≡ part (mod 8) 认领 → 链长 8 或 9（余数不均尾损 ≤6%），寻址复用 split4 的
// chunk→(slab, chunk_in_slab) 分解。归约 = 8 部分和连加 + 残差，仍属良性分歧类。
// NINFER_Q5LAD_SPLIT8=1 进路由（优先于 SPLIT4），默认关。
constexpr int kSplit8Parts = 8;

template <int Cols, int FullSlabs, int Stride>
__global__ __launch_bounds__(8 * 32, 5) void q5_linear_add_split8_kernel(
    const __nv_bfloat16* __restrict__ x, const std::uint8_t* __restrict__ codes,
    const std::uint8_t* __restrict__ high, const std::uint8_t* __restrict__ scales,
    __nv_bfloat16* __restrict__ out, std::int32_t n, std::int32_t padded_k,
    std::int32_t total_chunks) {
    __shared__ float s_part[8][Cols];
    const int lane  = static_cast<int>(threadIdx.x) & 31;
    const int part  = static_cast<int>(threadIdx.x) >> 5;
    const int row   = static_cast<int>(blockIdx.x);
    if (row >= n) { return; }

    const int kg_padded = padded_k / Q5RowSplitStorage::kGroupK;
    const std::uint8_t* code_row = codes + static_cast<std::int64_t>(row) * kg_padded * 32;
    const std::uint8_t* high_row = high + static_cast<std::int64_t>(row) * kg_padded *
                                              Q5RowSplitSimtSchedule::kHighBytesPerGroup;
    const std::uint8_t* scale_row = scales + static_cast<std::int64_t>(row) * kg_padded * 2;
    const __half2 bias = __half2half2(__ushort_as_half(0x6410));  // 1040.0

    float acc[Cols];
#pragma unroll
    for (int tt = 0; tt < Cols; ++tt) { acc[tt] = 0.0f; }

#pragma unroll 1
    for (int c = part; c < total_chunks; c += kSplit8Parts) {
        const int s   = c >> 2;
        const int cis = c & 3;
        const std::uint32_t word = *reinterpret_cast<const std::uint32_t*>(
            code_row + static_cast<std::int64_t>(s) * 512 + cis * 128 + lane * 4);
        const std::uint32_t hc = static_cast<std::uint32_t>(
                                     high_row[static_cast<std::int64_t>(s) * 128 + cis * 32 +
                                              lane]) ^
                                 0xffu;
        const int group_in_slab = cis * 4 + (lane >> 3);
        const float scale       = __half2float(__ushort_as_half(*reinterpret_cast<const std::uint16_t*>(
            scale_row + (static_cast<std::int64_t>(s) * 16 + group_in_slab) * 2)));
        float w[8];
#pragma unroll
        for (int p = 0; p < 4; ++p) {
            std::uint32_t bits = ((word >> (4 * p)) & 0x000f000fu) | 0x64006400u;
            bits |= (((hc >> p) & 1u) << 4) | (((hc >> (p + 4)) & 1u) << 20);
            const __half2 h  = __hsub2(half2_from_bits(bits), bias);
            const float2 f   = __half22float2(h);
            w[p]             = f.x * scale;
            w[p + 4]         = f.y * scale;
        }

        const std::int64_t xoff =
            static_cast<std::int64_t>(s) * 1024 + cis * 256 + lane * 8;
#pragma unroll
        for (int tt = 0; tt < Cols; ++tt) {
            const uint4 xv  = load_vec<uint4>(x + static_cast<std::int64_t>(tt) * Stride + xoff);
            const float2 f0 = bf16x2_bits_to_float2(xv.x);
            const float2 f1 = bf16x2_bits_to_float2(xv.y);
            const float2 f2 = bf16x2_bits_to_float2(xv.z);
            const float2 f3 = bf16x2_bits_to_float2(xv.w);
            acc[tt]         = fmaf(w[0], f0.x, acc[tt]);
            acc[tt]         = fmaf(w[1], f0.y, acc[tt]);
            acc[tt]         = fmaf(w[2], f1.x, acc[tt]);
            acc[tt]         = fmaf(w[3], f1.y, acc[tt]);
            acc[tt]         = fmaf(w[4], f2.x, acc[tt]);
            acc[tt]         = fmaf(w[5], f2.y, acc[tt]);
            acc[tt]         = fmaf(w[6], f3.x, acc[tt]);
            acc[tt]         = fmaf(w[7], f3.y, acc[tt]);
        }
    }

#pragma unroll
    for (int tt = 0; tt < Cols; ++tt) {
        const float a = warp_reduce_sum(acc[tt]);
        if (lane == 0) { s_part[part][tt] = a; }
    }
    __syncthreads();
    if (part == 0 && lane < Cols) {
        float sum = 0.0f;
#pragma unroll
        for (int p = 0; p < 8; ++p) { sum += s_part[p][lane]; }
        const std::int64_t index = static_cast<std::int64_t>(lane) * n + row;
        out[index] = __float2bfloat16(sum + __bfloat162float(out[index]));
    }
}

template <int Cols, int FullSlabs, int Stride>
void launch_split8_residual(const Tensor& x, const Weight& w, Tensor& residual_out,
                            cudaStream_t stream) {
    constexpr int kThreads = 8 * 32;
    constexpr int kTotalChunks = FullSlabs * 4;  // 每 slab 4 个 256-k chunk
    const dim3 grid(static_cast<unsigned>(residual_out.ne[0]), 1u, 1u);
    q5_linear_add_split8_kernel<Cols, FullSlabs, Stride>
        <<<grid, kThreads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(w.qdata),
            static_cast<const std::uint8_t*>(w.qhigh), static_cast<const std::uint8_t*>(w.scales),
            static_cast<__nv_bfloat16*>(residual_out.data), residual_out.ne[0],
            w.padded_shape[1], kTotalChunks);
    CUDA_CHECK(cudaGetLastError());
}

int lad_split8_enabled() {
    static const int enabled = [] {
        const char* value = std::getenv("NINFER_Q5LAD_SPLIT8");
        return value != nullptr && std::atoi(value) != 0;
    }();
    return enabled;
}

int lad_split4_enabled() {
    // 已默认开（2026-09-29）：同二进制 C8 129.8→133.7（+3.0%，两轮复现）、MLP 段
    // −2.0 ms/步、T3 PPL 4.975734 逐位一致、mtp3 两臂各自连贯 + 接受率 47.13→46.50
    // （单跑噪声内）、C1 mtp0/mtp3 32.1/62.0 无回退；NINFER_Q5LAD_SPLIT4=0 回退 split2。
    static const int enabled = [] {
        const char* value = std::getenv("NINFER_Q5LAD_SPLIT4");
        return value == nullptr || std::atoi(value) != 0;
    }();
    return enabled;
}

template <int Cols, int FullSlabs, int Stride>
void launch_split2(const Tensor& x, const Weight& w, Tensor& residual_out, cudaStream_t stream) {
    constexpr int kThreads = 2 * 32;
    const dim3 grid(static_cast<unsigned>(residual_out.ne[0]), 1u, 1u);
    q5_rowsplit_gemm_simt_split2_kernel<Q5RowSplitSimtSchedule, Cols, FullSlabs, Stride, false, 0,
                                        true><<<grid, kThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(w.qdata),
        static_cast<const std::uint8_t*>(w.qhigh), static_cast<const std::uint8_t*>(w.scales),
        static_cast<__nv_bfloat16*>(residual_out.data), residual_out.ne[0], x.ne[0], x.ne[1],
        w.padded_shape[1], FullSlabs);
}

template <int Cols>
void dispatch_shape(const Tensor& x, const Weight& w, Tensor& residual_out, cudaStream_t stream) {
    if (w.k == 6144) {
        launch_split2<Cols, 6, 6144>(x, w, residual_out, stream);
    } else if (w.k == 17408) {
        launch_split2<Cols, 17, 17408>(x, w, residual_out, stream);
    } else {
        throw std::invalid_argument("q5 linear_add split2: unsupported exact K");
    }
}

template <class Launch>
void dispatch_cols(std::int32_t cols, Launch&& launch) {
    switch (cols) {
#define NINFER_Q5_LINEAR_ADD_EXACT(COLS)                                                           \
    case COLS:                                                                                     \
        launch.template operator()<COLS>();                                                        \
        return
        NINFER_Q5_LINEAR_ADD_EXACT(2);
        NINFER_Q5_LINEAR_ADD_EXACT(3);
        NINFER_Q5_LINEAR_ADD_EXACT(4);
        NINFER_Q5_LINEAR_ADD_EXACT(5);
        NINFER_Q5_LINEAR_ADD_EXACT(6);
        NINFER_Q5_LINEAR_ADD_EXACT(7);
        NINFER_Q5_LINEAR_ADD_EXACT(8);
        NINFER_Q5_LINEAR_ADD_EXACT(9);
        NINFER_Q5_LINEAR_ADD_EXACT(10);
        NINFER_Q5_LINEAR_ADD_EXACT(11);
        NINFER_Q5_LINEAR_ADD_EXACT(12);
        NINFER_Q5_LINEAR_ADD_EXACT(13);
        NINFER_Q5_LINEAR_ADD_EXACT(14);
        NINFER_Q5_LINEAR_ADD_EXACT(15);
        NINFER_Q5_LINEAR_ADD_EXACT(16);
#undef NINFER_Q5_LINEAR_ADD_EXACT
    default:
        throw std::invalid_argument("q5 linear_add split2: T must be in [2,16]");
    }
}

} // namespace

void q5_linear_add_split2_exact_launch(const Tensor& x, const Weight& w, Tensor& residual_out,
                                       cudaStream_t stream) {
    const int rb = lad_rb_rows();
    if (lad_split8_enabled()) {
        dispatch_cols(x.ne[1], [&]<int Cols>() {
            if (w.k == 6144) {
                launch_split8_residual<Cols, 6, 6144>(x, w, residual_out, stream);
            } else if (w.k == 17408) {
                launch_split8_residual<Cols, 17, 17408>(x, w, residual_out, stream);
            } else {
                throw std::invalid_argument("q5 linear_add split2: unsupported exact K");
            }
        });
        return;
    }
    if (lad_split4_enabled()) {
        dispatch_cols(x.ne[1], [&]<int Cols>() {
            if (w.k == 6144) {
                launch_split4_residual<Cols, 6, 6144>(x, w, residual_out, stream);
            } else if (w.k == 17408) {
                launch_split4_residual<Cols, 17, 17408>(x, w, residual_out, stream);
            } else {
                throw std::invalid_argument("q5 linear_add split2: unsupported exact K");
            }
        });
        return;
    }
    // Cols>8 时 RW=4 的 2×4×16=128 累加器必溢出，钳到 2。
    const int rw = (rb == 0) ? 0 : (x.ne[1] > 8 ? 2 : rb);
    if (rw != 0) {
        dispatch_cols(x.ne[1], [&]<int Cols>() {
            if (w.k == 6144) {
                if constexpr (Cols <= 8) {
                    if (rw == 4) {
                        launch_rb<Cols, 6, 6144, 4>(x, w, residual_out, stream);
                        return;
                    }
                }
                launch_rb<Cols, 6, 6144, 2>(x, w, residual_out, stream);
            } else if (w.k == 17408) {
                if constexpr (Cols <= 8) {
                    if (rw == 4) {
                        launch_rb<Cols, 17, 17408, 4>(x, w, residual_out, stream);
                        return;
                    }
                }
                launch_rb<Cols, 17, 17408, 2>(x, w, residual_out, stream);
            } else {
                throw std::invalid_argument("q5 linear_add split2: unsupported exact K");
            }
        });
        return;
    }
    dispatch_cols(x.ne[1], [&]<int Cols>() { dispatch_shape<Cols>(x, w, residual_out, stream); });
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail

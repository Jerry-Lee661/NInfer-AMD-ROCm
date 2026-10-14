// Probe: how fast is the emulated-mma q4 small_t kernel that the T>=2 path uses?
//
// The MTP verify pass and prefill run >=2 active columns and route through the small_t/mma
// family, which in this HIP build is built on mma_bf16 = the m16n8k16 fragment emulation
// (ops/common/mma.cuh:87). The T=1 path instead runs the GEMV family, measured at 613 GB/s
// in the engine. This probe measures the emulated kernel directly, with the production
// schedule (Q4DraftSmallTSchedule: 8 warps, 16 rows/CTA, kGroupK 512) and the production
// geometry (k=5120), so the replacement has a number to beat.
//
// Build: hipcc -O3 -std=c++17 --offload-arch=gfx1100 -DNINFER_HIP=1 \
//          -I compat/cuda -I include -I src -I third_party -o q4stmma tools/q4_small_t_mma_probe.cu
// Run:   HIP_VISIBLE_DEVICES=1 ./q4stmma [peak_gbps]

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "ops/linear/q4/q4_small_t_mma.cuh"

#include <cstdint>
#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK(expr)                                                                           \
    do {                                                                                           \
        const cudaError_t error__ = (expr);                                                        \
        if (error__ != cudaSuccess) {                                                              \
            std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,                  \
                         cudaGetErrorString(error__));                                             \
            std::exit(EXIT_FAILURE);                                                               \
        }                                                                                          \
    } while (false)

namespace {

constexpr int kRows = 131072;  // Q4DraftHeadGeometry::kOutputRows; codes = 335 MB, above L2
constexpr int kK    = 5120;

using Geometry = ninfer::ops::detail::Q4DraftHeadGeometry<kK>;
using Schedule = ninfer::ops::detail::Q4DraftSmallTSchedule;

constexpr std::size_t kCodeBytes  = static_cast<std::size_t>(kRows) * (kK / 2);
constexpr std::size_t kScaleBytes = static_cast<std::size_t>(kRows) * (kK / 64) * 2;

template <int TileCols, int ActiveCols>
void run(const char* label, const __nv_bfloat16* x, const std::uint8_t* codes,
         const std::uint8_t* scales, __nv_bfloat16* out, double peak_gbps) {
    static_assert(ActiveCols >= 2 && ActiveCols <= TileCols && ActiveCols > TileCols - 8);
    constexpr int kBlocks = Geometry::kOutputRows / Schedule::kRowsPerCta;
    cudaEvent_t start{};
    cudaEvent_t stop{};
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    double best = 0.0;
    for (int trial = 0; trial < 5; ++trial) {
        CUDA_CHECK(cudaEventRecord(start));
        ninfer::ops::detail::q4_small_t_mma_kernel<Geometry, TileCols, ActiveCols>
            <<<kBlocks, Schedule::kThreads>>>(x, codes, scales, out);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms = 0.0F;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        const double gbps = static_cast<double>(kCodeBytes) / (ms * 1.0e6);
        if (gbps > best) { best = gbps; }
    }
    std::printf("%-24s %8.1f GB/s  %5.1f%% of %.0f\n", label, best, 100.0 * best / peak_gbps,
                peak_gbps);
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
}

} // namespace

int main(int argc, char** argv) {
    const double peak_gbps = argc > 1 ? std::atof(argv[1]) : 960.0;

    __nv_bfloat16* x      = nullptr;
    std::uint8_t* codes   = nullptr;
    std::uint8_t* scales  = nullptr;
    __nv_bfloat16* out    = nullptr;
    CUDA_CHECK(cudaMalloc(&x, static_cast<std::size_t>(kK) * 8 * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&codes, kCodeBytes));
    CUDA_CHECK(cudaMalloc(&scales, kScaleBytes));
    CUDA_CHECK(cudaMalloc(&out, static_cast<std::size_t>(kRows) * 8 * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMemset(codes, 0x37, kCodeBytes));
    CUDA_CHECK(cudaMemset(scales, 0x01, kScaleBytes));
    CUDA_CHECK(cudaMemset(x, 0x11, static_cast<std::size_t>(kK) * 8 * sizeof(__nv_bfloat16)));

    std::printf("q4_small_t_mma (emulated m16n8k16): rows=%d k=%d, codes=%.0f MB\n", kRows, kK,
                static_cast<double>(kCodeBytes) / 1.0e6);
    run<8, 4>("TileCols=8 ActiveCols=4", x, codes, scales, out, peak_gbps);
    run<8, 8>("TileCols=8 ActiveCols=8", x, codes, scales, out, peak_gbps);
    run<16, 16>("TileCols=16 ActiveCols=16", x, codes, scales, out, peak_gbps);

    CUDA_CHECK(cudaFree(x));
    CUDA_CHECK(cudaFree(codes));
    CUDA_CHECK(cudaFree(scales));
    CUDA_CHECK(cudaFree(out));
    return 0;
}

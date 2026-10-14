// Probe: the q4 rowsplit_gemm_mma family that prefill (T>=17) and the mma tiers use.
//
// Same reasoning as q4_small_t_mma_probe.cu: this family is built on mma_bf16 = the
// m16n8k16 FMA fragment emulation in the HIP branch (ops/common/mma.cuh:87). Its ISA shows
// 17k-22k instructions per kernel with v_mov_b32 + v_cmp_ne_u32 taking ~34% -- the emulation's
// fragment shuffling -- plus 255-760 scratch accesses. Prefill's 75 ms per prompt token lands
// here, so the replacement needs this number as its floor.
//
// Build: hipcc -O3 -std=c++17 --offload-arch=gfx1100 -DNINFER_HIP=1 \
//          -I compat/cuda -I include -I src -I third_party -o q4mma tools/q4_mma_family_probe.cu
// Run:   HIP_VISIBLE_DEVICES=1 ./q4mma [peak_gbps]

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "ops/linear/q4/q4_rowsplit_gemm_mma.cuh"

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

constexpr int kRows = 131072;  // any multiple of kBlockRows; codes = 335 MB, above L2
constexpr int kK    = 5120;

using Schedule = ninfer::ops::detail::Q4RowSplitMmaGemmSchedule<
    64, 128, 64, 64, 32, 2, 1, ninfer::ops::detail::Q4FragmentPipeline::Serial,
    ninfer::ops::Cache::cg, ninfer::ops::Cache::cg,
    ninfer::ops::detail::Q4ScaleLoad::Pair32>;

constexpr std::size_t kCodeBytes  = static_cast<std::size_t>(kRows) * (kK / 2);
constexpr std::size_t kScaleBytes = static_cast<std::size_t>(kRows) * (kK / 64) * 2;

template <int Cols, bool Full>
void run(const char* label, const __nv_bfloat16* x, const std::uint8_t* codes,
         const std::uint8_t* scales, __nv_bfloat16* out, double peak_gbps) {
    const dim3 grid(static_cast<unsigned>(kRows / Schedule::kBlockRows),
                    static_cast<unsigned>((Cols + Schedule::kBlockCols - 1) / Schedule::kBlockCols),
                    1u);
    cudaEvent_t start{};
    cudaEvent_t stop{};
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    double best = 0.0;
    for (int trial = 0; trial < 5; ++trial) {
        CUDA_CHECK(cudaEventRecord(start));
        ninfer::ops::detail::q4_rowsplit_gemm_mma_kernel<Schedule, Full>
            <<<grid, Schedule::kThreads>>>(x, codes, scales, out, kRows, kK, Cols, kK);
        CUDA_CHECK(cudaGetLastError());  // a failed launch would otherwise time as ~0 ms
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms = 0.0F;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        const double gbps = static_cast<double>(kCodeBytes) / (ms * 1.0e6);
        if (gbps > best) { best = gbps; }
    }
    std::printf("%-28s %8.1f GB/s  %5.1f%% of %.0f\n", label, best, 100.0 * best / peak_gbps,
                peak_gbps);
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
}

// Self-check anchor: stream the same buffer with plain uint4 loads. If the family above reads
// far slower than this, the kernel is the problem; if this is also slow, the probe or the
// machine is (the repo's rule: prove the judge before judging).
__global__ void stream_reference_kernel(const uint4* __restrict__ src, std::size_t n4,
                                        uint4* sink) {
    std::size_t index = (static_cast<std::size_t>(blockIdx.x) * blockDim.x) + threadIdx.x;
    const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    uint4 acc{0, 0, 0, 0};
    for (; index < n4; index += stride) {
        const uint4 v = src[index];
        acc.x ^= v.x;
        acc.y ^= v.y;
        acc.z ^= v.z;
        acc.w ^= v.w;
    }
    if (acc.x == 0xDEADBEEFu) { sink[0] = acc; }
}

void run_reference(const void* codes, double bytes, double peak_gbps) {
    cudaEvent_t start{};
    cudaEvent_t stop{};
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    uint4* sink = nullptr;
    CUDA_CHECK(cudaMalloc(&sink, 16));
    double best = 0.0;
    for (int trial = 0; trial < 5; ++trial) {
        CUDA_CHECK(cudaEventRecord(start));
        stream_reference_kernel<<<2048, 256>>>(reinterpret_cast<const uint4*>(codes),
                                              static_cast<std::size_t>(bytes) / 16, sink);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms = 0.0F;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        const double gbps = bytes / (ms * 1.0e6);
        if (gbps > best) { best = gbps; }
    }
    std::printf("%-28s %8.1f GB/s  %5.1f%% of %.0f\n", "stream ref (same buffer)", best,
                100.0 * best / peak_gbps, peak_gbps);
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(sink));
}

} // namespace

int main(int argc, char** argv) {
    const double peak_gbps = argc > 1 ? std::atof(argv[1]) : 960.0;

    __nv_bfloat16* x      = nullptr;
    std::uint8_t* codes   = nullptr;
    std::uint8_t* scales  = nullptr;
    __nv_bfloat16* out    = nullptr;
    CUDA_CHECK(cudaMalloc(&x, static_cast<std::size_t>(kK) * 128 * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&codes, kCodeBytes));
    CUDA_CHECK(cudaMalloc(&scales, kScaleBytes));
    CUDA_CHECK(cudaMalloc(&out, static_cast<std::size_t>(kRows) * 128 * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMemset(codes, 0x37, kCodeBytes));
    CUDA_CHECK(cudaMemset(scales, 0x01, kScaleBytes));
    CUDA_CHECK(cudaMemset(x, 0x11, static_cast<std::size_t>(kK) * 128 * sizeof(__nv_bfloat16)));

    std::printf("q4_rowsplit_gemm_mma (emulated m16n8k16): rows=%d k=%d, codes=%.0f MB, %d threads\n",
                kRows, kK, static_cast<double>(kCodeBytes) / 1.0e6, Schedule::kThreads);
    run<128, true>("BlockCols=128 T=128 (prefill)", x, codes, scales, out, peak_gbps);
    run<16, false>("T=16 (mma tier)", x, codes, scales, out, peak_gbps);
    run<4, false>("T=4 (verify shape)", x, codes, scales, out, peak_gbps);
    run_reference(codes, static_cast<double>(kCodeBytes), peak_gbps);

    CUDA_CHECK(cudaFree(x));
    CUDA_CHECK(cudaFree(codes));
    CUDA_CHECK(cudaFree(scales));
    CUDA_CHECK(cudaFree(out));
    return 0;
}

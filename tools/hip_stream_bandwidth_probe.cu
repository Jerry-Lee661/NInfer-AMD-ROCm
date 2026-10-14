// HIP sustained-read bandwidth probe for the gfx1100 port (tools/ sibling of
// hbm_bandwidth_probe.cu, which stays nvcc-only).
//
// Why a second probe: the q4/q5 decode kernels stream ~15 GB/token through thousands of
// concurrent short streams (one warp per output row, 512 B staged per pipeline stage).
// The question this answers is whether the card can reach its advertised bandwidth at all,
// and how much of it the many-short-streams shape costs.
//
//   read_shared   all warps grid-stride over one array  -> one long stream (DRAM-friendly)
//   read_private  each warp owns a private sequential stream, 512 B per step
//                 (mimics the q4 SIMT stage: 32 lanes x 16 B, then advance)
//
// Build (X99): hipcc -O3 -std=c++17 --offload-arch=gfx1100 -o hbw tools/hip_stream_bandwidth_probe.cu
// Run:         HIP_VISIBLE_DEVICES=1 ./hbw [peak_gbps]

#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define HIP_CHECK(expr)                                                                            \
    do {                                                                                           \
        const hipError_t error__ = (expr);                                                         \
        if (error__ != hipSuccess) {                                                               \
            std::fprintf(stderr, "HIP error at %s:%d: %s\n", __FILE__, __LINE__,                   \
                         hipGetErrorString(error__));                                              \
            std::exit(EXIT_FAILURE);                                                               \
        }                                                                                          \
    } while (false)

namespace {

constexpr int kThreads    = 256;
constexpr int kCtasPerCu  = 8;
constexpr std::size_t kBytes = std::size_t{4} << 30;  // 4 GiB working set
constexpr int kTrials     = 5;

__global__ void read_shared_kernel(const uint4* __restrict__ src, std::size_t n4, uint4* sink) {
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

// Each warp owns one private stream and advances 512 B (32 lanes x 16 B) per step, so the
// card sees thousands of concurrent sequential streams instead of one.
__global__ void read_private_kernel(const uint4* __restrict__ src, std::size_t steps_per_warp,
                                    uint4* sink) {
    const std::size_t warp_id = (static_cast<std::size_t>(blockIdx.x) * (blockDim.x / 32)) +
                                (threadIdx.x / 32);
    const int lane  = static_cast<int>(threadIdx.x) & 31;
    const std::size_t base = warp_id * steps_per_warp * 32;
    uint4 acc{0, 0, 0, 0};
    for (std::size_t step = 0; step < steps_per_warp; ++step) {
        const uint4 v = src[base + step * 32 + lane];
        acc.x ^= v.x;
        acc.y ^= v.y;
        acc.z ^= v.z;
        acc.w ^= v.w;
    }
    if (acc.x == 0xDEADBEEFu) { sink[0] = acc; }
}

double time_ms(hipEvent_t start, hipEvent_t stop) {
    float ms = 0.0F;
    HIP_CHECK(hipEventElapsedTime(&ms, start, stop));
    return static_cast<double>(ms);
}

template <class Launch>
double measure(const char* label, Launch launch, std::size_t bytes, double peak_gbps) {
    hipEvent_t start{};
    hipEvent_t stop{};
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));
    double best = 0.0;
    for (int trial = 0; trial < kTrials; ++trial) {
        HIP_CHECK(hipEventRecord(start));
        launch();
        HIP_CHECK(hipEventRecord(stop));
        HIP_CHECK(hipEventSynchronize(stop));
        const double ms = time_ms(start, stop);
        const double gbps = static_cast<double>(bytes) / (ms * 1.0e6);
        if (gbps > best) { best = gbps; }
    }
    std::printf("%-14s %8.1f GB/s  %5.1f%% of %.0f\n", label, best, 100.0 * best / peak_gbps,
                peak_gbps);
    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));
    return best;
}

} // namespace

int main(int argc, char** argv) {
    const double peak_gbps = argc > 1 ? std::atof(argv[1]) : 960.0;

    hipDeviceProp_t prop{};
    HIP_CHECK(hipGetDeviceProperties(&prop, 0));
    std::printf("device: %s  CUs=%d  clock=%.2f GHz\n", prop.name, prop.multiProcessorCount,
                prop.clockRate / 1.0e6);

    const std::size_t n4 = kBytes / sizeof(uint4);
    uint4* src           = nullptr;
    uint4* sink          = nullptr;
    HIP_CHECK(hipMalloc(&src, kBytes));
    HIP_CHECK(hipMalloc(&sink, 4096));
    HIP_CHECK(hipMemset(src, 0x5A, kBytes));

    const int cta_grid = prop.multiProcessorCount * kCtasPerCu;
    measure("read_shared", [&] { read_shared_kernel<<<cta_grid, kThreads>>>(src, n4, sink); },
            kBytes, peak_gbps);

    const std::size_t warps        = static_cast<std::size_t>(cta_grid) * (kThreads / 32);
    const std::size_t steps_per_warp = (n4 / 32) / warps;
    const std::size_t private_bytes  = warps * steps_per_warp * 32 * sizeof(uint4);
    measure("read_private",
            [&] { read_private_kernel<<<cta_grid, kThreads>>>(src, steps_per_warp, sink); },
            private_bytes, peak_gbps);

    HIP_CHECK(hipFree(src));
    HIP_CHECK(hipFree(sink));
    return 0;
}

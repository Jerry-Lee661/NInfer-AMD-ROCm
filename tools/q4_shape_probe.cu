// Microbenchmark: what is the practical ceiling for the q4 decode GEMV shape?
//
// The engine's q4/q5 SIMT kernels stream ~14.5 GB/token at ~551 GB/s (63% of the 869 GB/s
// the plain streaming probe reaches). Three candidate causes, isolated here by running the
// same shape with pieces removed:
//
//   lds      : the production shape -- codes staged in LDS, decoded with the half2 trick,
//              4 columns of FP32 FMA                     (q4_rowsplit_gemm_simt r8_c4)
//   direct   : same loop, codes read straight from global into registers (no LDS) -- the
//              shape q5 split2/split4 already use
//   nodecode : same as direct but the per-weight decode is replaced by a reinterpret, so the
//              FMA count is unchanged and only the decode instructions disappear
//
// Build (X99): hipcc -O3 -std=c++17 --offload-arch=gfx1100 -DNINFER_HIP=1 \
//                    -I compat/cuda -I include -I src -o q4shape tools/q4_shape_probe.cu
// Run:         HIP_VISIBLE_DEVICES=1 ./q4shape [peak_gbps]

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "ops/linear/q4/q4_rowsplit_gemm_simt.cuh"

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

constexpr int kRows        = 139264;  // 4x the gate_up row count, so the codes exceed L2
constexpr int kK           = 5120;
constexpr int kCols        = 4;       // decode runs width=4
constexpr int kWarpsPerCta = 8;
constexpr int kThreads     = kWarpsPerCta * 32;
constexpr int kCodeBytes   = kK / 2;  // q4: two weights per byte
constexpr int kPhases      = kK / 256;
constexpr int kStages      = 2;

__device__ __forceinline__ void q4_decode_eight(std::uint32_t packed, std::uint16_t scale_bits,
                                                float (&weights)[8]) {
    const std::uint32_t word = packed ^ 0x88888888u;
    const float scale        = __half2float(__ushort_as_half(scale_bits));
    const __half2 bias       = __half2half2(__ushort_as_half(0x6408));  // 1032.0
#pragma unroll
    for (int pair = 0; pair < 4; ++pair) {
        const std::uint32_t bits = ((word >> (4 * pair)) & 0x000f000fu) | 0x64006400u;
        const __half2 decoded    = __hsub2(__half2half2(__ushort_as_half(bits)), bias);
        const float2 values      = __half22float2(decoded);
        weights[pair]            = values.x * scale;
        weights[pair + 4]        = values.y * scale;
    }
}

__device__ __forceinline__ void q4_accumulate(std::uint32_t packed, std::uint16_t scale_bits,
                                              const __nv_bfloat16* __restrict__ x, std::int64_t xk,
                                              float (&acc)[kCols]) {
    float weights[8];
    q4_decode_eight(packed, scale_bits, weights);
#pragma unroll
    for (int col = 0; col < kCols; ++col) {
        const uint4 values =
            *reinterpret_cast<const uint4*>(x + static_cast<std::int64_t>(col) * kK + xk);
        const float2 x0 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&values.x));
        const float2 x1 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&values.y));
        const float2 x2 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&values.z));
        const float2 x3 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&values.w));
        acc[col]        = fmaf(weights[0], x0.x, acc[col]);
        acc[col]        = fmaf(weights[1], x0.y, acc[col]);
        acc[col]        = fmaf(weights[2], x1.x, acc[col]);
        acc[col]        = fmaf(weights[3], x1.y, acc[col]);
        acc[col]        = fmaf(weights[4], x2.x, acc[col]);
        acc[col]        = fmaf(weights[5], x2.y, acc[col]);
        acc[col]        = fmaf(weights[6], x3.x, acc[col]);
        acc[col]        = fmaf(weights[7], x3.y, acc[col]);
    }
}

__device__ __forceinline__ void q4_store(const __nv_bfloat16* __restrict__ /*x*/, int row, int lane,
                                         const float (&acc)[kCols],
                                         __nv_bfloat16* __restrict__ out) {
    if (lane == 0) {
#pragma unroll
        for (int col = 0; col < kCols; ++col) {
            out[static_cast<std::int64_t>(col) * kRows + row] = __float2bfloat16(acc[col]);
        }
    }
}

// Production shape: codes staged through LDS.
__global__ void q4_lds_kernel(const __nv_bfloat16* __restrict__ x,
                              const std::uint8_t* __restrict__ codes,
                              const std::uint16_t* __restrict__ scales,
                              __nv_bfloat16* __restrict__ out) {
    __shared__ __align__(16) uint4 s_codes[kWarpsPerCta][kStages][kPhases];
    __shared__ __align__(16) std::uint16_t s_scales[kWarpsPerCta][kStages][4];

    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int row  = static_cast<int>(blockIdx.x) * kWarpsPerCta + warp;
    if (row >= kRows) { return; }

    const std::uint8_t* code_row   = codes + static_cast<std::int64_t>(row) * kCodeBytes;
    const std::uint16_t* scale_row = scales + static_cast<std::int64_t>(row) * (kK / 64);
    float acc[kCols];
#pragma unroll
    for (int col = 0; col < kCols; ++col) { acc[col] = 0.0f; }

#pragma unroll 1
    for (int phase = 0; phase < kPhases; ++phase) {
        const int buf = phase % kStages;
        reinterpret_cast<std::uint32_t*>(s_codes[warp][buf])[lane] =
            *reinterpret_cast<const std::uint32_t*>(code_row + phase * 128 + lane * 4);
        if (lane < 4) { s_scales[warp][buf][lane] = scale_row[phase * 4 + lane]; }
        __syncwarp();
        const std::uint32_t packed =
            reinterpret_cast<const std::uint32_t*>(s_codes[warp][buf])[lane];
        const std::uint16_t sbits = s_scales[warp][buf][lane >> 3];
        q4_accumulate(packed, sbits, x, static_cast<std::int64_t>(phase) * 256 + lane * 8, acc);
        __syncwarp();
    }
    q4_store(x, row, lane, acc, out);
}

// Same loop, codes read directly from global (no LDS, no pipeline).
__global__ void q4_direct_kernel(const __nv_bfloat16* __restrict__ x,
                                 const std::uint8_t* __restrict__ codes,
                                 const std::uint16_t* __restrict__ scales,
                                 __nv_bfloat16* __restrict__ out) {
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int row  = static_cast<int>(blockIdx.x) * kWarpsPerCta + warp;
    if (row >= kRows) { return; }

    const std::uint8_t* code_row   = codes + static_cast<std::int64_t>(row) * kCodeBytes;
    const std::uint16_t* scale_row = scales + static_cast<std::int64_t>(row) * (kK / 64);
    float acc[kCols];
#pragma unroll
    for (int col = 0; col < kCols; ++col) { acc[col] = 0.0f; }

#pragma unroll 1
    for (int phase = 0; phase < kPhases; ++phase) {
        const std::uint32_t packed =
            *reinterpret_cast<const std::uint32_t*>(code_row + phase * 128 + lane * 4);
        const std::uint16_t sbits = scale_row[phase * 4 + (lane >> 3)];
        q4_accumulate(packed, sbits, x, static_cast<std::int64_t>(phase) * 256 + lane * 8, acc);
    }
    q4_store(x, row, lane, acc, out);
}

// Direct loads, decode replaced by a reinterpret: same FMA count, no decode instructions.
__global__ void q4_nodecode_kernel(const __nv_bfloat16* __restrict__ x,
                                   const std::uint8_t* __restrict__ codes,
                                   const std::uint16_t* __restrict__ /*scales*/,
                                   __nv_bfloat16* __restrict__ out) {
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int row  = static_cast<int>(blockIdx.x) * kWarpsPerCta + warp;
    if (row >= kRows) { return; }

    const std::uint8_t* code_row = codes + static_cast<std::int64_t>(row) * kCodeBytes;
    float acc[kCols];
#pragma unroll
    for (int col = 0; col < kCols; ++col) { acc[col] = 0.0f; }

#pragma unroll 1
    for (int phase = 0; phase < kPhases; ++phase) {
        const std::uint32_t packed =
            *reinterpret_cast<const std::uint32_t*>(code_row + phase * 128 + lane * 4);
        const float w = __uint_as_float(packed);
#pragma unroll
        for (int col = 0; col < kCols; ++col) {
            const uint4 values = *reinterpret_cast<const uint4*>(
                x + static_cast<std::int64_t>(col) * kK + static_cast<std::int64_t>(phase) * 256 +
                lane * 8);
            const float2 x0 =
                __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&values.x));
            const float2 x1 =
                __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&values.y));
            const float2 x2 =
                __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&values.z));
            const float2 x3 =
                __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&values.w));
            acc[col] = fmaf(w, x0.x, acc[col]);
            acc[col] = fmaf(w, x0.y, acc[col]);
            acc[col] = fmaf(w, x1.x, acc[col]);
            acc[col] = fmaf(w, x1.y, acc[col]);
            acc[col] = fmaf(w, x2.x, acc[col]);
            acc[col] = fmaf(w, x2.y, acc[col]);
            acc[col] = fmaf(w, x3.x, acc[col]);
            acc[col] = fmaf(w, x3.y, acc[col]);
        }
    }
    q4_store(x, row, lane, acc, out);
}

// Direct loads with the phase loop unrolled N ways, to expose more memory parallelism.
template <int Unroll>
__global__ void q4_direct_unroll_kernel(const __nv_bfloat16* __restrict__ x,
                                        const std::uint8_t* __restrict__ codes,
                                        const std::uint16_t* __restrict__ scales,
                                        __nv_bfloat16* __restrict__ out) {
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int row  = static_cast<int>(blockIdx.x) * kWarpsPerCta + warp;
    if (row >= kRows) { return; }

    const std::uint8_t* code_row   = codes + static_cast<std::int64_t>(row) * kCodeBytes;
    const std::uint16_t* scale_row = scales + static_cast<std::int64_t>(row) * (kK / 64);
    float acc[kCols];
#pragma unroll
    for (int col = 0; col < kCols; ++col) { acc[col] = 0.0f; }

#pragma unroll Unroll
    for (int phase = 0; phase < kPhases; ++phase) {
        const std::uint32_t packed =
            *reinterpret_cast<const std::uint32_t*>(code_row + phase * 128 + lane * 4);
        const std::uint16_t sbits = scale_row[phase * 4 + (lane >> 3)];
        q4_accumulate(packed, sbits, x, static_cast<std::int64_t>(phase) * 256 + lane * 8, acc);
    }
    q4_store(x, row, lane, acc, out);
}

using Kernel = void (*)(const __nv_bfloat16*, const std::uint8_t*, const std::uint16_t*,
                        __nv_bfloat16*);

// The engine's own kernel, at the production schedule (q4 SIMT r8_c4 = the gate_up route).
using EngineSchedule =
    ninfer::ops::detail::Q4RowSplitSimtGemmSchedule<8, 4, 16, 2, ninfer::ops::Cache::ca, 1>;

template <class Launch>
void run(const char* label, Launch launch, const __nv_bfloat16* x, const std::uint8_t* codes,
         const std::uint16_t* scales, __nv_bfloat16* out, double peak_gbps) {
    cudaEvent_t start{};
    cudaEvent_t stop{};
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    double best = 0.0;
    for (int trial = 0; trial < 5; ++trial) {
        CUDA_CHECK(cudaEventRecord(start));
        launch();
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms = 0.0F;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        const double gbps = static_cast<double>(kRows) * kCodeBytes / (ms * 1.0e6);
        if (gbps > best) { best = gbps; }
    }
    std::printf("%-10s %8.1f GB/s  %5.1f%% of %.0f\n", label, best, 100.0 * best / peak_gbps,
                peak_gbps);
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
}

void run_kernel(const char* label, Kernel kernel, const __nv_bfloat16* x,
                const std::uint8_t* codes, const std::uint16_t* scales, __nv_bfloat16* out,
                double peak_gbps) {
    const int grid = (kRows + kWarpsPerCta - 1) / kWarpsPerCta;
    run(label, [&] { kernel<<<grid, kThreads>>>(x, codes, scales, out); }, x, codes, scales, out,
        peak_gbps);
}

void run(const char* label, Kernel kernel, const __nv_bfloat16* x, const std::uint8_t* codes,
         const std::uint16_t* scales, __nv_bfloat16* out, double peak_gbps) {
    const int grid = (kRows + kWarpsPerCta - 1) / kWarpsPerCta;
    cudaEvent_t start{};
    cudaEvent_t stop{};
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    double best = 0.0;
    for (int trial = 0; trial < 5; ++trial) {
        CUDA_CHECK(cudaEventRecord(start));
        kernel<<<grid, kThreads>>>(x, codes, scales, out);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms = 0.0F;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        const double gbps = static_cast<double>(kRows) * kCodeBytes / (ms * 1.0e6);
        if (gbps > best) { best = gbps; }
    }
    std::printf("%-10s %8.1f GB/s  %5.1f%% of %.0f\n", label, best, 100.0 * best / peak_gbps,
                peak_gbps);
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
}

} // namespace

int main(int argc, char** argv) {
    const double peak_gbps = argc > 1 ? std::atof(argv[1]) : 960.0;

    __nv_bfloat16* x      = nullptr;
    std::uint8_t* codes   = nullptr;
    std::uint16_t* scales = nullptr;
    __nv_bfloat16* out    = nullptr;
    CUDA_CHECK(cudaMalloc(&x, static_cast<std::size_t>(kCols) * kK * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&codes, static_cast<std::size_t>(kRows) * kCodeBytes));
    CUDA_CHECK(
        cudaMalloc(&scales, static_cast<std::size_t>(kRows) * (kK / 64) * sizeof(std::uint16_t)));
    CUDA_CHECK(cudaMalloc(&out, static_cast<std::size_t>(kCols) * kRows * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMemset(codes, 0x37, static_cast<std::size_t>(kRows) * kCodeBytes));
    CUDA_CHECK(cudaMemset(scales, 0x01, static_cast<std::size_t>(kRows) * (kK / 64) * 2));
    CUDA_CHECK(cudaMemset(x, 0x11, static_cast<std::size_t>(kCols) * kK * 2));

    std::printf("shape: rows=%d k=%d cols=%d  codes=%.0f MB\n", kRows, kK, kCols,
                static_cast<double>(kRows) * kCodeBytes / 1.0e6);
    run_kernel("lds", q4_lds_kernel, x, codes, scales, out, peak_gbps);
    run("engine",
        [&] {
            constexpr int kRowsPerCta = EngineSchedule::kRowsPerCta;
            const int grid            = (kRows + kRowsPerCta - 1) / kRowsPerCta;
            ninfer::ops::detail::q4_rowsplit_gemm_simt_kernel<EngineSchedule, true>
                <<<grid, EngineSchedule::kThreads>>>(
                    x, codes, reinterpret_cast<const std::uint8_t*>(scales), out, nullptr, kRows, 0,
                    kRows, kK, kCols, kK);
        },
        x, codes, scales, out, peak_gbps);
    run_kernel("direct", q4_direct_kernel, x, codes, scales, out, peak_gbps);
    run_kernel("direct_u2", q4_direct_unroll_kernel<2>, x, codes, scales, out, peak_gbps);
    run_kernel("direct_u4", q4_direct_unroll_kernel<4>, x, codes, scales, out, peak_gbps);
    run_kernel("direct_u8", q4_direct_unroll_kernel<8>, x, codes, scales, out, peak_gbps);
    run_kernel("nodecode", q4_nodecode_kernel, x, codes, scales, out, peak_gbps);

    CUDA_CHECK(cudaFree(x));
    CUDA_CHECK(cudaFree(codes));
    CUDA_CHECK(cudaFree(scales));
    CUDA_CHECK(cudaFree(out));
    return 0;
}

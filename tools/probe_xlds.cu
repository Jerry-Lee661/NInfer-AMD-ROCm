// P-D b 线探针(2026-10-05): T=1 PQ2 GEMV 的 x 装载策略——全局直读(现行) vs
// CTA 内 LDS 共享(gfx906 pass2 手段的 gfx1100 版)。
//
// 现行 ternary_pq2_gemv_kernel: 8 warp/CTA 各自读同一 x(5120 bf16 = 10KB),
// 每 warp 每 group 8×4=32B ×2(bf16 对)。x 全程在 L1, 但 8 warp 的重复读
// 占用 L1 带宽与发射槽。LDS 版: CTA 启动先把 x 装进 shared(10KB), 各 warp
// 从 LDS 读(gfx1100 LDS 带宽 >> L1)。
//
// 判据: 数值 ≤1e-6(纯装载路径差, 算术同); 时序 [65536,5120] 大形状。
// 编译(X99): hipcc --offload-arch=gfx1100 probe_xlds.cu -o /tmp/probe_xlds

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#define HIP_CHECK(expr)                                                          \
    do {                                                                         \
        hipError_t err__ = (expr);                                               \
        if (err__ != hipSuccess) {                                               \
            std::printf("HIP error %s at %s:%d\n", hipGetErrorString(err__),     \
                        __FILE__, __LINE__);                                     \
            std::exit(1);                                                        \
        }                                                                        \
    } while (0)

namespace {

constexpr int kGroupK        = 128;
constexpr int kCodeBytes     = 32;
constexpr int kScaleBytes    = 2;
constexpr int kWarpsPerBlock = 8;

// bf16 裸位表示(HIP 7.1 独立编译时无 __nv_bfloat16)。
__device__ __forceinline__ float bf2f(std::uint32_t bits) {
    union { std::uint32_t int32; float fp32; } u = {bits << 16};
    return u.fp32;
}

__device__ __forceinline__ float gemv_scale(const std::uint8_t* p) {
    const std::uint16_t bits = *reinterpret_cast<const std::uint16_t*>(p);
    return __half2float(__ushort_as_half(bits));
}

// 臂 A: 现行(x 全局直读; 与引擎 ternary_pq2_gemv_kernel 语句级同构)。
__global__ __launch_bounds__(kWarpsPerBlock * 32)
void gemv_global(const std::uint16_t* __restrict__ x,   // bf16 位型
                 const std::uint8_t* __restrict__ codes,
                 const std::uint8_t* __restrict__ scales, float* __restrict__ out, int rows,
                 int groups_per_row) {
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp =
        static_cast<int>(blockIdx.x) * kWarpsPerBlock + (static_cast<int>(threadIdx.x) >> 5);
    if (warp >= rows) { return; }
    const std::uint8_t* code_row =
        codes + static_cast<std::int64_t>(warp) * groups_per_row * kCodeBytes;
    const std::uint8_t* scale_row =
        scales + static_cast<std::int64_t>(warp) * groups_per_row * kScaleBytes;
    float acc = 0.0f;
#pragma unroll 8
    for (int group = 0; group < groups_per_row; ++group) {
        const std::int32_t base = group * kGroupK + lane * 4;
        const std::uint32_t lo_bits = *reinterpret_cast<const std::uint32_t*>(x + base);
        const std::uint32_t hi_bits = *reinterpret_cast<const std::uint32_t*>(x + base + 2);
        const float2 low  = make_float2(bf2f(lo_bits & 0xFFFFu), bf2f(lo_bits >> 16));
        const float2 high = make_float2(bf2f(hi_bits & 0xFFFFu), bf2f(hi_bits >> 16));
        const std::uint8_t raw = code_row[group * kCodeBytes + lane];
        const float dot = fmaf(static_cast<float>(static_cast<int>(raw & 3u) - 1), low.x,
                               fmaf(static_cast<float>(static_cast<int>((raw >> 2) & 3u) - 1),
                                    low.y,
                                    fmaf(static_cast<float>(static_cast<int>((raw >> 4) & 3u) - 1),
                                         high.x,
                                         static_cast<float>(static_cast<int>((raw >> 6) & 3u) - 1) *
                                             high.y)));
        acc = fmaf(gemv_scale(scale_row + group * kScaleBytes), dot, acc);
    }
    for (int off = 16; off > 0; off >>= 1) {
        acc += __shfl_xor_sync(0xffffffffffffffffULL, acc, off);
    }
    if (lane == 0) { out[warp] = acc; }
}

// 臂 B: CTA 级 LDS 共享(1024 线程 CTA = 32 warp, 对应引擎格式的 kGemvWarpsPerBlock 升档)。
// shared x: 5120 bf16 = 10KB; CTA 启动先全量装, 之后各 warp 从 LDS 读。
__global__ __launch_bounds__(1024)
void gemv_lds(const std::uint16_t* __restrict__ x,   // bf16 位型
              const std::uint8_t* __restrict__ codes,
              const std::uint8_t* __restrict__ scales, float* __restrict__ out, int rows,
              int groups_per_row, int k) {
    __shared__ std::uint16_t xs[5120];   // bf16 位型
    const int tid = static_cast<int>(threadIdx.x);
    for (int i = tid; i < k; i += 1024) { xs[i] = x[i]; }
    __syncthreads();

    constexpr int kWarps = 32;
    const int lane = tid & 31;
    const int warp = static_cast<int>(blockIdx.x) * kWarps + (tid >> 5);
    if (warp >= rows) { return; }
    const std::uint8_t* code_row =
        codes + static_cast<std::int64_t>(warp) * groups_per_row * kCodeBytes;
    const std::uint8_t* scale_row =
        scales + static_cast<std::int64_t>(warp) * groups_per_row * kScaleBytes;
    float acc = 0.0f;
#pragma unroll 8
    for (int group = 0; group < groups_per_row; ++group) {
        const std::int32_t base = group * kGroupK + lane * 4;
        const std::uint32_t lo_bits = *reinterpret_cast<const std::uint32_t*>(xs + base);
        const std::uint32_t hi_bits = *reinterpret_cast<const std::uint32_t*>(xs + base + 2);
        const float2 low  = make_float2(bf2f(lo_bits & 0xFFFFu), bf2f(lo_bits >> 16));
        const float2 high = make_float2(bf2f(hi_bits & 0xFFFFu), bf2f(hi_bits >> 16));
        const std::uint8_t raw = code_row[group * kCodeBytes + lane];
        const float dot = fmaf(static_cast<float>(static_cast<int>(raw & 3u) - 1), low.x,
                               fmaf(static_cast<float>(static_cast<int>((raw >> 2) & 3u) - 1),
                                    low.y,
                                    fmaf(static_cast<float>(static_cast<int>((raw >> 4) & 3u) - 1),
                                         high.x,
                                         static_cast<float>(static_cast<int>((raw >> 6) & 3u) - 1) *
                                             high.y)));
        acc = fmaf(gemv_scale(scale_row + group * kScaleBytes), dot, acc);
    }
    for (int off = 16; off > 0; off >>= 1) {
        acc += __shfl_xor_sync(0xffffffffffffffffULL, acc, off);
    }
    if (lane == 0) { out[warp] = acc; }
}

}  // namespace

int main() {
    const int rows = 262144, k = 5120;
    const int groups = k / kGroupK;
    std::mt19937 rng(9);
    std::uniform_real_distribution<float> u(0.0f, 1.0f);
    std::vector<std::uint16_t> x(k);   // bf16 位型
    std::normal_distribution<float> gauss(0.0f, 1.0f);
    union { float fp32; std::uint32_t int32; } conv{};
    for (auto& v : x) {
        conv.fp32 = gauss(rng);
        if (~conv.int32 & 0x7f800000u) { conv.int32 += 0x7fffu + ((conv.int32 >> 16) & 1); }
        v = static_cast<std::uint16_t>(conv.int32 >> 16);
    }
    std::vector<std::uint8_t> codes(static_cast<std::size_t>(rows) * groups * kCodeBytes);
    std::vector<std::uint8_t> scales(static_cast<std::size_t>(rows) * groups * kScaleBytes);
    for (auto& c : codes) { c = static_cast<std::uint8_t>(rng() % 3); }
    for (std::size_t i = 0; i < scales.size(); i += 2) {
        const auto h = __float2half(0.5f + u(rng));
        const auto bits = *reinterpret_cast<const std::uint16_t*>(&h);
        scales[i] = static_cast<std::uint8_t>(bits & 0xFF);
        scales[i + 1] = static_cast<std::uint8_t>(bits >> 8);
    }

    std::uint16_t* dx;
    std::uint8_t *dc, *ds;
    float *da, *db;
    HIP_CHECK(hipMalloc(&dx, x.size() * 2));
    HIP_CHECK(hipMalloc(&dc, codes.size()));
    HIP_CHECK(hipMalloc(&ds, scales.size()));
    HIP_CHECK(hipMalloc(&da, static_cast<std::size_t>(rows) * 4));
    HIP_CHECK(hipMalloc(&db, static_cast<std::size_t>(rows) * 4));
    HIP_CHECK(hipMemcpy(dx, x.data(), x.size() * 2, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dc, codes.data(), codes.size(), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(ds, scales.data(), scales.size(), hipMemcpyHostToDevice));

    // 数值门: 行 0。
    {
        gemv_global<<<dim3((rows + kWarpsPerBlock - 1) / kWarpsPerBlock), kWarpsPerBlock * 32>>>(
            dx, dc, ds, da, rows, groups);
        gemv_lds<<<dim3((rows + 31) / 32), 1024>>>(dx, dc, ds, db, rows, groups, k);
        HIP_CHECK(hipDeviceSynchronize());
        float a = 0.0f, b = 0.0f;
        HIP_CHECK(hipMemcpy(&a, da, 4, hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemcpy(&b, db, 4, hipMemcpyDeviceToHost));
        std::printf("NUM row0 A=%.6f B=%.6f rel=%.2e\n", a, b,
                    std::fabs(a - b) / (std::fabs(a) + 1e-9));
    }

    hipEvent_t e0, e1;
    HIP_CHECK(hipEventCreate(&e0));
    HIP_CHECK(hipEventCreate(&e1));
    const int iters = 30;
    for (int w = 0; w < 3; ++w) {
        gemv_global<<<dim3((rows + kWarpsPerBlock - 1) / kWarpsPerBlock), kWarpsPerBlock * 32>>>(
            dx, dc, ds, da, rows, groups);
    }
    HIP_CHECK(hipEventRecord(e0));
    for (int i = 0; i < iters; ++i) {
        gemv_global<<<dim3((rows + kWarpsPerBlock - 1) / kWarpsPerBlock), kWarpsPerBlock * 32>>>(
            dx, dc, ds, da, rows, groups);
    }
    HIP_CHECK(hipEventRecord(e1));
    HIP_CHECK(hipEventSynchronize(e1));
    float ms_a = 0.0f;
    HIP_CHECK(hipEventElapsedTime(&ms_a, e0, e1));
    ms_a /= iters;

    for (int w = 0; w < 3; ++w) { gemv_lds<<<dim3((rows + 31) / 32), 1024>>>(dx, dc, ds, db, rows, groups, k); }
    HIP_CHECK(hipEventRecord(e0));
    for (int i = 0; i < iters; ++i) {
        gemv_lds<<<dim3((rows + 31) / 32), 1024>>>(dx, dc, ds, db, rows, groups, k);
    }
    HIP_CHECK(hipEventRecord(e1));
    HIP_CHECK(hipEventSynchronize(e1));
    float ms_b = 0.0f;
    HIP_CHECK(hipEventElapsedTime(&ms_b, e0, e1));
    ms_b /= iters;

    const double gcode = static_cast<double>(rows) * k * 34.0 / 128.0 / 1e9;
    std::printf("SHAPE [%d,%d] A(global): %.3f ms %.1f GB/s | B(lds): %.3f ms %.1f GB/s | B/A=%.2fx\n",
                rows, k, ms_a, gcode / (ms_a / 1e3), ms_b, gcode / (ms_b / 1e3), ms_a / ms_b);
    HIP_CHECK(hipFree(dx)); HIP_CHECK(hipFree(dc)); HIP_CHECK(hipFree(ds));
    HIP_CHECK(hipFree(da)); HIP_CHECK(hipFree(db));
    return 0;
}

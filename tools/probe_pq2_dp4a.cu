// P-A 探针(2026-10-03): PQ2_0 decode GEMV 两种配方同卡对拍 —— 我方现行
// (bf16 激活 + shift/and 解码 + fp32 fma 链, ternary_rowsplit_gemv.cuh 同款) vs
// PrismML mmvq 配方(q8_1 预量化激活 + v_perm 符号解码 + dp4a 整数点积,
// PrismML-Eng/llama.cpp vecdotq.cuh q2_0_symbols4_hip/vec_dot_q2_0_q8_1 同款)。
//
// 判据: 数值 = 对 host long double 参考的相对误差(≤1e-3); 性能 = 理论流量墙钟
// GB/s(权重 34B/128w + 激活流)。主形状 [262144,8192] 权重 572MB >> 96MB
// Infinity Cache, 保证 DRAM 墙; 副形状 [27648,5120] 为引擎 gate_up 真形状
// (37.5MB, L2 残留偏高, 只作参照)。
//
// 编译(X99): hipcc --offload-arch=gfx1100 probe_pq2_dp4a.cu -o /tmp/probe_dp4a
// 运行: HIP_VISIBLE_DEVICES=1 /tmp/probe_dp4a

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

constexpr int kGroupK        = 128;  // PQ2_0 组宽
constexpr int kCodeBytes     = 32;   // 每组码字节
constexpr int kScaleBytes    = 2;    // 每组 fp16 scale
constexpr int kWarpsPerBlock = 8;

std::uint16_t host_half_bits(float v) {
    const __half h = __float2half(v);
    return *reinterpret_cast<const std::uint16_t*>(&h);
}

// bf16 裸位表示(HIP 7.1 的 hip_bfloat16 API 换名,探针零依赖内建向量)。
std::uint16_t host_float_to_bf16(float f) {
    union { float fp32; std::uint32_t int32; } u = {f};
    if (~u.int32 & 0x7f800000u) {
        u.int32 += 0x7fffu + ((u.int32 >> 16) & 1);
    } else if (u.int32 & 0xffffu) {
        u.int32 |= 0x10000u;
    }
    return static_cast<std::uint16_t>(u.int32 >> 16);
}

float host_bf16_to_float(std::uint16_t bits) {
    union { std::uint32_t int32; float fp32; } u = {static_cast<std::uint32_t>(bits) << 16};
    return u.fp32;
}

__device__ __forceinline__ float bf2f(std::uint32_t bits) {
    union { std::uint32_t int32; float fp32; } u = {bits << 16};
    return u.fp32;
}

__device__ __forceinline__ float gemv_scale(const std::uint8_t* p) {
    const std::uint16_t bits = *reinterpret_cast<const std::uint16_t*>(p);
    return __half2float(__ushort_as_half(bits));
}

// ---- 配方 A: 我方现行(bf16 x + 手工解码 + fp32 fma) ----
__global__ __launch_bounds__(kWarpsPerBlock * 32)
void gemv_baseline(const std::uint16_t* __restrict__ x,   // bf16 位型
                   const std::uint8_t* __restrict__ codes,
                   const std::uint8_t* __restrict__ scales,
                   float* __restrict__ out, int rows, int groups_per_row) {
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp =
        static_cast<int>(blockIdx.x) * kWarpsPerBlock + (static_cast<int>(threadIdx.x) >> 5);
    if (warp >= rows) { return; }
    const std::uint8_t* code_row =
        codes + static_cast<std::int64_t>(warp) * groups_per_row * kCodeBytes;
    const std::uint8_t* scale_row =
        scales + static_cast<std::int64_t>(warp) * groups_per_row * kScaleBytes;

    float accumulator = 0.0f;
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
        accumulator = fmaf(gemv_scale(scale_row + group * kScaleBytes), dot, accumulator);
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        accumulator += __shfl_xor_sync(0xffffffffffffffffULL, accumulator, offset);
    }
    if (lane == 0) { out[warp] = accumulator; }
}

// ---- 配方 B: q8_1 激活 + v_perm 解码 + dp4a ----
// xq: 每组 4 个 32 子块, 每子块 int8×32 + fp16 ds8。lane 的 4 权重恰在一个码字节里。
__device__ __forceinline__ int symbols4_hip(unsigned int b) {
    const unsigned int y = (b & 0x33u) | ((b & 0xCCu) << 6);
    const unsigned int z = (y & 0x0303u) | ((y & 0x3030u) << 12);
    return static_cast<int>(__builtin_amdgcn_perm(0x020100FFu, 0x020100FFu, z));
}

__device__ __forceinline__ int dp4a(int a, int b, int c) {
    return __builtin_amdgcn_sudot4(true, a, true, b, c, false);
}

__global__ __launch_bounds__(kWarpsPerBlock * 32)
void gemv_dp4a(const std::int8_t* __restrict__ xq,
               const std::uint16_t* __restrict__ ds8,   // [groups*4] fp16
               const std::uint8_t* __restrict__ codes,
               const std::uint8_t* __restrict__ scales,
               float* __restrict__ out, int rows, int groups_per_row) {
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp =
        static_cast<int>(blockIdx.x) * kWarpsPerBlock + (static_cast<int>(threadIdx.x) >> 5);
    if (warp >= rows) { return; }
    const std::uint8_t* code_row =
        codes + static_cast<std::int64_t>(warp) * groups_per_row * kCodeBytes;
    const std::uint8_t* scale_row =
        scales + static_cast<std::int64_t>(warp) * groups_per_row * kScaleBytes;

    float accumulator = 0.0f;
#pragma unroll 8
    for (int group = 0; group < groups_per_row; ++group) {
        const std::uint8_t raw = code_row[group * kCodeBytes + lane];
        const int w4 = symbols4_hip(raw);
        const int x4 = *reinterpret_cast<const int*>(
            xq + static_cast<std::int64_t>(group) * kGroupK + lane * 4);
        const int dot = dp4a(w4, x4, 0);
        const float wsc =
            gemv_scale(scale_row + group * kScaleBytes) *
            __half2float(__ushort_as_half(ds8[group * 4 + (lane >> 3)]));
        accumulator = fmaf(wsc, static_cast<float>(dot), accumulator);
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        accumulator += __shfl_xor_sync(0xffffffffffffffffULL, accumulator, offset);
    }
    if (lane == 0) { out[warp] = accumulator; }
}

int symbol_cpu(int b, int i) {
    // 与 symbols4_hip 等价的 CPU 仿真: LUT {0xFF,0,1,2}[码], 结果字节 i = 第 i 个码的符号。
    const int y = (b & 0x33) | ((b & 0xCC) << 6);
    const int z = (y & 0x0303) | ((y & 0x3030) << 12);
    static const int lut[4] = {0xFF, 0x00, 0x01, 0x02};
    const int sel = (z >> (8 * i)) & 0xFF;
    return static_cast<std::int8_t>(lut[sel]);
}

void run_shape(int rows, int k, bool l2_note, unsigned seed) {
    const int groups = k / kGroupK;
    const std::int64_t total_groups = static_cast<std::int64_t>(rows) * groups;

    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> u(0.0f, 1.0f);
    std::normal_distribution<float> gauss(0.0f, 1.0f);

    std::vector<std::uint8_t> codes(static_cast<std::size_t>(total_groups) * kCodeBytes);
    std::vector<std::uint16_t> scales(static_cast<std::size_t>(total_groups));
    for (std::size_t i = 0; i < codes.size(); ++i) {
        codes[i] = static_cast<std::uint8_t>(rng() % 3);
    }
    for (std::size_t i = 0; i < scales.size(); ++i) {
        scales[i] = host_half_bits(0.5f + u(rng));
    }

    // 激活: bf16 N(0,1) + 其 q8_1 量化(int8 + 每 32 元素 fp16 ds)。
    std::vector<std::uint16_t> x(k);   // bf16 位型
    std::vector<std::int8_t> xq(k);
    std::vector<std::uint16_t> ds8(static_cast<std::size_t>(groups) * 4);
    std::vector<long double> xq_exact(k);
    for (int g = 0; g < groups * 4; ++g) {
        float amax = 0.0f;
        for (int i = 0; i < 32; ++i) { amax = std::max(amax, std::fabs(gauss(rng))); }
        const float d = amax / 127.0f;
        ds8[g] = host_half_bits(d);
        for (int i = 0; i < 32; ++i) {
            x[g * 32 + i] = host_float_to_bf16(gauss(rng));
            const float as_f = host_bf16_to_float(x[g * 32 + i]);
            const long q = std::lrint(as_f / d);
            xq[g * 32 + i] = static_cast<std::int8_t>(std::max(-127L, std::min(127L, q)));
            xq_exact[g * 32 + i] = static_cast<long double>(xq[g * 32 + i]) * d;
        }
    }

    std::uint8_t *dcodes, *dscales;
    std::uint16_t* dx;
    std::int8_t* dxq;
    std::uint16_t* dds8;
    float *da, *db;
    HIP_CHECK(hipMalloc(&dcodes, codes.size()));
    HIP_CHECK(hipMalloc(&dscales, scales.size() * 2));
    HIP_CHECK(hipMalloc(&dx, x.size() * 2));
    HIP_CHECK(hipMalloc(&dxq, xq.size()));
    HIP_CHECK(hipMalloc(&dds8, ds8.size() * 2));
    HIP_CHECK(hipMalloc(&da, static_cast<std::size_t>(rows) * 4));
    HIP_CHECK(hipMalloc(&db, static_cast<std::size_t>(rows) * 4));
    HIP_CHECK(hipMemcpy(dcodes, codes.data(), codes.size(), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dscales, scales.data(), scales.size() * 2, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dx, x.data(), x.size() * 2, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dxq, xq.data(), xq.size(), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dds8, ds8.data(), ds8.size() * 2, hipMemcpyHostToDevice));

    const int blocks = (rows + kWarpsPerBlock - 1) / kWarpsPerBlock;
    const dim3 grid(blocks), block(kWarpsPerBlock * 32);

    // 数值门: 行 0 与行 rows/2 对拍 CPU 参考(内核每行读同一 x 向量, 行间只有权重行不同)。
    for (const int row : {0, rows / 2}) {
        std::vector<int> wref(static_cast<std::size_t>(groups) * kGroupK);
        const std::int64_t gbase = static_cast<std::int64_t>(row) * groups * kCodeBytes;
        for (int g = 0; g < groups; ++g) {
            for (int lane = 0; lane < 32; ++lane) {
                const std::uint8_t raw = codes[static_cast<std::size_t>(gbase) +
                                               static_cast<std::size_t>(g) * kCodeBytes + lane];
                for (int i = 0; i < 4; ++i) {
                    wref[static_cast<std::size_t>(g) * kGroupK + lane * 4 + i] = symbol_cpu(raw, i);
                }
            }
        }
        long double ref_a = 0.0L, ref_b = 0.0L;
        for (int g = 0; g < groups; ++g) {
            const long double sc =
                static_cast<long double>(__half2float(
                    __ushort_as_half(scales[static_cast<std::size_t>(row) * groups + g])));
            long double pa = 0.0L, pb = 0.0L;
            for (int i = 0; i < kGroupK; ++i) {
                const int idx = g * kGroupK + i;
                pa += static_cast<long double>(wref[idx]) * host_bf16_to_float(x[idx]);
                pb += static_cast<long double>(wref[idx]) * xq_exact[idx];
            }
            ref_a += pa * sc;
            ref_b += pb * sc;
        }

        gemv_baseline<<<grid, block>>>(dx, dcodes, dscales, da, rows, groups);
        gemv_dp4a<<<grid, block>>>(dxq, dds8, dcodes, dscales, db, rows, groups);
        HIP_CHECK(hipDeviceSynchronize());
        float ha = 0.0f, hb = 0.0f;
        HIP_CHECK(hipMemcpy(&ha, da + row, 4, hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemcpy(&hb, db + row, 4, hipMemcpyDeviceToHost));

        const double rel_a = std::fabs((ha - static_cast<double>(ref_a)) /
                                       (std::fabs(static_cast<double>(ref_a)) + 1e-9));
        const double rel_b = std::fabs((hb - static_cast<double>(ref_b)) /
                                       (std::fabs(static_cast<double>(ref_b)) + 1e-9));
        std::printf("  NUM row=%d A_rel=%.3e B_rel=%.3e (ref_a=%.6Le ref_b=%.6Le)\n",
                    row, rel_a, rel_b, ref_a, ref_b);
    }

    hipEvent_t t0, t1;
    HIP_CHECK(hipEventCreate(&t0));
    HIP_CHECK(hipEventCreate(&t1));
    const int iters = 30;
    for (int w = 0; w < 3; ++w) { gemv_baseline<<<grid, block>>>(dx, dcodes, dscales, da, rows, groups); }
    HIP_CHECK(hipEventRecord(t0));
    for (int i = 0; i < iters; ++i) { gemv_baseline<<<grid, block>>>(dx, dcodes, dscales, da, rows, groups); }
    HIP_CHECK(hipEventRecord(t1));
    HIP_CHECK(hipEventSynchronize(t1));
    float ms_a = 0.0f;
    HIP_CHECK(hipEventElapsedTime(&ms_a, t0, t1));
    ms_a /= iters;

    for (int w = 0; w < 3; ++w) { gemv_dp4a<<<grid, block>>>(dxq, dds8, dcodes, dscales, db, rows, groups); }
    HIP_CHECK(hipEventRecord(t0));
    for (int i = 0; i < iters; ++i) { gemv_dp4a<<<grid, block>>>(dxq, dds8, dcodes, dscales, db, rows, groups); }
    HIP_CHECK(hipEventRecord(t1));
    HIP_CHECK(hipEventSynchronize(t1));
    float ms_b = 0.0f;
    HIP_CHECK(hipEventElapsedTime(&ms_b, t0, t1));
    ms_b /= iters;

    // 口径一(名义全流量, 历史对照): A = rows*(k*34/128 + k*2); B = rows*(k*34/128 + k*1 + k*2/32)。
    //   x 向量仅 2-16KB 恒驻 L1, 逐行重读不计 DRAM, 此口径系统性高估。
    // 口径二(码流, 真实 DRAM 判据): rows*k*34/128 权重码字节 / 时间。
    const double gball_a = static_cast<double>(rows) * (k * 34.0 / 128.0 + k * 2.0) / 1e9;
    const double gball_b = static_cast<double>(rows) *
                           (k * 34.0 / 128.0 + k * 1.0 + k * 2.0 / 32.0) / 1e9;
    const double gcode = static_cast<double>(rows) * k * 34.0 / 128.0 / 1e9;
    std::printf(
        "SHAPE [%d,%d]%s  A: %.3f ms codes %.1f GB/s | B: %.3f ms codes %.1f GB/s"
        " | B/A=%.2fx | (名义口径 A %.1f / B %.1f)\n",
        rows, k, l2_note ? " (L2残留意)" : "",
        ms_a, gcode / (ms_a / 1e3), ms_b, gcode / (ms_b / 1e3), ms_a / ms_b,
        gball_a / (ms_a / 1e3), gball_b / (ms_b / 1e3));

    HIP_CHECK(hipFree(dcodes));
    HIP_CHECK(hipFree(dscales));
    HIP_CHECK(hipFree(dx));
    HIP_CHECK(hipFree(dxq));
    HIP_CHECK(hipFree(dds8));
    HIP_CHECK(hipFree(da));
    HIP_CHECK(hipFree(db));
}

}  // namespace

int main() {
    std::printf("== PQ2 decode GEMV 配方对拍 (gfx1100)\n");
    for (int b = 0; b < 256; ++b) {
        for (int i = 0; i < 4; ++i) {
            if (symbol_cpu(b, i) != ((b >> (2 * i)) & 3) - 1) {
                std::printf("SYMBOLS4 MISMATCH b=%d i=%d\n", b, i);
                return 1;
            }
        }
    }
    std::printf("  symbols4 CPU 仿真 256/256 PASS\n");
    run_shape(262144, 8192, false, 42);  // 572MB >> 96MB Infinity Cache, 主判据(DRAM 墙)
    run_shape(27648, 5120, true, 43);    // 引擎 gate_up 真形状, 参照
    return 0;
}

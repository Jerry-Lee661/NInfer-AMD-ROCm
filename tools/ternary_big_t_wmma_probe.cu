// PQ2 ternary big-T WMMA kernel probe (prefill). Host reference = sum((code-1)*scale * x)
// per (row, col), in bf16 input precision. Same generation scheme as
// ternary_small_t_wmma_probe.cu (its decode atom is already probe-proven); this probe
// targets the big-T kernel's own risks: 64-row tiling, 128-token column tiling (incl. a
// t > BN case), the 4-bytes-per-thread decode split, and the out_stride path.
// Build (X99): hipcc -O3 -std=gnu++20 --offload-arch=gfx1100 \
//   -I ~/ninfer/compat/cuda -I ~/ninfer/include -I ~/ninfer/src \
//   ~/ninfer/tools/ternary_big_t_wmma_probe.cu -o /tmp/bigt_probe && /tmp/bigt_probe
#include "ops/linear/ternary/ternary_big_t_wmma.cuh"

#include "core/device.h"  // CUDA_CHECK

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

using ninfer::ops::detail::TernaryBigTJob;

namespace {

int failures = 0;

void run_case(int n, int k, int t, int out_stride, unsigned seed, int bn = 128) {
    const int padded_k = k;  // probe uses k % 128 == 0
    const int groups = k / 128;
    std::vector<std::uint8_t> codes(static_cast<std::size_t>(n) * groups * 32);
    std::vector<std::uint8_t> scales(static_cast<std::size_t>(n) * groups * 2);
    std::vector<float> w_ref(static_cast<std::size_t>(n) * k);
    srand(seed);
    for (int r = 0; r < n; ++r) {
        for (int g = 0; g < groups; ++g) {
            std::uint8_t* base =
                codes.data() + (static_cast<std::size_t>(r) * groups + g) * 32;
            std::uint8_t* sb = scales.data() + (static_cast<std::size_t>(r) * groups + g) * 2;
            const int p2 = rand() % 3;
            static const std::uint16_t kP2Bits[3] = {0x0800, 0x1000, 0x1800};
            const float scale = [] (std::uint16_t bits) {
                const int e = static_cast<int>((bits >> 10) & 0x1f) - 15;
                const float frac = 1.0f + static_cast<float>(bits & 0x3ff) / 1024.0f;
                return std::ldexp(frac, e);
            }(kP2Bits[p2]);
            sb[0] = static_cast<std::uint8_t>(kP2Bits[p2] & 0xff);
            sb[1] = static_cast<std::uint8_t>(kP2Bits[p2] >> 8);
            for (int i = 0; i < 32; ++i) {
                std::uint8_t byte = 0;
                for (int slot = 0; slot < 4; ++slot) {
                    const int code = rand() % 3;
                    byte |= static_cast<std::uint8_t>(code << (2 * slot));
                    const float v = static_cast<float>(code - 1) * scale;
                    w_ref[static_cast<std::size_t>(r) * k + g * 128 + i * 4 + slot] = v;
                }
                base[i] = byte;
            }
        }
    }
    std::vector<float> x(static_cast<std::size_t>(t) * k);
    for (auto& v : x) { v = (rand() % 2000 - 1000) / 500.0f; }
    std::vector<__nv_bfloat16> x_bf(x.size());
    for (std::size_t i = 0; i < x.size(); ++i) { x_bf[i] = __float2bfloat16_rn(x[i]); }
    std::vector<double> ref(static_cast<std::size_t>(t) * n, 0.0);
    for (int c = 0; c < t; ++c) {
        for (int r = 0; r < n; ++r) {
            double acc = 0.0;
            for (int kk = 0; kk < k; ++kk) {
                acc += static_cast<double>(__bfloat162float(x_bf[static_cast<std::size_t>(c) * k + kk])) *
                       w_ref[static_cast<std::size_t>(r) * k + kk];
            }
            ref[static_cast<std::size_t>(c) * n + r] = acc;
        }
    }

    __nv_bfloat16* d_x = nullptr;
    std::uint8_t* d_codes = nullptr;
    std::uint8_t* d_scales = nullptr;
    __nv_bfloat16* d_out = nullptr;
    hipMalloc(&d_x, x_bf.size() * 2);
    hipMalloc(&d_codes, codes.size());
    hipMalloc(&d_scales, scales.size());
    hipMalloc(&d_out, static_cast<std::size_t>(t) * out_stride * 2);
    hipMemcpy(d_x, x_bf.data(), x_bf.size() * 2, hipMemcpyHostToDevice);
    hipMemcpy(d_codes, codes.data(), codes.size(), hipMemcpyHostToDevice);
    hipMemcpy(d_scales, scales.data(), scales.size(), hipMemcpyHostToDevice);
    hipMemset(d_out, 0, static_cast<std::size_t>(t) * out_stride * 2);

    TernaryBigTJob job{d_x, d_codes, d_scales, d_out, n, k, padded_k, out_stride};
    // C1: bn=128(256线程) 与 bn=64(128线程,每 SIMD 2 CTA 驻留)两个实例
    const dim3 grid(bn == 128 ? dim3((n + 63) / 64, (t + 127) / 128)
                              : dim3((n + 63) / 64, (t + 63) / 64));
    const dim3 block(bn == 128 ? 256 : 128);
    // C1 计时:同小 T 探针口径
    const double weight_bytes = static_cast<double>(n) * k * 34.0 / 128.0;
    const double total_bytes =
        weight_bytes + static_cast<double>(t) * (k + out_stride) * 2.0;
    hipEvent_t ev0, ev1;
    hipEventCreate(&ev0);
    hipEventCreate(&ev1);
    hipEventRecord(ev0);
    for (int rep = 0; rep < 20; ++rep) {
        if (bn == 128) {
            // 模板实参带逗号,不能用 hipLaunchKernelGGL 宏(会被当参数劈裂)
            ninfer::ops::detail::ternary_big_t_wmma_kernel<128, 256>
                <<<grid, block, 0, 0>>>(job, t);
        } else {
            ninfer::ops::detail::ternary_big_t_wmma_kernel<64, 128><<<grid, block, 0, 0>>>(job, t);
        }
    }
    hipEventRecord(ev1);
    hipEventSynchronize(ev1);
    float ms = 0.0f;
    hipEventElapsedTime(&ms, ev0, ev1);
    ms /= 20.0f;
    std::printf("  timing(bn=%d): %.4f ms = %.1f GB/s\n", bn, ms, total_bytes / ms / 1e6);
    if (bn == 128) {
        ninfer::ops::detail::ternary_big_t_wmma_kernel<128, 256><<<grid, block, 0, 0>>>(job, t);
    } else {
        ninfer::ops::detail::ternary_big_t_wmma_kernel<64, 128><<<grid, block, 0, 0>>>(job, t);
    }
    const hipError_t launch_err = hipGetLastError();
    if (launch_err != hipSuccess) {
        std::printf("  LAUNCH FAIL: %s\n", hipGetErrorString(launch_err));
        ++failures;
        hipFree(d_x); hipFree(d_codes); hipFree(d_scales); hipFree(d_out);
        return;
    }
    std::vector<__nv_bfloat16> out(static_cast<std::size_t>(t) * out_stride);
    hipMemcpy(out.data(), d_out, out.size() * 2, hipMemcpyDeviceToHost);

    double worst = 0.0;
    std::size_t bad = 0;
    for (int c = 0; c < t; ++c) {
        for (int r = 0; r < n; ++r) {
            const double got = __bfloat162float(out[static_cast<std::size_t>(c) * out_stride + r]);
            const double want = ref[static_cast<std::size_t>(c) * n + r];
            const double diff = std::fabs(got - want);
            const double tol = 0.05 + 0.01 * std::fabs(want);
            if (diff > tol) {
                if (bad < 6) {
                    std::printf("  bad c=%d r=%d got=%.4f want=%.4f\n", c, r, got, want);
                }
                ++bad;
                worst = std::max(worst, diff);
            }
        }
    }
    std::printf("case n=%d k=%d t=%d stride=%d seed=%u: %s (bad=%zu worst=%.4f)\n", n, k, t,
                out_stride, seed, bad == 0 ? "PASS" : "FAIL", bad, worst);
    if (bad != 0) { ++failures; }
    hipFree(d_x); hipFree(d_codes); hipFree(d_scales); hipFree(d_out);
}

}  // namespace

int main() {
    run_case(64, 512, 17, 64, 1);     // min big-T
    run_case(64, 512, 33, 64, 2);
    run_case(128, 256, 100, 128, 3);  // min k (2 groups), t in one col tile
    run_case(96, 512, 130, 96, 4);    // n tail tile + t crosses BN=128 (2 col tiles)
    run_case(4096, 5120, 100, 4096, 5);   // engine: gdn qk parent shape at prefill T
    run_case(34816, 5120, 17, 34816, 6);  // engine: gate_up N (host ref cost caps T here)
    run_case(5120, 17408, 33, 5120, 7);   // engine: gdn value parent k
    run_case(4096, 5120, 20, 10240, 8);   // fused qkv buffer: out_stride != n
    run_case(34816, 5120, 3, 34816, 9);   // C1: big-T 核在小 t(MLP gate_up 形状)
    run_case(5120, 17408, 3, 5120, 10);   // C1: big-T 核在小 t(down 形状)
    run_case(34816, 5120, 3, 34816, 11, 64);   // C1: bn=64 占位变体
    run_case(5120, 17408, 3, 5120, 12, 64);    // C1: bn=64 占位变体
    run_case(34816, 5120, 100, 34816, 13, 64); // C1: bn=64 大 t 对照
    run_case(4096, 5120, 100, 4096, 14, 64);   // C1: bn=64 大 t 对照(n 小 波数少)
    std::printf(failures == 0 ? "ALL PASS\n" : "FAILURES: %d\n", failures);
    return failures == 0 ? 0 : 1;
}

// KV int4 第一刀探针(2026-10-01):① 数值自证 = chain 0 逐 lane dump 对 host
// double 全链参考(含 warp_sum 复算);② int8 vs int4 计时,双链数 regime
// (argv[1] = CTA 数:64 ≈ 带宽级 512 链,16 ≈ 引擎级 128 链;G1 教训:微基准
// 外推需标注 regime)。
// 打包约定:值域 [-8,7],nibble n = v+8,字节 = (奇维高 4 位 | 偶维低 4 位)。
// Build (X99): hipcc -O3 -std=gnu++20 --offload-arch=gfx1100 -DNINFER_HIP=1 \
//   -I compat/cuda -I include -I src tools/attn_int4_probe.cu -o /tmp/int4_probe
#include "core/device.h"

#include <cuda_fp16.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

constexpr int kWarpsPerCta = 8;
constexpr int kKeys        = 256;
constexpr int kReps        = 8;
constexpr float kQkScale   = 0.125f;
constexpr float kLog2E     = 1.4426950408889634f;

__device__ __forceinline__ float warp_sum32(float v) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        v += __shfl_xor_sync(0xffffffffffffffffULL, v, off, 32);
    }
    return v;
}

// mode 0 = int8 全链;mode 3 = int4 全链;mode 4 = int4 dump(chain0 逐 lane:
// out[(lane)*9 + 0..8] = {l, acc[0..7]})
__global__ __launch_bounds__(256, 1) void attn_chain4_kernel(
    const int2* __restrict__ k8, const int2* __restrict__ v8,
    const std::uint32_t* __restrict__ k4, const std::uint32_t* __restrict__ v4,
    const __half* __restrict__ ks, float* __restrict__ out, int mode, int n_keys) {
    const int tid   = static_cast<int>(threadIdx.x);
    const int lane  = tid & 31;
    const int warp  = tid >> 5;
    const int chain = static_cast<int>(blockIdx.x) * kWarpsPerCta + warp;
    const std::int64_t base8 = static_cast<std::int64_t>(chain) * kKeys * 32;
    const std::int64_t base4 = base8;

    float qv[8];
    float acc[8];
    float qsum = 0.0f;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        qv[i]  = 0.03125f * static_cast<float>((lane + i) % 7 - 3);
        acc[i] = 0.0f;
        qsum += qv[i];
    }
    const float qsum8 = 8.0f * qsum;
    float m = -1e30f;
    float l = 0.0f;

    for (int key = 0; key < n_keys; ++key) {
        const std::int64_t idx8 = base8 + static_cast<std::int64_t>(key) * 32 + lane;
        const std::int64_t idx4 = base4 + static_cast<std::int64_t>(key) * 32 + lane;
        const float ksc = __half2float(ks[idx8 / 8 + lane]);
        const float vsc = __half2float(ks[idx8 / 8 + lane + 1]);
        float partial = 0.0f;
        if (mode == 0) {
            const int2 kk  = k8[idx8];
            const auto* kc = reinterpret_cast<const std::int8_t*>(&kk);
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                partial = fmaf(qv[i], static_cast<float>(kc[i]) * ksc, partial);
            }
        } else {
            const std::uint32_t w4k = k4[idx4];
            float p4 = 0.0f;
#pragma unroll
            for (int b = 0; b < 4; ++b) {
                const std::uint32_t kb = (w4k >> (8 * b)) & 0xFFu;
                p4 = fmaf(qv[2 * b], static_cast<float>(kb & 0xFu), p4);
                p4 = fmaf(qv[2 * b + 1], static_cast<float>(kb >> 4), p4);
            }
            partial = (p4 - qsum8) * ksc;  // 偏移技巧:值 = n-8
        }
        const float sc = warp_sum32(partial) * kQkScale;
        const float nm = fmaxf(m, sc);
        const float a  = (m == -1e30f) ? 0.0f : exp2f((m - nm) * kLog2E);
        const float p  = exp2f((sc - nm) * kLog2E);
#pragma unroll
        for (int i = 0; i < 8; ++i) { acc[i] *= a; }
        l = l * a + p;
        m = nm;
        if (mode == 0) {
            const int2 kv2 = v8[idx8];
            const auto* vc = reinterpret_cast<const std::int8_t*>(&kv2);
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                acc[i] = fmaf(p, static_cast<float>(vc[i]) * vsc, acc[i]);
            }
        } else {
            const std::uint32_t w4v = v4[idx4];
            const float pv = p * vsc;
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const std::uint32_t vb = (w4v >> (8 * (i / 2))) & 0xFFu;
                const float dn = static_cast<float>((i % 2 == 0) ? (vb & 0xFu) : (vb >> 4)) - 8.0f;
                acc[i] = fmaf(pv, dn, acc[i]);
            }
        }
    }

    if (mode == 4 && chain == 0) {
        float* slot = out + lane * 9;
        slot[0] = l;
#pragma unroll
        for (int i = 0; i < 8; ++i) { slot[1 + i] = acc[i]; }
        return;
    }
    float s = l;
#pragma unroll
    for (int i = 0; i < 8; ++i) { s += acc[i]; }
    if (s == 12345.678f) { out[chain] = s; }  // 防 DCE
}

}  // namespace

int main(int argc, char** argv) {
    int ctas = 64;
    if (argc > 1) { ctas = std::atoi(argv[1]); }
    const int chains = ctas * kWarpsPerCta;
    const std::size_t i8n = static_cast<std::size_t>(chains) * kKeys * 32;

    std::vector<int2> k8(i8n);
    std::vector<int2> v8(i8n);
    std::vector<std::uint32_t> k4(i8n);
    std::vector<std::uint32_t> v4(i8n);
    std::vector<__half> ks(i8n / 4);
    srand(11);
    const auto clip4 = [](int x) { return (x < -8 ? -8 : (x > 7 ? 7 : x)) + 8; };
    for (std::size_t i = 0; i < i8n; ++i) {
        const int a = rand() % 256 - 128;
        const int b = rand() % 256 - 128;
        k8[i] = {a, b};
        v8[i] = {b, a};
        const unsigned char* kb = reinterpret_cast<const unsigned char*>(&k8[i]);
        const unsigned char* vb = reinterpret_cast<const unsigned char*>(&v8[i]);
        std::uint32_t pk = 0;
        std::uint32_t pv = 0;
        for (int j = 0; j < 4; ++j) {
            const std::uint32_t lok = static_cast<std::uint32_t>(clip4(static_cast<signed char>(kb[2 * j])));
            const std::uint32_t hik = static_cast<std::uint32_t>(clip4(static_cast<signed char>(kb[2 * j + 1])));
            const std::uint32_t lov = static_cast<std::uint32_t>(clip4(static_cast<signed char>(vb[2 * j])));
            const std::uint32_t hiv = static_cast<std::uint32_t>(clip4(static_cast<signed char>(vb[2 * j + 1])));
            pk |= (hik << (8 * j + 4)) | (lok << (8 * j));
            pv |= (hiv << (8 * j + 4)) | (lov << (8 * j));
        }
        k4[i] = pk;
        v4[i] = pv;
    }
    for (auto& h : ks) { h = __float2half(0.1f); }  // 0.01 时 32-lane 平均 sc 差异 < fp32 分辨率,p 全饱和成 1

    int2* d_k8 = nullptr;
    int2* d_v8 = nullptr;
    std::uint32_t* d_k4 = nullptr;
    std::uint32_t* d_v4 = nullptr;
    __half* d_ks = nullptr;
    float* d_out = nullptr;
    hipMalloc(&d_k8, i8n * sizeof(int2));
    hipMalloc(&d_v8, i8n * sizeof(int2));
    hipMalloc(&d_k4, i8n * 4);
    hipMalloc(&d_v4, i8n * 4);
    hipMalloc(&d_ks, ks.size() * 2);
    hipMalloc(&d_out, static_cast<std::size_t>(chains) * 9 * 4 + 64);
    hipMemcpy(d_k8, k8.data(), i8n * sizeof(int2), hipMemcpyHostToDevice);
    hipMemcpy(d_v8, v8.data(), i8n * sizeof(int2), hipMemcpyHostToDevice);
    hipMemcpy(d_k4, k4.data(), i8n * 4, hipMemcpyHostToDevice);
    hipMemcpy(d_v4, v4.data(), i8n * 4, hipMemcpyHostToDevice);
    hipMemcpy(d_ks, ks.data(), ks.size() * 2, hipMemcpyHostToDevice);

    // ---- 数值自证:1 CTA,mode 4 dump chain0 全 32 lane 对 host double 参考 ----
    {
        std::vector<float> hout(32 * 9, -1.0f);
        hipMemcpy(d_out, hout.data(), 32 * 9 * 4, hipMemcpyHostToDevice);
        attn_chain4_kernel<<<1, 256>>>(d_k8, d_v8, d_k4, d_v4, d_ks, d_out, 4, 8);
        hipMemcpy(hout.data(), d_out, 32 * 9 * 4, hipMemcpyDeviceToHost);
        // host double 参考:q 由 lane 决定;partial[key][lane] → warp_sum(全 32 lane 和)
        const int probe_keys = 8;  // 8 个 key 已足够暴露解码/偏移/softmax 错
        std::vector<double> partial(probe_keys * 32);
        for (int key = 0; key < probe_keys; ++key) {
            for (int lane = 0; lane < 32; ++lane) {
                double qv[8];
                double qs = 0.0;
                for (int i = 0; i < 8; ++i) {
                    qv[i] = 0.03125 * static_cast<double>((lane + i) % 7 - 3);
                    qs += qv[i];
                }
                const std::uint32_t w4 = k4[(static_cast<std::size_t>(key) * 32) + lane];
                double p4 = 0.0;
                for (int b = 0; b < 4; ++b) {
                    const std::uint32_t kb = (w4 >> (8 * b)) & 0xFFu;
                    p4 += qv[2 * b] * static_cast<double>(kb & 0xFu);
                    p4 += qv[2 * b + 1] * static_cast<double>(kb >> 4);
                }
                const double ksc = 0.1;
                partial[key * 32 + lane] = (p4 - 8.0 * qs) * ksc;
            }
        }
        // 逐 lane 在线 softmax(double;sc = 全 lane 和 × scale)
        double worst = 0.0;
        int w_lane = -1;
        const char* w_slot = "";
        double w_kernel = 0.0;
        double w_host = 0.0;
        for (int lane = 0; lane < 32; ++lane) {
            double qv[8];
            for (int i = 0; i < 8; ++i) { qv[i] = 0.03125 * ((lane + i) % 7 - 3); }
            double acc[8] = {0};
            double m = -1e300;
            double l = 0.0;
            for (int key = 0; key < probe_keys; ++key) {
                double sp = 0.0;
                for (int l2 = 0; l2 < 32; ++l2) { sp += partial[key * 32 + l2]; }
                const double ksc = 0.1;
                const double vsc = 0.1;
                const std::uint32_t w4v = v4[(static_cast<std::size_t>(key) * 32) + lane];
                const double sc = sp * 0.125;
                const double nm = std::fmax(m, sc);
                const double a = (m == -1e300) ? 0.0 : std::exp2((m - nm) * 1.4426950408889634);
                const double p = std::exp2((sc - nm) * 1.4426950408889634);
                for (int i = 0; i < 8; ++i) { acc[i] *= a; }
                l = l * a + p;
                m = nm;
                for (int i = 0; i < 8; ++i) {
                    const int b = (w4v >> (8 * (i / 2))) & 0xFFu;
                    const double n = (i % 2 == 0) ? (double)(b & 0xFu) : (double)(b >> 4);
                    acc[i] += p * vsc * (n - 8.0);
                }
            }
            const float* slot = &hout[lane * 9];
            const double dl = std::fabs(static_cast<double>(slot[0]) - l);
            if (dl > worst) {
                worst = dl; w_lane = lane; w_slot = "l"; w_kernel = slot[0]; w_host = l;
            }
            for (int i = 0; i < 8; ++i) {
                const double d = std::fabs(static_cast<double>(slot[1 + i]) - acc[i]);
                if (d > worst) {
                    worst = d; w_lane = lane; w_slot = "acc"; w_kernel = slot[1 + i];
                    w_host = acc[i];
                }
            }
        }
        std::printf("numeric(int4 vs double, 8 keys): worst=%.6g %s\n", worst,
                    worst < 2e-2 ? "OK" : "BAD");
        if (worst >= 5e-3) {
            std::printf("  worst lane=%d slot=%s kernel=%.6f host=%.6f\n", w_lane, w_slot,
                        w_kernel, w_host);
        }
    }

    // ---- 计时:int8(mode0) vs int4(mode3) ----
    for (int mode = 0; mode <= 3; mode += 3) {
        hipEvent_t ev0, ev1;
        hipEventCreate(&ev0);
        hipEventCreate(&ev1);
        attn_chain4_kernel<<<ctas, 256>>>(d_k8, d_v8, d_k4, d_v4, d_ks, d_out, mode, kKeys);
        hipDeviceSynchronize();
        hipEventRecord(ev0);
        for (int r = 0; r < kReps; ++r) {
            attn_chain4_kernel<<<ctas, 256>>>(d_k8, d_v8, d_k4, d_v4, d_ks, d_out, mode, kKeys);
        }
        hipEventRecord(ev1);
        hipEventSynchronize(ev1);
        float ms = 0.0f;
        hipEventElapsedTime(&ms, ev0, ev1);
        const double bytes_per_key = (mode == 0 ? 512.0 : 256.0);
        const double gb_s = static_cast<double>(kReps) * chains * kKeys * bytes_per_key / ms / 1e6;
        std::printf("chains=%d mode %d(%s): %.3f ms/rep = %.0f GB/s(逻辑 K+V 流)\n", chains, mode,
                    mode == 0 ? "int8" : "int4", ms / kReps, gb_s);
        hipEventDestroy(ev0);
        hipEventDestroy(ev1);
    }
    const hipError_t err = hipGetLastError();
    if (err != hipSuccess) { std::printf("ERR: %s\n", hipGetErrorString(err)); }
    hipFree(d_k8); hipFree(d_v8); hipFree(d_k4); hipFree(d_v4); hipFree(d_ks); hipFree(d_out);
    return 0;
}

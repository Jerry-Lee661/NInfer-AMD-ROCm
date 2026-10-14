// attn key 链微基准(fork ① 判定实验,2026-10-01):回答"每 key 的 K/V 载入是否已被
// 编译器跨迭代管线化"。镜像 small_t_lean.cu Contig8 分支的逐 key 链(int2 载入 +
// 8 FMA + warp_sum 5-shfl + 在线 softmax + V 8 FMA),三模式:
//   mode 0 = 全链(现状)
//   mode 1 = keys-only(去掉 warp_sum/softmax 的纯载入+FMA)
//   mode 2 = 全链 + 手动 K/V[key+1] 寄存器预取
// 判读:mode0 ≈ mode1 ⇒ 载入已管线化,收益在 softmax/shfl 链(难吃);
//       mode0 ≫ mode1 ⇒ 载入串行化,预取/管线化即赢;mode2 ≪ mode0 同证。
// Build (X99): hipcc -O3 -std=gnu++20 --offload-arch=gfx1100 -DNINFER_HIP=1 \
//   -I compat/cuda -I include -I src tools/attn_keychain_probe.cu -o /tmp/attn_probe
#include "core/device.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

constexpr int kWarpsPerCta = 8;
constexpr int kCtas        = 64;
constexpr int kChains      = kWarpsPerCta * kCtas;  // 512 条独立链
constexpr int kKeys        = 256;                    // 每链 key 窗口
constexpr int kReps        = 8;                      // 计时重复

__device__ __forceinline__ float warp_sum32(float v) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        v += __shfl_xor_sync(0xffffffffffffffffULL, v, off, 32);
    }
    return v;
}

__global__ __launch_bounds__(256, 1) void attn_chain_kernel(const int2* __restrict__ k_buf,
                                                            const int2* __restrict__ v_buf,
                                                            const __half* __restrict__ ks_buf,
                                                            float* __restrict__ out, int mode) {
    const int tid  = static_cast<int>(threadIdx.x);
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int chain = static_cast<int>(blockIdx.x) * kWarpsPerCta + warp;
    const std::int64_t base = static_cast<std::int64_t>(chain) * kKeys * 32;

    float qv[8];
    float acc[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        qv[i]  = 0.03125f * static_cast<float>((lane + i) % 7 - 3);
        acc[i] = 0.0f;
    }
    float m = -1e30f;
    float l = 0.0f;
    float sink = 0.0f;

    if (mode == 1) {
        // keys-only:载入 + FMA,无归约/softmax
        for (int key = 0; key < kKeys; ++key) {
            const std::int64_t idx = (base + static_cast<std::int64_t>(key) * 32) + lane;
            const int2 kk = k_buf[idx];
            const int2 vv = v_buf[idx];
            const float ksc = __half2float(ks_buf[(base + key * 32) / 8 + lane]);
            const float vsc = __half2float(ks_buf[(base + key * 32) / 8 + lane + 1]);
            const auto* kc = reinterpret_cast<const std::int8_t*>(&kk);
            const auto* vc = reinterpret_cast<const std::int8_t*>(&vv);
            float p = 0.0f;
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                p = fmaf(qv[i], static_cast<float>(kc[i]) * ksc, p);
                acc[i] = fmaf(p, static_cast<float>(vc[i]) * vsc, acc[i]);
            }
            sink += p;
        }
    } else if (mode == 2) {
        // 全链 + 手动预取:K/V[key+1] 在本 key 的 softmax 段之前进寄存器
        int2 kk = k_buf[base + lane];
        int2 vv = v_buf[base + lane];
        for (int key = 0; key < kKeys; ++key) {
            const bool more = (key + 1 < kKeys);
            const std::int64_t nxt = base + static_cast<std::int64_t>(key + more) * 32 + lane;
            const int2 kk_n = more ? k_buf[nxt] : kk;
            const int2 vv_n = more ? v_buf[nxt] : vv;
            const std::int64_t idx = base + static_cast<std::int64_t>(key) * 32 + lane;
            const float ksc = __half2float(ks_buf[idx / 8 + lane]);
            const auto* kc = reinterpret_cast<const std::int8_t*>(&kk);
            float partial = 0.0f;
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                partial = fmaf(qv[i], static_cast<float>(kc[i]) * ksc, partial);
            }
            const float sc = warp_sum32(partial) * 0.125f;
            const float nm = fmaxf(m, sc);
            const float a  = (m == -1e30f) ? 0.0f : exp2f((m - nm) * 1.4426950408889634f);
            const float p  = exp2f((sc - nm) * 1.4426950408889634f);
#pragma unroll
            for (int i = 0; i < 8; ++i) { acc[i] *= a; }
            l = l * a + p;
            m = nm;
            const float vsc = __half2float(ks_buf[idx / 8 + lane + 1]);
            const auto* vc = reinterpret_cast<const std::int8_t*>(&vv);
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                acc[i] = fmaf(p, static_cast<float>(vc[i]) * vsc, acc[i]);
            }
            kk = kk_n;
            vv = vv_n;
        }
    } else {
        // mode 0:全链(现状结构)
        for (int key = 0; key < kKeys; ++key) {
            const std::int64_t idx = base + static_cast<std::int64_t>(key) * 32 + lane;
            const int2 kk = k_buf[idx];
            const float ksc = __half2float(ks_buf[idx / 8 + lane]);
            const auto* kc = reinterpret_cast<const std::int8_t*>(&kk);
            float partial = 0.0f;
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                partial = fmaf(qv[i], static_cast<float>(kc[i]) * ksc, partial);
            }
            const float sc = warp_sum32(partial) * 0.125f;
            const float nm = fmaxf(m, sc);
            const float a  = (m == -1e30f) ? 0.0f : exp2f((m - nm) * 1.4426950408889634f);
            const float p  = exp2f((sc - nm) * 1.4426950408889634f);
#pragma unroll
            for (int i = 0; i < 8; ++i) { acc[i] *= a; }
            l = l * a + p;
            m = nm;
            const int2 vv = v_buf[idx];
            const float vsc = __half2float(ks_buf[idx / 8 + lane + 1]);
            const auto* vc = reinterpret_cast<const std::int8_t*>(&vv);
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                acc[i] = fmaf(p, static_cast<float>(vc[i]) * vsc, acc[i]);
            }
        }
    }
    (void)sink;
    float s = l;
#pragma unroll
    for (int i = 0; i < 8; ++i) { s += acc[i]; }
    if (s == 12345.678f) { out[chain] = s; }  // 防 DCE
}

}  // namespace

int main() {
    const std::size_t ints_per_chain = static_cast<std::size_t>(kKeys) * 32 * 2;  // int2 计
    std::vector<int2> buf(kChains * kKeys * 32);
    srand(7);
    for (std::size_t i = 0; i < buf.size(); ++i) {
        const int lo = rand() % 256 - 128;
        const int hi = rand() % 256 - 128;
        buf[i].x = lo;
        buf[i].y = hi;
    }
    std::vector<__half> ks(buf.size() / 4);
    for (auto& h : ks) { h = __float2half(0.01f); }
    std::vector<float> out(kChains, 0.0f);

    int2* d_k = nullptr;
    int2* d_v = nullptr;
    __half* d_ks = nullptr;
    float* d_out = nullptr;
    hipMalloc(&d_k, buf.size() * sizeof(int2));
    hipMalloc(&d_v, buf.size() * sizeof(int2));
    hipMalloc(&d_ks, ks.size() * 2);
    hipMalloc(&d_out, out.size() * 4);
    hipMemcpy(d_k, buf.data(), buf.size() * sizeof(int2), hipMemcpyHostToDevice);
    hipMemcpy(d_v, buf.data(), buf.size() * sizeof(int2), hipMemcpyHostToDevice);
    hipMemcpy(d_ks, ks.data(), ks.size() * 2, hipMemcpyHostToDevice);

    for (int mode = 0; mode <= 2; ++mode) {
        hipEvent_t ev0, ev1;
        hipEventCreate(&ev0);
        hipEventCreate(&ev1);
        // 预热
        attn_chain_kernel<<<kCtas, 256>>>(d_k, d_v, d_ks, d_out, mode);
        hipDeviceSynchronize();
        hipEventRecord(ev0);
        for (int r = 0; r < kReps; ++r) {
            attn_chain_kernel<<<kCtas, 256>>>(d_k, d_v, d_ks, d_out, mode);
        }
        hipEventRecord(ev1);
        hipEventSynchronize(ev1);
        float ms = 0.0f;
        hipEventElapsedTime(&ms, ev0, ev1);
        const double ns_per_key =
            static_cast<double>(ms) * 1e6 / (static_cast<double>(kReps) * kChains * kKeys);
        const double gb_s = static_cast<double>(kReps) * kChains * kKeys * 512.0 / ms / 1e6;
        std::printf("mode %d: %.3f ms/rep = %.0f ns/key = %.0f GB/s\n", mode, ms / kReps,
                    ns_per_key, gb_s);
        hipEventDestroy(ev0);
        hipEventDestroy(ev1);
    }
    const hipError_t err = hipGetLastError();
    if (err != hipSuccess) { std::printf("ERR: %s\n", hipGetErrorString(err)); }
    hipFree(d_k); hipFree(d_v); hipFree(d_ks); hipFree(d_out);
    return 0;
}

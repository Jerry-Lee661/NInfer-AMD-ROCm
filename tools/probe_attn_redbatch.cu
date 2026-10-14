// P-C⑦ 探针(2026-10-04): 注意力 keys 循环的"归约批化"结构 vs 现行逐 key/MK4 展开。
//
// 病灶(排除法定案, pc6/pc3-历史): keys 循环每 key 的串行依赖链 =
// warp_sum(5 shfl) → 在线 softmax m/l 更新 → V 累加, 首尾相接; 加载宽度/预取/批宽
// 均非杠杆。归约批化 = 每 lane 持 RB 个 key 的部分和(寄存器), 依赖链改为:
//   [部分和 ×RB 并行累] → [一次多值归约(1 条链出 RB 个 sc)] → [一次批量 softmax 重放] → [V 累加 ×RB]
// 依赖链长度 ÷RB; 代价 = 归约中间值重建(RB-1 次额外 __shfl_xor 交换)。
//
// 三臂: A=逐 key(现行 lean MultiKey=1 主路径语句级同构), B=MK4 展开(现行),
// C=归约批化 RB=4(本轮新结构)。同 512 keys × head_dim 64 × 8 heads 网格,
// int8 KV 语义(K 装载+dequant 与引擎同款), 计时 keys 段墙钟。
// 判据: C 对 A 数值 ≤1e-4(softmax 结合序变化走 lean 协议先例), 时序 B/A/C 三点。
//
// 编译(X99): hipcc --offload-arch=gfx1100 probe_attn_redbatch.cu -o /tmp/probe_rb
// 运行: HIP_VISIBLE_DEVICES=1 /tmp/probe_rb

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

constexpr int kHeadDim   = 64;
constexpr int kWarps     = 8;    // 与引擎 256 线程一致
constexpr int kKVHeads   = 8;    // GQA=1: 每 head 一 warp 行
constexpr int kKeys      = 512;  // 单 tile 窗口上限(引擎 split 后单 split 同量级)
constexpr int kScaleShfl = 5;    // warp_sum 链长 log2(32)

__device__ __forceinline__ float warp_sum1(float v) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        v += __shfl_xor_sync(0xffffffffffffffffULL, v, off);
    }
    return v;
}

// 归约批化核心: 每 lane 持 RB 个部分和, 一次走出 RB 个全和。
// 实现 = RB-1 对 __shfl_xor 交换 + 局部求和(交换矩阵法, 依赖深度 2×ceil(log2(RB))
// 而非 RB×kScaleShfl):
//   对 RB=4: 和 j = sum_i v[i]·[lane 组 (i+j) mod RB 相加] —— 4 次交换 2 轮。
// 这里用最直白的成对交换展开(RB=4 专写), 每 lane 返回 4 个全和。
__device__ __forceinline__ void warp_sum4(float v0, float v1, float v2, float v3, float* out) {
    // 轮 1: 与 lane^1 交换 (v1,v0,v3,v2) → s01 = v0+v1', s10 = v1+v0', 同理 v2/v3。
    const float x0 = __shfl_xor_sync(0xffffffffffffffffULL, v0, 1);
    const float x1 = __shfl_xor_sync(0xffffffffffffffffULL, v1, 1);
    const float x2 = __shfl_xor_sync(0xffffffffffffffffULL, v2, 1);
    const float x3 = __shfl_xor_sync(0xffffffffffffffffULL, v3, 1);
    const float s01 = v0 + x1;
    const float s10 = v1 + x0;
    const float s23 = v2 + x3;
    const float s32 = v3 + x2;
    // 轮 2: 与 lane^2 交换半值。
    const float y0 = __shfl_xor_sync(0xffffffffffffffffULL, s01, 2);
    const float y2 = __shfl_xor_sync(0xffffffffffffffffULL, s23, 2);
    // 全和: out[0] = 本 lane 的 s01 + lane^2 的 s23(配对由 xor 矩阵保证)。
    out[0] = s01 + y2;
    out[1] = s10 + y2;  // 近似分配: 充分利用交换值, 见下方数值门
    out[2] = s23 + y0;
    out[3] = s32 + y0;
}

// ---- 臂 A: 逐 key(现行主路径同构, 一 warp 一 head, keys 循环) ----
__global__ __launch_bounds__(256)
void attn_perkey(const std::int8_t* __restrict__ kcache, const std::int8_t* __restrict__ vcache,
                 const __half* __restrict__ kscale, const __half* __restrict__ vscale,
                 const float* __restrict__ q, float* __restrict__ out, int n_keys, float scale) {
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp = static_cast<int>(blockIdx.x) * kWarps + (static_cast<int>(threadIdx.x) >> 5);
    if (warp >= kKVHeads) { return; }
    const std::int8_t* kbase = kcache + static_cast<std::int64_t>(warp) * kKeys * kHeadDim;
    const std::int8_t* vbase = vcache + static_cast<std::int64_t>(warp) * kKeys * kHeadDim;
    const __half* ks = kscale + static_cast<std::int64_t>(warp) * kKeys * 8;
    const __half* vs = vscale + static_cast<std::int64_t>(warp) * kKeys * 8;
    const float* qv = q + static_cast<std::int64_t>(warp) * kHeadDim;

    float acc[8];
#pragma unroll
    for (int r = 0; r < 8; ++r) { acc[r] = 0.0f; }
    float m = -INFINITY, l = 0.0f;

    for (int key = 0; key < n_keys; ++key) {
        float partial = 0.0f;
        const int2 kk = *reinterpret_cast<const int2*>(kbase + key * kHeadDim + 8 * lane);
        const std::int8_t* kc = reinterpret_cast<const std::int8_t*>(&kk);
        const float ksc = __half2float(ks[key * 8 + (lane >> 3)]);
#pragma unroll
        for (int i = 0; i < 8; ++i) { partial = fmaf(qv[8 * lane + i], static_cast<float>(kc[i]) * ksc, partial); }
        const float sc = warp_sum1(partial) * scale;
        const int2 vv = *reinterpret_cast<const int2*>(vbase + key * kHeadDim + 8 * lane);
        const std::int8_t* vc = reinterpret_cast<const std::int8_t*>(&vv);
        const float vsc = __half2float(vs[key * 8 + (lane >> 3)]);
        const float nm = fmaxf(m, sc);
        const float a = (m == -INFINITY) ? 0.0f : exp2f((m - nm) * 1.4426950408889634f);
        const float p = exp2f((sc - nm) * 1.4426950408889634f);
#pragma unroll
        for (int r = 0; r < 8; ++r) { acc[r] *= a; }
        l = l * a + p;
        m = nm;
#pragma unroll
        for (int i = 0; i < 8; ++i) { acc[i] = fmaf(p, static_cast<float>(vc[i]) * vsc, acc[i]); }
    }
#pragma unroll
    for (int r = 0; r < 8; ++r) { acc[r] /= l; }
    if (lane == 0) {
        for (int r = 0; r < 8; ++r) { out[warp * kHeadDim + r] = acc[r]; }
    }
}

// ---- 臂 B: MK4 展开(现行批路径同构: 4 组独立部分和 + 4 独立 warp_sum) ----
__global__ __launch_bounds__(256)
void attn_mk4(const std::int8_t* __restrict__ kcache, const std::int8_t* __restrict__ vcache,
              const __half* __restrict__ kscale, const __half* __restrict__ vscale,
              const float* __restrict__ q, float* __restrict__ out, int n_keys, float scale) {
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp = static_cast<int>(blockIdx.x) * kWarps + (static_cast<int>(threadIdx.x) >> 5);
    if (warp >= kKVHeads) { return; }
    const std::int8_t* kbase = kcache + static_cast<std::int64_t>(warp) * kKeys * kHeadDim;
    const std::int8_t* vbase = vcache + static_cast<std::int64_t>(warp) * kKeys * kHeadDim;
    const __half* ks = kscale + static_cast<std::int64_t>(warp) * kKeys * 8;
    const __half* vs = vscale + static_cast<std::int64_t>(warp) * kKeys * 8;
    const float* qv = q + static_cast<std::int64_t>(warp) * kHeadDim;

    float acc[8];
#pragma unroll
    for (int r = 0; r < 8; ++r) { acc[r] = 0.0f; }
    float m = -INFINITY, l = 0.0f;

    for (int key = 0; key < n_keys; key += 4) {
        float partials[4] = {0, 0, 0, 0};
        float vcontrib[4][8];
        int live = 0;
        for (int b = 0; b < 4 && key + b < n_keys; ++b) {
            ++live;
            const int2 kk = *reinterpret_cast<const int2*>(kbase + (key + b) * kHeadDim + 8 * lane);
            const std::int8_t* kc = reinterpret_cast<const std::int8_t*>(&kk);
            const float ksc = __half2float(ks[(key + b) * 8 + (lane >> 3)]);
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                partials[b] = fmaf(qv[8 * lane + i], static_cast<float>(kc[i]) * ksc, partials[b]);
            }
        }
        for (int b = 0; b < live; ++b) {
            const int2 vv = *reinterpret_cast<const int2*>(vbase + (key + b) * kHeadDim + 8 * lane);
            const std::int8_t* vc = reinterpret_cast<const std::int8_t*>(&vv);
            const float vsc = __half2float(vs[(key + b) * 8 + (lane >> 3)]);
#pragma unroll
            for (int i = 0; i < 8; ++i) { vcontrib[b][i] = static_cast<float>(vc[i]) * vsc; }
        }
        for (int b = 0; b < live; ++b) {
            const float sc = warp_sum1(partials[b]) * scale;
            const float nm = fmaxf(m, sc);
            const float a = (m == -INFINITY) ? 0.0f : exp2f((m - nm) * 1.4426950408889634f);
            const float p = exp2f((sc - nm) * 1.4426950408889634f);
#pragma unroll
            for (int r = 0; r < 8; ++r) { acc[r] *= a; }
            l = l * a + p;
            m = nm;
#pragma unroll
            for (int i = 0; i < 8; ++i) { acc[i] = fmaf(p, vcontrib[b][i], acc[i]); }
        }
    }
#pragma unroll
    for (int r = 0; r < 8; ++r) { acc[r] /= l; }
    if (lane == 0) {
        for (int r = 0; r < 8; ++r) { out[warp * kHeadDim + r] = acc[r]; }
    }
}

// ---- 臂 C: 归约批化 RB=4 ----
__global__ __launch_bounds__(256)
void attn_redbatch(const std::int8_t* __restrict__ kcache, const std::int8_t* __restrict__ vcache,
                   const __half* __restrict__ kscale, const __half* __restrict__ vscale,
                   const float* __restrict__ q, float* __restrict__ out, int n_keys, float scale) {
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp = static_cast<int>(blockIdx.x) * kWarps + (static_cast<int>(threadIdx.x) >> 5);
    if (warp >= kKVHeads) { return; }
    const std::int8_t* kbase = kcache + static_cast<std::int64_t>(warp) * kKeys * kHeadDim;
    const std::int8_t* vbase = vcache + static_cast<std::int64_t>(warp) * kKeys * kHeadDim;
    const __half* ks = kscale + static_cast<std::int64_t>(warp) * kKeys * 8;
    const __half* vs = vscale + static_cast<std::int64_t>(warp) * kKeys * 8;
    const float* qv = q + static_cast<std::int64_t>(warp) * kHeadDim;

    float acc[8];
#pragma unroll
    for (int r = 0; r < 8; ++r) { acc[r] = 0.0f; }
    float m = -INFINITY, l = 0.0f;

    for (int key = 0; key < n_keys; key += 4) {
        // 1) 4 key 的部分和并行累(无依赖)。
        float partials[4] = {0, 0, 0, 0};
        float vcontrib[4][8];
        int live = 0;
        for (int b = 0; b < 4 && key + b < n_keys; ++b) {
            ++live;
            const int2 kk = *reinterpret_cast<const int2*>(kbase + (key + b) * kHeadDim + 8 * lane);
            const std::int8_t* kc = reinterpret_cast<const std::int8_t*>(&kk);
            const float ksc = __half2float(ks[(key + b) * 8 + (lane >> 3)]);
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                partials[b] = fmaf(qv[8 * lane + i], static_cast<float>(kc[i]) * ksc, partials[b]);
            }
        }
        for (int b = 0; b < live; ++b) {
            const int2 vv = *reinterpret_cast<const int2*>(vbase + (key + b) * kHeadDim + 8 * lane);
            const std::int8_t* vc = reinterpret_cast<const std::int8_t*>(&vv);
            const float vsc = __half2float(vs[(key + b) * 8 + (lane >> 3)]);
#pragma unroll
            for (int i = 0; i < 8; ++i) { vcontrib[b][i] = static_cast<float>(vc[i]) * vsc; }
        }
        // 2) 一次多值归约: 2 条 shfl 链走出 4 个全和(每 lane 全和四元组的 4 个分量
        //    在 4 个 lane 组间错位, 由 xor-1/xor-2 矩阵配对; 每分量的 32-lane 全和
        //    需再走一条 5-shfl 全归约, 见下)。
        // 简化正确实现: 4 值打包进 2 次 5-shfl 全归约(用 lane 奇偶分相)。
        // lane 偶数持 v0/v2 的贡献, 奇数持 v1/v3 —— 打包 fp32 对不可行, 改用
        // 两遍全归约 + 轮转: 第一遍归 (v0+v1 的半和), 第二遍归 (v2+v3 的半和),
        // 每半和内部再由 xor-1 拆回 —— 共 2×5 + 2 = 12 条 shfl(对比独立 4×5=20)。
        const float even_sum = (lane & 1) ? partials[1] : partials[0];
        const float odd_pair = (lane & 1) ? partials[3] : partials[2];
        const float half0 = warp_sum1(even_sum);  // = sum(v0) 与 sum(v1) 的混合...
        const float half1 = warp_sum1(odd_pair);
        (void)half0; (void)half1;
        // 上述打包在浮点加法下无法无损拆分 —— 归约批化的正确路径是 4 个部分和
        // 经 4 条独立链, 但链可以**重叠发射**(ILP): 4 条 warp_sum1 的 20 条 shfl
        // 之间无数据依赖, 编译器/调度器可流水。真正的批化收益 = 依赖链从
        // [sum→softmax→acc]×4 串行变为 [4×sum 并行]→[softmax×4 独立]→[acc 合并]。
        float scs[4];
#pragma unroll
        for (int b = 0; b < 4; ++b) { scs[b] = warp_sum1(partials[b]) * scale; }
        // 3) 批量 softmax 重放: 批内先定批最大值, 4 个 sc 一次重放(批间 m/l 仍串行,
        //    但批内 4 次更新合并为 1 次 m/l 写)。
        float bmax = -INFINITY;
#pragma unroll
        for (int b = 0; b < 4; ++b) { bmax = fmaxf(bmax, scs[b]); }
        const float nm = fmaxf(m, bmax);
        const float a = (m == -INFINITY) ? 0.0f : exp2f((m - nm) * 1.4426950408889634f);
        float ps[4], psum = 0.0f;
#pragma unroll
        for (int b = 0; b < 4; ++b) {
            ps[b] = (b < live) ? exp2f((scs[b] - nm) * 1.4426950408889634f) : 0.0f;
            psum += ps[b];
        }
#pragma unroll
        for (int r = 0; r < 8; ++r) { acc[r] *= a; }
        l = l * a + psum;
        m = nm;
        // 4) V 累加: 4 行并行(无依赖), 与 2) 的 ILP 合并覆盖整段。
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            float vi = 0.0f;
#pragma unroll
            for (int b = 0; b < 4; ++b) { vi = fmaf(ps[b], vcontrib[b][i], vi); }
            acc[i] += vi;
        }
    }
#pragma unroll
    for (int r = 0; r < 8; ++r) { acc[r] /= l; }
    if (lane == 0) {
        for (int r = 0; r < 8; ++r) { out[warp * kHeadDim + r] = acc[r]; }
    }
}

void run_case(int n_keys, int iters) {
    std::mt19937 rng(7);
    std::normal_distribution<float> gauss(0.0f, 1.0f);
    std::vector<std::int8_t> kc(static_cast<std::size_t>(kKVHeads) * kKeys * kHeadDim);
    std::vector<std::int8_t> vc(kc.size());
    for (auto& v : kc) { v = static_cast<std::int8_t>(rng() % 61 - 30); }
    for (auto& v : vc) { v = static_cast<std::int8_t>(rng() % 61 - 30); }
    std::vector<__half> ks(static_cast<std::size_t>(kKVHeads) * kKeys * 8);
    std::vector<__half> vs(ks.size());
    for (auto& h : ks) { h = __float2half(0.02f + 0.02f * std::fabs(gauss(rng))); }
    for (auto& h : vs) { h = __float2half(0.02f + 0.02f * std::fabs(gauss(rng))); }
    std::vector<float> q(static_cast<std::size_t>(kKVHeads) * kHeadDim);
    for (auto& f : q) { f = gauss(rng); }

    std::int8_t *dkc, *dvc;
    __half *dks, *dvs;
    float *dq, *da, *db, *dc;
    HIP_CHECK(hipMalloc(&dkc, kc.size()));
    HIP_CHECK(hipMalloc(&dvc, vc.size()));
    HIP_CHECK(hipMalloc(&dks, ks.size() * 2));
    HIP_CHECK(hipMalloc(&dvs, vs.size() * 2));
    HIP_CHECK(hipMalloc(&dq, q.size() * 4));
    HIP_CHECK(hipMalloc(&da, static_cast<std::size_t>(kKVHeads) * kHeadDim * 4));
    HIP_CHECK(hipMalloc(&db, static_cast<std::size_t>(kKVHeads) * kHeadDim * 4));
    HIP_CHECK(hipMalloc(&dc, static_cast<std::size_t>(kKVHeads) * kHeadDim * 4));
    HIP_CHECK(hipMemcpy(dkc, kc.data(), kc.size(), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dvc, vc.data(), vc.size(), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dks, ks.data(), ks.size() * 2, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dvs, vs.data(), vs.size() * 2, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dq, q.data(), q.size() * 4, hipMemcpyHostToDevice));

    const dim3 grid((kKVHeads + kWarps - 1) / kWarps), block(kWarps * 32);
    const float scale = 0.125f;

    auto bench = [&](auto kern, float* dout, const char* tag) {
        for (int w = 0; w < 3; ++w) { kern<<<grid, block>>>(dkc, dvc, dks, dvs, dq, dout, n_keys, scale); }
        HIP_CHECK(hipDeviceSynchronize());
        hipEvent_t e0, e1;
        HIP_CHECK(hipEventCreate(&e0));
        HIP_CHECK(hipEventCreate(&e1));
        HIP_CHECK(hipEventRecord(e0));
        for (int i = 0; i < iters; ++i) { kern<<<grid, block>>>(dkc, dvc, dks, dvs, dq, dout, n_keys, scale); }
        HIP_CHECK(hipEventRecord(e1));
        HIP_CHECK(hipEventSynchronize(e1));
        float ms = 0.0f;
        HIP_CHECK(hipEventElapsedTime(&ms, e0, e1));
        HIP_CHECK(hipEventDestroy(e0));
        HIP_CHECK(hipEventDestroy(e1));
        std::printf("  %s: %.4f ms/iter (%d keys)\n", tag, ms / iters, n_keys);
        return ms / iters;
    };

    const float ms_a = bench(attn_perkey, da, "A per-key ");
    const float ms_b = bench(attn_mk4, db, "B mk4     ");
    const float ms_c = bench(attn_redbatch, dc, "C redbatch");

    // 数值门: C/A 与 B/A 逐元素相对误差。
    std::vector<float> ha(static_cast<std::size_t>(kKVHeads) * kHeadDim);
    std::vector<float> hb(ha.size()), hc(ha.size());
    HIP_CHECK(hipMemcpy(ha.data(), da, ha.size() * 4, hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(hb.data(), db, ha.size() * 4, hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(hc.data(), dc, ha.size() * 4, hipMemcpyDeviceToHost));
    double worst_b = 0.0, worst_c = 0.0;
    for (std::size_t i = 0; i < ha.size(); ++i) {
        worst_b = std::max(worst_b, std::fabs(hb[i] - ha[i]) / (std::fabs(ha[i]) + 1e-9));
        worst_c = std::max(worst_c, std::fabs(hc[i] - ha[i]) / (std::fabs(ha[i]) + 1e-9));
    }
    std::printf("  NUM n=%d B_rel=%.3e C_rel=%.3e | C/A=%.2fx B/A=%.2fx\n",
                n_keys, worst_b, worst_c, ms_a / ms_c, ms_a / ms_b);

    HIP_CHECK(hipFree(dkc)); HIP_CHECK(hipFree(dvc)); HIP_CHECK(hipFree(dks));
    HIP_CHECK(hipFree(dvs)); HIP_CHECK(hipFree(dq));
    HIP_CHECK(hipFree(da)); HIP_CHECK(hipFree(db)); HIP_CHECK(hipFree(dc));
}

}  // namespace

int main() {
    std::printf("== attention keys 归约批化探针 (gfx1100)\n");
    run_case(256, 50);
    run_case(1024, 50);
    return 0;
}

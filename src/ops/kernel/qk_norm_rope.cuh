#pragma once

// megakernel M1b 刀一(2026-10-08): attention prologue 三合一 —— rmsnorm(q) +
// rmsnorm(k) + rope(qn, kn) 单核完成(Text1D, QHeads=24, KHeads=4, HeadDim=256,
// Half=32 = qwen3.6-27B decode 的实际调用形态)。每注意力层省 2 次发射 ×16 层。
//
// 位一致契约(vs rmsnorm_warp_bf16x2 + rope_fixed 3 发射路径):
//   * rmsnorm: warp-per-head、lane 对 = lane + k*32、求和顺序与 rsqrtf 表达式同
//     rmsnorm_warp_bf16x2_kernel(d=256 → warp 档);
//   * 归一化结果先按 __floats2bfloat162_rn 舍入成 bf16 再进 rope —— 复刻 3 发射
//     路径"写全局 → rope 读全局"的 bf16 往返舍入;
//   * rope: 复刻 rope_fixed_kernel<Text1D,24,4> 的交错半旋(pairs 0..31 = dims
//     0..63),角度 = fixed_sincos<Text1D>(positions, pair),lane 的 c0/c1/s0/s1
//     = cache[lane*2]/[lane*2+1],pair lane ↔ pair lane+16 交换(shfl_xor 16);
//   * qn/kn 落盘为 attention 核的直接输入,布局 [head_dim, heads, tokens] 不变。

#include "ops/common/warp.cuh"
#include "ops/kernel/rmsnorm.cuh"
#include "ops/kernel/rope.cuh"

#include <cuda_bf16.h>

#include <cstdint>

namespace ninfer::ops {

// Grid = tokens(one CTA per token), Block = 1024(32 warp:0..23 = q 头,24..27 = k 头,
// 28..31 空转但不越过 __syncthreads)。
__global__ __launch_bounds__(1024) void qk_norm_rope_kernel(
    const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ q_norm_w, const __nv_bfloat16* __restrict__ k_norm_w,
    float eps, const std::int32_t* __restrict__ positions, __nv_bfloat16* __restrict__ qn,
    __nv_bfloat16* __restrict__ kn, std::int32_t tokens) {
    constexpr int kQHeads  = 24;
    constexpr int kKHeads  = 4;
    constexpr int kHeadDim = 256;
    constexpr int kHalf    = 32;            // rope 交错对数(rotary_dim 64)
    constexpr int kPairs   = kHeadDim / 2;  // 128 bf16x2 对/头
    constexpr int kWarpSize = 32;

    const int token = blockIdx.x;
    const int warp  = static_cast<int>(threadIdx.x) >> 5;
    const int lane  = static_cast<int>(threadIdx.x) & (kWarpSize - 1);

    __shared__ float sin_cache[kHalf];
    __shared__ float cos_cache[kHalf];
    if (threadIdx.x < kHalf) {
        fixed_sincos<RopeKernelMode::Text1D>(positions, tokens, token,
                                             static_cast<int>(threadIdx.x),
                                             &sin_cache[threadIdx.x], &cos_cache[threadIdx.x]);
    }
    __syncthreads();

    const bool is_q = warp < kQHeads;
    const int head  = is_q ? warp : warp - kQHeads;
    if (warp >= kQHeads + kKHeads) { return; }

    // 布局 [head_dim, heads, tokens]:q 的 token 步长 = n_q*256,k 的 = n_kv*256
    // (GQA 头数不同!2026-10-08 修:k 步长误用 q 步长致 prefill t>=1 的 k 行越界读)。
    const int heads = is_q ? kQHeads : kKHeads;
    const std::int64_t token_stride = static_cast<std::int64_t>(heads) * kHeadDim;
    const __nv_bfloat162* __restrict__ src2 =
        reinterpret_cast<const __nv_bfloat162*>((is_q ? q : k) +
                                                static_cast<std::int64_t>(head) * kHeadDim +
                                                static_cast<std::int64_t>(token) * token_stride);
    const __nv_bfloat162* __restrict__ w2 =
        reinterpret_cast<const __nv_bfloat162*>(is_q ? q_norm_w : k_norm_w);
    __nv_bfloat162* __restrict__ dst2 = reinterpret_cast<__nv_bfloat162*>(
        (is_q ? qn : kn) + static_cast<std::int64_t>(head) * kHeadDim +
        static_cast<std::int64_t>(token) * token_stride);

    // rmsnorm(lane 对 = lane + k*32,求和序与 rmsnorm_warp_bf16x2 一致)
    __nv_bfloat162 values[4];
    float sum = 0.0f;
#pragma unroll
    for (int kk = 0; kk < 4; ++kk) {
        const int pair  = lane + kk * kWarpSize;
        values[kk]      = src2[pair];
        const float2 xf = __bfloat1622float2(values[kk]);
        sum += xf.x * xf.x + xf.y * xf.y;
    }
    sum       = warp_reduce_sum(sum);
    float inv = lane == 0 ? rsqrtf(sum / static_cast<float>(kHeadDim) + eps) : 0.0f;
    inv       = __shfl_sync(kFullWarpMask, inv, 0);

    // 归一化 × (weight + 1.0) —— 调用点 unit_offset=true(rmsnorm_epilogue<Offset>:
    // weight += 1.0f);2026-10-09 修:漏此语义致输出系统性偏差(w/(w+1) 逐维散布)。
    // 舍入 bf16 后落盘;k=0 槽(values[0])= rope 的 pair lane
#pragma unroll
    for (int kk = 0; kk < 4; ++kk) {
        const int pair  = lane + kk * kWarpSize;
        const float2 xf = __bfloat1622float2(values[kk]);
        const float2 wf = __bfloat1622float2(w2[pair]);
        values[kk] = __floats2bfloat162_rn(xf.x * inv * (wf.x + 1.0f),
                                           xf.y * inv * (wf.y + 1.0f));
        dst2[pair] = values[kk];
    }
    __syncwarp();

    // rope(交错半旋,pairs 0..31):lane<16 持 c/s,与 lane+16 交换配对
    float c0 = 0.F, c1 = 0.F, s0 = 0.F, s1 = 0.F;
    if (lane < kHalf / 2) {
        c0 = cos_cache[lane * 2];
        c1 = cos_cache[lane * 2 + 1];
        s0 = sin_cache[lane * 2];
        s1 = sin_cache[lane * 2 + 1];
    }
    const unsigned int bits0  = *reinterpret_cast<unsigned int*>(&values[0]);
    const unsigned int bits16 = __shfl_xor_sync(kFullWarpMask, bits0, 16);
    const __nv_bfloat162 first  = values[0];
    __nv_bfloat162       second = *reinterpret_cast<const __nv_bfloat162*>(&bits16);
    const float2 f2 = __bfloat1622float2(first);
    const float2 s2 = __bfloat1622float2(second);
    if (lane < kHalf / 2) {
        const __nv_bfloat162 new_first =
            __floats2bfloat162_rn(f2.x * c0 - s2.x * s0, f2.y * c1 - s2.y * s1);
        const __nv_bfloat162 new_second =
            __floats2bfloat162_rn(s2.x * c0 + f2.x * s0, s2.y * c1 + f2.y * s1);
        dst2[lane]      = new_first;
        dst2[lane + 16] = new_second;
    }
}

} // namespace ninfer::ops

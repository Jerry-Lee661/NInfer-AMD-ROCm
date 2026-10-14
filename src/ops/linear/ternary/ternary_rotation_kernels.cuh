#pragma once

// Device kernels for the folded (rotated-basis) ternary transform.
//
// Split out of ternary_rotation.cu so a standalone nvcc harness can compile and exercise the
// REAL kernels against a numpy oracle -- the transform is the one piece of this port that cannot
// be validated by byte or size checks, and a wrong sign row, a wrong register span or a wrong
// permutation all produce fluent-looking nonsense rather than an error.
//
// This header is self-contained on purpose: it includes only cuda_runtime.h, cuda_bf16.h and the
// in-tree D256 building block (which itself only needs cuda_runtime.h), so the harness compiles it
// without dragging in Tensor/Weight/arena.

#include "ops/kv_cache/hadamard_d256.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace ninfer::ops::detail {
namespace {

constexpr int kBlockSize      = 1024;
constexpr int kWarpsPerBlock  = 8;
constexpr int kThreadsPerWarp = 32;

// D1024 = 32 lanes x 32 registers: normalized_hadamard_d1024_inplace 已提升到公共头
// (ops/kv_cache/hadamard_d256.cuh,2026-10-09 M2e 单一事实源),此处直接用。

// One warp owns one (1024-block, token) pair of the [k, tokens] activation. Lane l carries
// dimensions l + 32*r of the block in values[r].
//
// LAYOUT: ninfer follows the ggml convention where ne[0] is the CONTIGUOUS axis, so an activation
// tensor with ne = (k, tokens) stores element (column, token) at token * k + column -- TOKEN-major,
// not row-major. Applying the transform with a k * tokens + t stride scrambles every prefill while
// still passing any test at tokens == 1, where the two layouts coincide exactly.
//
// Forward (the folded weights read the activation):
//     out = H * (s * (P * x))
// The permutation P is a pure index permutation and is folded into the load. Given the post-P
// column j, the source column is recovered as
//     hd = j % perm_hd, q = j / perm_hd, nk = q / perm_rep, rep = q % perm_rep
//     i  = hd + perm_hd * nk + perm_hd * perm_nk * rep
// which is the inverse of llama-graph.cpp's reshape(perm_hd, perm_nk, perm_rep) + permute(0,2,1).
__global__ void ternary_rotate_bf16_kernel(const __nv_bfloat16* __restrict__ x,
                                           __nv_bfloat16* __restrict__ out,
                                           const float* __restrict__ signs, int n_blk, int k,
                                           int tokens, int perm_hd, int perm_nk, int perm_rep,
                                           int inverse) {
    const int lane   = threadIdx.x & (kThreadsPerWarp - 1);
    const int warp   = static_cast<int>(blockIdx.x) * kWarpsPerBlock + (threadIdx.x >> 5);
    const int blocks = k >> 10;
    if (warp >= blocks * tokens) { return; }
    const int block = warp % blocks;
    const int token = warp / blocks;

    const float* const signs_row  = signs + (block % n_blk) * kBlockSize;
    const bool permuted           = !inverse && perm_rep > 1;
    const __nv_bfloat16* x_token  = x + static_cast<std::int64_t>(token) * k;
    __nv_bfloat16* out_token      = out + static_cast<std::int64_t>(token) * k;

    float values[32];

    if (inverse) {
#pragma unroll
        for (int r = 0; r < 32; ++r) {
            values[r] = __bfloat162float(x_token[(block << 10) + r * 32 + lane]);
        }
    } else {
#pragma unroll
        for (int r = 0; r < 32; ++r) {
            const int column = (block << 10) + r * 32 + lane;
            int source       = column;
            if (permuted) {
                const int hd = column % perm_hd;
                const int q  = column / perm_hd;
                source = hd + perm_hd * (q / perm_rep) + perm_hd * perm_nk * (q % perm_rep);
            }
            const float value = __bfloat162float(x_token[source]);
            values[r]         = __fmul_rn(value, signs_row[r * 32 + lane]);
        }
    }

    normalized_hadamard_d1024_inplace(values, lane);

    if (inverse) {
#pragma unroll
        for (int r = 0; r < 32; ++r) {
            const float value = __fmul_rn(values[r], signs_row[r * 32 + lane]);
            out_token[(block << 10) + r * 32 + lane] = __float2bfloat16_rn(value);
        }
    } else {
#pragma unroll
        for (int r = 0; r < 32; ++r) {
            out_token[(block << 10) + r * 32 + lane] = __float2bfloat16_rn(values[r]);
        }
    }
}

// 旋转 + q8_1 量化融合核(2026-10-08, megakernel M2)。
//
// 与 ternary_rotate_bf16_kernel 同一线程映射(一个 warp 一个 (1024-block, token), lane l 持维度
// l+32r 于 values[r]),但把 Hadamard 留在寄存器里直接量化成 dp4a 的 q8_1 激活(每 32 元素一组:
// int8 码 + fp16 scale),省掉中间 bf16 缓冲的写+读与一次 launch(每步 257 个 rotate 节点)。
//
// 分组几何:量化组 g = block*32 + r 的 32 个元素(全局列 g*32 .. g*32+31)恰为 values[r] 的 32 个
// lane,所以 absmax 是"跨 lane、按 r"的 warp 归约,而不是 quant 核里的行内串行扫描。
//
// 数值契约:与"先跑 rotate 落 bf16、再跑 ternary_pq2_dp4a_quant_kernel"逐位一致 —— 旋转结果
// 同样过一遍 __float2bfloat16_rn 再取 absmax/量化, 量化表达式(amax/127、lrint、clamp)逐字照抄。
// 任何一位不同都说明映射或舍入点错了, 所以 A/B 用 md5 逐位判。
__global__ __launch_bounds__(kWarpsPerBlock * kThreadsPerWarp)
void ternary_rotate_quant_q8_kernel(const __nv_bfloat16* __restrict__ x,
                                    std::int8_t* __restrict__ xq,
                                    std::uint16_t* __restrict__ ds8,
                                    const float* __restrict__ signs, int n_blk, int k, int tokens,
                                    int perm_hd, int perm_nk, int perm_rep) {
    const int lane   = threadIdx.x & (kThreadsPerWarp - 1);
    const int warp   = static_cast<int>(blockIdx.x) * kWarpsPerBlock + (threadIdx.x >> 5);
    const int blocks = k >> 10;
    if (warp >= blocks * tokens) { return; }
    const int block = warp % blocks;
    const int token = warp / blocks;

    const float* const signs_row = signs + (block % n_blk) * kBlockSize;
    const bool permuted          = perm_rep > 1;
    const __nv_bfloat16* x_token = x + static_cast<std::int64_t>(token) * k;

    float values[32];
#pragma unroll
    for (int r = 0; r < 32; ++r) {
        const int column = (block << 10) + r * 32 + lane;
        int source       = column;
        if (permuted) {
            const int hd = column % perm_hd;
            const int q  = column / perm_hd;
            source = hd + perm_hd * (q / perm_rep) + perm_hd * perm_nk * (q % perm_rep);
        }
        values[r] = __fmul_rn(__bfloat162float(x_token[source]), signs_row[r * 32 + lane]);
    }

    normalized_hadamard_d1024_inplace(values, lane);

#pragma unroll
    for (int r = 0; r < 32; ++r) {
        // 两核路径在这里把 bf16 写回显存再读回来, 融合版就地过同一舍入点。
        values[r] = __bfloat162float(__float2bfloat16_rn(values[r]));
    }

    // 4 个 8 组块: 一次只留 8 个 amax 寄存器(32 个会顶到溢出区)。
    std::int8_t* const xq_token   = xq + static_cast<std::int64_t>(token) * k;
    std::uint16_t* const ds_token = ds8 + static_cast<std::int64_t>(token) * (k >> 5);
    // 分组 absmax v2(2026-10-09,LDS 转置替代 butterfly):v1 用 4 chunk × 5 stride ×
    // 8 reg 的 shuffle 蝶形(160 shfl + 160 max 的延迟链,小网格上 ~+8µs/发,导致 M2 刀一
    // 判负)。v2 = 一次 32×33 LDS 转置:lane i 写 values[g] → LDS[g][i],sync 后每 lane
    // 认领一个组(lane l = 组 l),串行读 32 元素求 max 再量化 —— 32 st + 32 ld + 1 sync
    // + 31 max,无 shuffle。写 [g][lane] 与读 [lane][i] 步距 33 words,均无 bank 冲突。
    // 数值不变(absmax 可交换;量化表达式逐字未动)。
    __shared__ float lds_t[8][32][33];  // [warp][element i][group g](pad 33 防 bank 冲突)
    float* const lw = &lds_t[threadIdx.x >> 5][0][0];
#pragma unroll
    for (int g = 0; g < 32; ++g) { lw[g * 33 + lane] = values[g]; }
    __syncwarp();
    float amax = 0.0f;
#pragma unroll
    for (int i = 0; i < 32; ++i) { amax = fmaxf(amax, fabsf(lw[lane * 33 + i])); }
    const float d   = amax / 127.0f;
    const float inv = d > 0.0f ? 1.0f / d : 0.0f;
    ds_token[(block << 5) + lane] = __half_as_ushort(__float2half(d));
#pragma unroll
    for (int i = 0; i < 32; ++i) {
        const int q = std::lrint(lw[lane * 33 + i] * inv);
        xq_token[(block << 10) + lane * 32 + i] =
            static_cast<std::int8_t>(std::max(-127, std::min(127, q)));
    }
}

// v3(2026-10-09, M3a): 4 warp 合作一个 1024 块(每 warp 一个 256 切片),并行度 ×4
// (k=5120: 5→20 warps;17408: 17→68)。Sylvester 阶段序与 v2 完全一致:
// bits 0-7 局部(normalized_hadamard_d256_inplace = lane stages + spans 1,2,4),
// bits 8,9 跨切片经 LDS 交接(span 式 ±,low/high 角色按 q 的位)—— 逐位一致由构造保证。
// CTA = 256 线程 = 8 warp = 2 块(每块 4 warp);grid = (ceil(blocks/2), tokens)。
// 每 lane 8 维(qbase = q*256 + lane + 32r);量化组 = warp 内 8 组(absmax 转置 8×33)。
// 注意:非活跃 warp(block 越界)不得提前 return —— __syncthreads 是块级,必须全体到达。
__global__ __launch_bounds__(256) void ternary_rotate_quant_q8_v3_kernel(
    const __nv_bfloat16* __restrict__ x, std::int8_t* __restrict__ xq,
    std::uint16_t* __restrict__ ds8, const float* __restrict__ signs, int n_blk, int k,
    int tokens, int perm_hd, int perm_nk, int perm_rep) {
    const int lane    = threadIdx.x & 31;
    const int warp_q  = (threadIdx.x >> 5) & 3;         // 块内切片号
    const int blk_lcl = (threadIdx.x >> 5) >> 2;        // CTA 内块号(0..1)
    const int blocks  = k >> 10;
    const int block   = static_cast<int>(blockIdx.x) * 2 + blk_lcl;
    const int token   = static_cast<int>(blockIdx.y);
    const bool active = block < blocks && token < tokens;

    const float* const signs_row = signs + (block % n_blk) * kBlockSize;
    const bool permuted          = perm_rep > 1;
    const __nv_bfloat16* x_token = x + static_cast<std::int64_t>(token) * k;
    std::int8_t* const xq_token  = xq + static_cast<std::int64_t>(token) * k;
    std::uint16_t* const ds_token = ds8 + static_cast<std::int64_t>(token) * (k >> 5);
    const int qbase              = warp_q * 256;

    __shared__ float ex[8][256];      // stage-2 交接:[warp][切片内维]
    __shared__ float lds_t[8][8][33]; // absmax 转置:[warp][组内 r][lane](pad 33)

    float values[8];
    if (active) {
#pragma unroll
        for (int r = 0; r < 8; ++r) {
            const int column = (block << 10) + qbase + r * 32 + lane;
            int source       = column;
            if (permuted) {
                const int hd = column % perm_hd;
                const int q  = column / perm_hd;
                source = hd + perm_hd * (q / perm_rep) + perm_hd * perm_nk * (q % perm_rep);
            }
            values[r] = __fmul_rn(__bfloat162float(x_token[source]),
                                  signs_row[qbase + r * 32 + lane]);
        }
        normalized_hadamard_d256_inplace(values, lane);
#pragma unroll
        for (int r = 0; r < 8; ++r) { ex[threadIdx.x >> 5][lane + 32 * r] = values[r]; }
    }
    __syncthreads();

    // stage bit8:切片对 (0,2) / (1,3);low(q bit=0)= own+partner,high = own−partner。
    // stage bit9:切片对 (0,1) / (2,3) 作用于 stage8 结果。
    if (active) {
        const float* own    = ex[threadIdx.x >> 5];
        const float* partner = ex[(threadIdx.x >> 5) ^ 2];  // 同 CTA 同块的 partner warp(块 0 之外 warp_in_cta ≠ warp_q)
        const int pb = (warp_q & 2) ? 1 : 0; // high 切片 = low − high = partner − own
#pragma unroll
        for (int r = 0; r < 8; ++r) {
            values[r] = pb ? __fsub_rn(partner[lane + 32 * r], own[lane + 32 * r])
                           : __fadd_rn(own[lane + 32 * r], partner[lane + 32 * r]);
        }
        float* u = ex[threadIdx.x >> 5];
#pragma unroll
        for (int r = 0; r < 8; ++r) { u[lane + 32 * r] = values[r]; }
    }
    __syncthreads();

    if (active) {
        const float* own     = ex[threadIdx.x >> 5];
        const float* partner = ex[(threadIdx.x >> 5) ^ 1];
        const int pb         = (warp_q & 1) ? 1 : 0;
#pragma unroll
        for (int r = 0; r < 8; ++r) {
            values[r] = pb ? __fsub_rn(partner[lane + 32 * r], own[lane + 32 * r])
                           : __fadd_rn(own[lane + 32 * r], partner[lane + 32 * r]);
        }
        // 归一化收尾:D256 自带 2^-4(覆盖其 8 阶段),跨 warp 的 2 个 span 阶段还需
        // ×2^-1 —— D1024 总归一 = 2^-5(v2 末尾一次性乘)。漏掉 = 输出恰大 2×
        // (codes 是尺度不变的比值所以看不出来,scale 全组 ×2 暴露)。
#pragma unroll
        for (int r = 0; r < 8; ++r) {
            values[r] = __fmul_rn(values[r], 0x1p-1f);
        }
        // 两核路径舍入点:全部 Hadamard 阶段结束才落 bf16(v3 首版把舍入放 stage 交换前
        // = 跨 warp 阶段吃到双重舍入 ⇒ 320 scale 全错的教训)。
#pragma unroll
        for (int r = 0; r < 8; ++r) {
            values[r] = __bfloat162float(__float2bfloat16_rn(values[r]));
        }
        // 量化:warp 的 8 组(组 = 切片内 32 连续维,组号 = block*32 + q*8 + r)。
        // lane<8 各认领一组。⚠️ 暂存与 __syncwarp 必须全 warp 执行(分歧屏障 = UB,坑 #5);
        // 只有 absmax/量化/写回收敛在 lane<8 分支内。
        float* lw = &lds_t[threadIdx.x >> 5][0][0];
#pragma unroll
        for (int r = 0; r < 8; ++r) { lw[r * 33 + lane] = values[r]; }
        __syncwarp();
        if (lane < 8) {
            const int g = lane;
            float amax = 0.0f;
#pragma unroll
            for (int i = 0; i < 32; ++i) { amax = fmaxf(amax, fabsf(lw[g * 33 + i])); }
            const float d   = amax / 127.0f;
            const float inv = d > 0.0f ? 1.0f / d : 0.0f;
            ds_token[(block << 5) + warp_q * 8 + g] = __half_as_ushort(__float2half(d));
#pragma unroll
            for (int i = 0; i < 32; ++i) {
                const int qv = std::lrint(lw[g * 33 + i] * inv);
                xq_token[(block << 10) + (warp_q * 8 + g) * 32 + i] =
                    static_cast<std::int8_t>(std::max(-127, std::min(127, qv)));
            }
        }
    }
}

// Inverse mapping of a freshly gathered embedding, in place, over the layout the embedding op
// ACTUALLY writes.
//
// Every embed_gather_* kernel in this tree indexes `out + token * hidden + k` -- the output buffer
// is TOKEN-major, even though the tensor's extents read [hidden, T]. Transforming it with the
// [hidden, T] stride (k * T + t) would scramble every prefill: at T == 1 the two layouts coincide
// exactly, so a decode-only test cannot see the difference, and a whole prefill window comes out
// as noise (which reads as a perplexity close to uniform).
//
// Single pointer, so there is no restrict-aliasing contract to violate: each warp loads its 32
// registers, transforms them, and writes the same 32 slots back.
__global__ void ternary_rotate_inverse_inplace_bf16_kernel(__nv_bfloat16* __restrict__ data,
                                                           const float* __restrict__ signs,
                                                           int n_blk, int k, int tokens) {
    const int lane   = threadIdx.x & (kThreadsPerWarp - 1);
    const int warp   = static_cast<int>(blockIdx.x) * kWarpsPerBlock + (threadIdx.x >> 5);
    const int blocks = k >> 10;
    if (warp >= blocks * tokens) { return; }
    const int block = warp % blocks;
    const int token = warp / blocks;

    const float* const signs_row = signs + (block % n_blk) * kBlockSize;
    __nv_bfloat16* const base    = data + static_cast<std::int64_t>(token) * k;

    float values[32];
#pragma unroll
    for (int r = 0; r < 32; ++r) {
        values[r] = __bfloat162float(base[(block << 10) + r * 32 + lane]);
    }

    normalized_hadamard_d1024_inplace(values, lane);

#pragma unroll
    for (int r = 0; r < 32; ++r) {
        const float value = __fmul_rn(values[r], signs_row[r * 32 + lane]);
        base[(block << 10) + r * 32 + lane] = __float2bfloat16_rn(value);
    }
}

} // namespace
} // namespace ninfer::ops::detail

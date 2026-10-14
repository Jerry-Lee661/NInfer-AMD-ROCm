#pragma once

// ninfer::ops — silu_mul kernel: out = silu(gate) * up, elementwise.
// silu(x) = x / (1 + e^-x), computed exactly in fp32 (NOT a polynomial fit).
// Vectorized over bf16 pairs; included only by its launcher. See
// docs/op-development.md §6 (no math approximation).

#include "ops/common/math.cuh"
#include "ops/kv_cache/hadamard_d256.cuh"

#include <cuda_bf16.h>

#include <cstdint>

namespace ninfer::ops {

inline constexpr int kSiluAndMulPairsPerThread = 4;

__device__ __forceinline__ __nv_bfloat162 silu_mul_pair(__nv_bfloat162 g, __nv_bfloat162 u) {
    const float r0 = silu(__low2float(g)) * __low2float(u);
    const float r1 = silu(__high2float(g)) * __high2float(u);
    return __floats2bfloat162_rn(r0, r1);
}

__global__ void silu_and_mul_scalar_kernel(const __nv_bfloat16* gate, const __nv_bfloat16* up,
                                           __nv_bfloat16* out, std::int64_t n) {
    const std::int64_t start  = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x;
    const std::int64_t stride = static_cast<std::int64_t>(gridDim.x) * blockDim.x;
    for (std::int64_t i = start; i < n; i += stride) {
        out[i] = __float2bfloat16(silu(__bfloat162float(gate[i])) * __bfloat162float(up[i]));
    }
}

// silu_mul + Hadamard 旋转融合(2026-10-09, megakernel M2e): MLP down 投影的激活本来要
// 走 silu_and_mul(落 bf16)→ ternary_rotate_bf16(读回→符号乘→Hadamard→落 bf16)两核;
// 本核在寄存器里串起来,省一次 launch(-1 节点/层)与激活缓冲的写+读。
// 线程映射 = 旋转核的规范布局(warp 一个 (1024-block, token),lane l 持维度 l+32r);
// 数值逐位一致契约:silu 数学/舍入点逐字照抄 silu_and_mul_kernel(同 silu()、先 bf16_rn
// 落一次再读回),符号乘与 Hadamard 逐字照抄 ternary_rotate_bf16_kernel(公共头的
// normalized_hadamard_d1024_inplace)。gate/up 载荷为 lane 间连续(合并访存)。
// 置换 P(仅 GDN 输出携带)按旋转核同式支持;MLP down 实际 perm_rep=1。
__global__ __launch_bounds__(256) void silu_and_mul_rotate_kernel(
    const __nv_bfloat16* __restrict__ gate, const __nv_bfloat16* __restrict__ up,
    __nv_bfloat16* __restrict__ out, const float* __restrict__ signs, int n_blk, int k,
    int tokens, int perm_hd, int perm_nk, int perm_rep, std::int64_t gate_token_stride,
    std::int64_t up_token_stride, std::int64_t out_token_stride) {
    const int lane   = threadIdx.x & 31;
    const int warp   = static_cast<int>(blockIdx.x) * 8 + (threadIdx.x >> 5);
    const int blocks = k >> 10;
    if (warp >= blocks * tokens) { return; }
    const int block = warp % blocks;
    const int token = warp / blocks;

    const float* const signs_row = signs + (block % n_blk) * 1024;
    const bool permuted          = perm_rep > 1;
    // token 步长显式传入:gate/up 是 [2K,T] 父张量的 ne[0] 切片,保留父步长(34816),
    // is_contiguous() 在切片上为假,不能按整张量连续校验/索引。
    const __nv_bfloat16* g_token = gate + static_cast<std::int64_t>(token) * gate_token_stride;
    const __nv_bfloat16* u_token = up + static_cast<std::int64_t>(token) * up_token_stride;

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
        // 舍入点与 silu_and_mul_kernel 逐位一致:先 bf16_rn 落一次(silu 核的输出就是 bf16),
        // 旋转核读到的是这个 bf16。
        const float act = __bfloat162float(
            __float2bfloat16_rn(silu(__bfloat162float(g_token[source])) *
                                __bfloat162float(u_token[source])));
        values[r] = __fmul_rn(act, signs_row[r * 32 + lane]);
    }

    normalized_hadamard_d1024_inplace(values, lane);

#pragma unroll
    for (int r = 0; r < 32; ++r) {
        out[(static_cast<std::int64_t>(token) * out_token_stride) + (block << 10) + r * 32 + lane] =
            __float2bfloat16_rn(values[r]);
    }
}

__global__ void silu_and_mul_strided_input_kernel(
    const __nv_bfloat16* gate, const __nv_bfloat16* up, __nv_bfloat16* out, std::int64_t n,
    std::int32_t ne0, std::int32_t ne1, std::int32_t ne2, std::int64_t gnb0, std::int64_t gnb1,
    std::int64_t gnb2, std::int64_t gnb3, std::int64_t unb0, std::int64_t unb1, std::int64_t unb2,
    std::int64_t unb3) {
    const std::int64_t start  = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x;
    const std::int64_t stride = static_cast<std::int64_t>(gridDim.x) * blockDim.x;
    const auto* gate_bytes    = reinterpret_cast<const unsigned char*>(gate);
    const auto* up_bytes      = reinterpret_cast<const unsigned char*>(up);
    for (std::int64_t i = start; i < n; i += stride) {
        std::int64_t rem = i;
        const auto d0    = static_cast<std::int32_t>(rem % ne0);
        rem /= ne0;
        const auto d1 = static_cast<std::int32_t>(rem % ne1);
        rem /= ne1;
        const auto d2 = static_cast<std::int32_t>(rem % ne2);
        const auto d3 = static_cast<std::int32_t>(rem / ne2);

        const std::int64_t goff =
            static_cast<std::int64_t>(d0) * gnb0 + static_cast<std::int64_t>(d1) * gnb1 +
            static_cast<std::int64_t>(d2) * gnb2 + static_cast<std::int64_t>(d3) * gnb3;
        const std::int64_t uoff =
            static_cast<std::int64_t>(d0) * unb0 + static_cast<std::int64_t>(d1) * unb1 +
            static_cast<std::int64_t>(d2) * unb2 + static_cast<std::int64_t>(d3) * unb3;
        const auto gv = *reinterpret_cast<const __nv_bfloat16*>(gate_bytes + goff);
        const auto uv = *reinterpret_cast<const __nv_bfloat16*>(up_bytes + uoff);
        out[i]        = __float2bfloat16(silu(__bfloat162float(gv)) * __bfloat162float(uv));
    }
}

__launch_bounds__(256) __global__
    void silu_and_mul_dim0_split_kernel(const __nv_bfloat16* gate, const __nv_bfloat16* up,
                                        __nv_bfloat16* out, std::int32_t rows,
                                        std::int64_t gate_col_stride, std::int64_t up_col_stride) {
    const int col            = blockIdx.y;
    const std::int64_t pair0 = (blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x) *
                               kSiluAndMulPairsPerThread;
    const std::int64_t row_pairs = rows / 2;

    const auto* gate2 = reinterpret_cast<const __nv_bfloat162*>(gate + col * gate_col_stride);
    const auto* up2   = reinterpret_cast<const __nv_bfloat162*>(up + col * up_col_stride);
    auto* out2 = reinterpret_cast<__nv_bfloat162*>(out + col * static_cast<std::int64_t>(rows));

    for (int p = 0; p < kSiluAndMulPairsPerThread; ++p) {
        const std::int64_t j = pair0 + p;
        if (j < row_pairs) { out2[j] = silu_mul_pair(gate2[j], up2[j]); }
    }
}

__launch_bounds__(256) __global__
    void silu_and_mul_kernel(const __nv_bfloat16* gate, const __nv_bfloat16* up, __nv_bfloat16* out,
                             std::int64_t n) {
    const std::int64_t tid = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x;
    const std::int64_t stride =
        static_cast<std::int64_t>(gridDim.x) * blockDim.x * kSiluAndMulPairsPerThread;
    const std::int64_t n2 = n / 2;

    const auto* gate2 = reinterpret_cast<const __nv_bfloat162*>(gate);
    const auto* up2   = reinterpret_cast<const __nv_bfloat162*>(up);
    auto* out2        = reinterpret_cast<__nv_bfloat162*>(out);
    for (std::int64_t j = tid * kSiluAndMulPairsPerThread; j < n2; j += stride) {
        const __nv_bfloat162 g0 = gate2[j];
        const __nv_bfloat162 u0 = up2[j];
        if (j + 3 < n2) {
            const __nv_bfloat162 g1 = gate2[j + 1];
            const __nv_bfloat162 u1 = up2[j + 1];
            const __nv_bfloat162 g2 = gate2[j + 2];
            const __nv_bfloat162 u2 = up2[j + 2];
            const __nv_bfloat162 g3 = gate2[j + 3];
            const __nv_bfloat162 u3 = up2[j + 3];
            out2[j]                 = silu_mul_pair(g0, u0);
            out2[j + 1]             = silu_mul_pair(g1, u1);
            out2[j + 2]             = silu_mul_pair(g2, u2);
            out2[j + 3]             = silu_mul_pair(g3, u3);
        } else {
            out2[j] = silu_mul_pair(g0, u0);
            if (j + 1 < n2) { out2[j + 1] = silu_mul_pair(gate2[j + 1], up2[j + 1]); }
            if (j + 2 < n2) { out2[j + 2] = silu_mul_pair(gate2[j + 2], up2[j + 2]); }
        }
    }

    if (tid == 0) {
        if ((n & 1) != 0) {
            const std::int64_t i = n - 1;
            out[i] = __float2bfloat16(silu(__bfloat162float(gate[i])) * __bfloat162float(up[i]));
        }
    }
}

} // namespace ninfer::ops

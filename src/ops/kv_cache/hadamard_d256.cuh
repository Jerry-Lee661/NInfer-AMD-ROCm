#pragma once

#include <cuda_runtime.h>

namespace ninfer::ops {

// Apply the five lane dimensions of a Sylvester transform to independent columns. Keeping the
// column loop inside each stage exposes their independent shuffle instructions while preserving
// the butterfly and rounding order of every column.
template <int Columns>
__device__ __forceinline__ void hadamard_d32_columns_inplace(float (&values)[Columns], int lane) {
    constexpr unsigned long long FullMask = 0xffffffffffffffffULL;

#pragma unroll
    for (int stride = 1; stride <= 16; stride <<= 1) {
#pragma unroll
        for (int column = 0; column < Columns; ++column) {
            const float value = values[column];
            const float peer  = __shfl_xor_sync(FullMask, value, stride);
            values[column] = (lane & stride) == 0 ? __fadd_rn(value, peer) : __fsub_rn(peer, value);
        }
    }
}

// Factorized H256 support for latency-oriented kernels that distribute one D256 row across four
// G64 warps. The H64 fragments are intentionally unnormalized; the final H4 leaf applies 2^-4.
__device__ __forceinline__ void hadamard_d64_fragment_inplace(float (&values)[2], int lane) {
    hadamard_d32_columns_inplace(values, lane);
    const float low  = values[0];
    const float high = values[1];
    values[0]        = __fadd_rn(low, high);
    values[1]        = __fsub_rn(low, high);
}

__device__ __forceinline__ float
normalized_hadamard_d256_group_value_from_h64(float x0, float x1, float x2, float x3, int group) {
    const float y0  = (group & 1) == 0 ? __fadd_rn(x0, x1) : __fsub_rn(x0, x1);
    const float y1  = (group & 1) == 0 ? __fadd_rn(x2, x3) : __fsub_rn(x2, x3);
    const float out = (group & 2) == 0 ? __fadd_rn(y0, y1) : __fsub_rn(y0, y1);
    return __fmul_rn(out, 0x1p-4f);
}

// One full warp owns one D256 row. Lane l carries dimensions l+32*r in values[r]. The fixed
// normalized Sylvester transform is shared by persistent K preparation and transient Q
// preparation; it never materializes an intermediate row outside registers.
__device__ __forceinline__ void normalized_hadamard_d256_inplace(float (&values)[8], int lane) {
    hadamard_d32_columns_inplace(values, lane);

#pragma unroll
    for (int span = 1; span < 8; span <<= 1) {
#pragma unroll
        for (int base = 0; base < 8; base += 2 * span) {
#pragma unroll
            for (int offset = 0; offset < span; ++offset) {
                const float low              = values[base + offset];
                const float high             = values[base + offset + span];
                values[base + offset]        = __fadd_rn(low, high);
                values[base + offset + span] = __fsub_rn(low, high);
            }
        }
    }

#pragma unroll
    for (float& value : values) { value = __fmul_rn(value, 0x1p-4f); }
}

// Same transform, contiguous layout: lane l carries dimensions 8*l + i in values[i], so each
// dimension row can be fetched with one 64-bit load. Value-identical to the strided helper
// above: every stage is the standard per-bit Sylvester butterfly (the d32 helper's "peer-own"
// branch is that same butterfly written from the bit-1 holder), per-bit stages commute, and
// the normalization is shared.
__device__ __forceinline__ void normalized_hadamard_d256_contig8_inplace(float (&values)[8],
                                                                         int lane) {
    constexpr unsigned long long FullMask = 0xffffffffffffffffULL;

    // d-bits 0..2 live in the register index.
#pragma unroll
    for (int span = 1; span < 8; span <<= 1) {
#pragma unroll
        for (int base = 0; base < 8; base += 2 * span) {
#pragma unroll
            for (int offset = 0; offset < span; ++offset) {
                const float low              = values[base + offset];
                const float high             = values[base + offset + span];
                values[base + offset]        = __fadd_rn(low, high);
                values[base + offset + span] = __fsub_rn(low, high);
            }
        }
    }

    // d-bits 3..7 live in the lane id; every register pairs with the same-index register of
    // the xor-peer lane.
#pragma unroll
    for (int stride = 1; stride <= 16; stride <<= 1) {
#pragma unroll
        for (int column = 0; column < 8; ++column) {
            const float value = values[column];
            const float peer  = __shfl_xor_sync(FullMask, value, stride);
            values[column]    = (lane & stride) == 0 ? __fadd_rn(value, peer)
                                                     : __fsub_rn(peer, value);
        }
    }

#pragma unroll
    for (float& value : values) { value = __fmul_rn(value, 0x1p-4f); }
}

// D1024 = 32 lanes x 32 registers(每 warp 一个 1024 块;lane l 持维度 l+32r 于 values[r]):
// d32 列级(stage)+ 5 个寄存器 span + 2^-5 归一。原为 ternary_rotation_kernels.cuh
// 匿名空间私有件,2026-10-09 M2e(silu+rotate 融合)提升到公共头 —— 单一事实源,
// 旋转核/旋转量化融合核/silu 旋转融合核共用,禁止第二份手抄蝶形。
__device__ __forceinline__ void normalized_hadamard_d1024_inplace(float (&values)[32], int lane) {
    hadamard_d32_columns_inplace(values, lane);

#pragma unroll
    for (int span = 1; span < 32; span <<= 1) {
#pragma unroll
        for (int base = 0; base < 32; base += 2 * span) {
#pragma unroll
            for (int offset = 0; offset < span; ++offset) {
                const float low              = values[base + offset];
                const float high             = values[base + offset + span];
                values[base + offset]        = __fadd_rn(low, high);
                values[base + offset + span] = __fsub_rn(low, high);
            }
        }
    }

#pragma unroll
    for (float& value : values) { value = __fmul_rn(value, 0x1p-5f); }
}

} // namespace ninfer::ops

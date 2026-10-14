#pragma once

// Signed int8, per-token G64 KV-cache codec shared by append and causal-attention kernels.
// This header owns index math, vectorized decode, and scalar encode; there is deliberately no
// standalone transcode kernel in the production path.

#include "ops/common/math.cuh"
#include "ops/common/memory.cuh"
#include "ops/kernel/paged_kv_address.cuh"
#include "ops/kv_cache/hadamard_d256.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>
#include <cstdlib>

namespace ninfer::ops {

inline constexpr int kKVCacheInt8HeadDim = 256;
inline constexpr int kKVCacheInt8Group   = 64;
inline constexpr int kKVCacheInt8Groups  = kKVCacheInt8HeadDim / kKVCacheInt8Group;

template <typename Geometry>
__device__ __forceinline__ std::int64_t
kv_cache_int8_quant_code_index(int physical_page, int kv_head, int d, int page_offset) {
    return paged_kv_element_offset<kKVCacheInt8HeadDim, Geometry::KVHeads>(physical_page, kv_head,
                                                                           page_offset, d);
}

template <typename Geometry>
__device__ __forceinline__ std::int64_t
kv_cache_int8_quant_scale_index(int physical_page, int kv_head, int group, int page_offset) {
    return paged_kv_element_offset<kKVCacheInt8Groups, Geometry::KVHeads>(physical_page, kv_head,
                                                                          page_offset, group);
}

template <typename Geometry>
__device__ __forceinline__ std::int64_t kv_cache_int8_quant_src_index(int kv_head, int d,
                                                                      int token) {
    return static_cast<std::int64_t>(d) +
           static_cast<std::int64_t>(kKVCacheInt8HeadDim) *
               (static_cast<std::int64_t>(kv_head) +
                static_cast<std::int64_t>(Geometry::KVHeads) * token);
}

struct KVCacheInt8QuantParams {
    __half scale;
    float inverse_scale;
};

// Exact persistent group-scale boundary shared by standalone and fused append. The stored scale is
// FP16-RNE(absmax/127); codes always use the reciprocal of that represented FP16 value.
__device__ __forceinline__ KVCacheInt8QuantParams kv_cache_int8_quant_params(float absmax) {
    const __half scale            = __float2half_rn(absmax > 0.0f ? absmax / 127.0f : 0.0f);
    const float represented_scale = __half2float(scale);
    return {
        .scale         = scale,
        .inverse_scale = represented_scale > 0.0f ? 1.0f / represented_scale : 0.0f,
    };
}

__device__ __forceinline__ std::int8_t kv_cache_int8_quant_code(float x, float inv_scale) {
    if (inv_scale == 0.0f) { return static_cast<std::int8_t>(0); }
    int q = __float2int_rn(x * inv_scale);
    q     = max(-127, min(127, q));
    return static_cast<std::int8_t>(q);
}

__device__ __forceinline__ int4 kv_cache_int8_dequant_i8x8_from(const std::int8_t* codes8,
                                                                float s) {
    const int2 raw       = load_vec<int2>(codes8);
    const std::int8_t* c = reinterpret_cast<const std::int8_t*>(&raw);
    unsigned packed[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float x0 = static_cast<float>(c[2 * i]) * s;
        const float x1 = static_cast<float>(c[2 * i + 1]) * s;
        packed[i]      = pack_bf16x2(x0, x1);
    }
    return make_int4(static_cast<int>(packed[0]), static_cast<int>(packed[1]),
                     static_cast<int>(packed[2]), static_cast<int>(packed[3]));
}

// ---- int4 打包变体(NINFER_KV_INT4=1,2026-10-01)---------------------------------
// 页布局/容量不变:同一 256B/key 行的低半 128B 装打包码,高半闲置。值域 [-8,7],
// nibble n = v+8;字节 = (奇维高 4 位 | 偶维低 4 位);scale 语义 = fp16(absmax/7),
// 组边界(G64)与 scale 索引与 int8 完全一致。

inline bool kv_cache_int4_enabled() {
    static const bool enabled = [] {
        const char* v = std::getenv("NINFER_KV_INT4");
        return v != nullptr && std::atoi(v) != 0;
    }();
    return enabled;
}

__device__ __forceinline__ KVCacheInt8QuantParams kv_cache_int4_quant_params(float absmax) {
    const __half scale            = __float2half_rn(absmax > 0.0f ? absmax / 7.0f : 0.0f);
    const float represented_scale = __half2float(scale);
    return {
        .scale         = scale,
        .inverse_scale = represented_scale > 0.0f ? 1.0f / represented_scale : 0.0f,
    };
}

// 单值量化到 nibble 码(已 +8 偏移,0..15)
__device__ __forceinline__ std::uint32_t kv_cache_int4_quant_nibble(float x, float inv_scale) {
    if (inv_scale == 0.0f) { return 8u; }
    int q = __float2int_rn(x * inv_scale);
    q     = max(-8, min(7, q));
    return static_cast<std::uint32_t>(q + 8);
}

// 打包字节:低半字节 = 偶维,高半字节 = 奇维
__device__ __forceinline__ std::uint8_t kv_cache_int4_pack_byte(std::uint32_t n_even,
                                                                std::uint32_t n_odd) {
    return static_cast<std::uint8_t>(n_even | (n_odd << 4));
}

// 读取:4 字节 u32 = 8 维,半字节解到 [-8,7] 浮点
__device__ __forceinline__ void kv_cache_int4_dequant_u32x8(std::uint32_t raw, float s,
                                                            float (&out)[8]) {
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const std::uint32_t byte = (raw >> (8 * (i / 2))) & 0xFFu;
        const float n = static_cast<float>((i % 2 == 0) ? (byte & 0xFu) : (byte >> 4));
        out[i]        = (n - 8.0f) * s;
    }
}

} // namespace ninfer::ops

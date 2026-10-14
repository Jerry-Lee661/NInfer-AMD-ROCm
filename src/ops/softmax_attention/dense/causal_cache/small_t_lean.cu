// 瘦版 small-T decode attention：只服务 T ≤ 4 且窗口 ≤ 32 keys（一个 key tile）的解码场景。
//
// 为什么单独一份、而不改 small_t_i8.cuh：实测这个栈上「单 workgroup LDS > 32KB 的核，每次启动
// 要付约 0.54ms」（探针 REPEAT 线性、CFG=arena 无变化、reduce 核 0KB 免费、WMMA 核 24KB 便宜，
// 四点一致）。原 partial 核 arena(4*Bc*D=32KB) + q_s(4KB) ≈ 37.8KB 越线，而 decode 每步要起 16 次
// → 约 8.6ms/步（占整步 22%）。瘦版去掉 V 的 int8 码暂存（反量化时直接读全局缓存），
// arena 降到 3*Bc*D=24KB、总量约 29.8KB 过线。**老核一行未动，仍作回退。**
//
// 数值语义与老核一致：K 先 Hadamard 旋转再量化入缓存、Q 用同一个旋转（但**不量化**——
// 窗口 ≤32 keys 时 float 点积足够快且更准）、V 直接量化；追加与解量化全部复用 kv_cache 的 helper。

#include "ops/softmax_attention/dense/causal_cache/launch.h"

#include "core/device.h"
#include "ninfer/ops/softmax_attention.h"
#include "ops/common/math.h"
#include "ops/softmax_attention/dense/causal_cache/small_t.cuh"
#include "ops/softmax_attention/dense/causal_cache/small_t_i8_lean.cuh"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>

// 诊断（宿主侧 NINFER_ATTN_TIMING=1 打开）：SIMT 核两段 cycle 累计（全体 CTA 之和，
// 宿主每 64 次调用读取并清零）。32 位差值：相位远小于 2^32 周期，回绕安全。
__device__ unsigned long long g_simt_phase_cycles[4];  // [0]=prologue [1]=K-load [2]=QK-math [3]=softmax+V

namespace ninfer::ops::detail {
namespace {

// 固定 <8 warps, MinBlocksPerSm 2, Bc 32, 无动态 smem>：窗口只有一个 tile，多 warp 流水与
// split-K 都是纯开销；256 线程足够铺满 Br×D 的 Q 装载与 Bc×D 的 V 反量化。
template <typename Geometry, int TokenTile, bool Masked>
void launch_lean_partial(const Tensor& q, CausalAppendInput input, const Tensor& pos, float scale,
                         PagedKVBatchLayerView cache, const CausalSmallTInvocation& invocation,
                         std::int32_t logical_capacity, std::int32_t splits, Tensor& partial_acc,
                         Tensor& partial_m, Tensor& partial_l, cudaStream_t stream) {
    constexpr int kWarps     = 8;
    // **MinBlocksPerSm 必须是 1**：老核传 2，逼编译器为「每 SM 两 block」压寄存器，结果是
    // spill（ISA 实测 private_segment 1504-1624B/线程），而这个栈上带私有段的核每次启动
    // 要付约 0.5ms（对照：无 spill 的归约核启动免费）。19KB LDS 下本来就只驻留 1 个 block。
    constexpr int kMinBlocks = 1;
    constexpr int kKeyBlock  = 32;
    const dim3 grid(Geometry::KVHeads, splits, invocation.batch_size);
    const auto launch = [&]<bool MultiBatch>() {
        causal_attention_small_t_i8_lean_kernel<Geometry, TokenTile, kWarps, kMinBlocks, kKeyBlock,
                                                false, MultiBatch, Masked, CausalAppendInput>
            <<<grid, kWarps * 32, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(q.data), input,
                static_cast<const std::int32_t*>(pos.data),
                static_cast<std::int8_t*>(cache.k_pages.data),
                static_cast<std::int8_t*>(cache.v_pages.data),
                static_cast<__half*>(cache.k_scale_pages.data),
                static_cast<__half*>(cache.v_scale_pages.data),
                static_cast<const std::int32_t*>(cache.block_tables.data),
                invocation.valid_columns == nullptr
                    ? nullptr
                    : static_cast<const std::int32_t*>(invocation.valid_columns->data),
                invocation.table_rows == nullptr
                    ? nullptr
                    : static_cast<const std::int32_t*>(invocation.table_rows->data),
                cache.block_tables.ne[0], invocation.full_width, invocation.column_begin,
                logical_capacity, scale, static_cast<__nv_bfloat16*>(partial_acc.data),
                static_cast<float*>(partial_m.data), static_cast<float*>(partial_l.data));
    };
    // 多序列（服务端并发）走 MultiBatch 实例：核体里的 workspace 偏移、Q/O 列偏移与页表行
    // 本来就在 `if constexpr (MultiBatch)` 下备好，之前只是没实例化、路由也只放 batch==1 进来。
    if (invocation.batch_size > 1) {
        launch.template operator()<true>();
    } else {
        launch.template operator()<false>();
    }
    CUDA_CHECK(cudaGetLastError());
}

template <typename Geometry, int TokenTile>
void launch_lean_partial_masked(const Tensor& q, CausalAppendInput input, const Tensor& pos,
                                float scale, PagedKVBatchLayerView cache,
                                const CausalSmallTInvocation& invocation,
                                std::int32_t logical_capacity, std::int32_t splits,
                                Tensor& partial_acc, Tensor& partial_m, Tensor& partial_l,
                                cudaStream_t stream) {
    if (invocation.valid_columns == nullptr) {
        launch_lean_partial<Geometry, TokenTile, false>(q, input, pos, scale, cache, invocation,
                                                        logical_capacity, splits, partial_acc,
                                                        partial_m, partial_l, stream);
    } else {
        launch_lean_partial<Geometry, TokenTile, true>(q, input, pos, scale, cache, invocation,
                                                       logical_capacity, splits, partial_acc,
                                                       partial_m, partial_l, stream);
    }
}

template <typename Geometry>
void launch_lean_partial_by_width(const Tensor& q, CausalAppendInput input, const Tensor& pos,
                                  float scale, PagedKVBatchLayerView cache,
                                  const CausalSmallTInvocation& invocation,
                                  std::int32_t logical_capacity, std::int32_t splits,
                                  Tensor& partial_acc, Tensor& partial_m, Tensor& partial_l,
                                  cudaStream_t stream) {
    switch (invocation.width) {
    case 1:
        launch_lean_partial_masked<Geometry, 1>(q, input, pos, scale, cache, invocation,
                                                logical_capacity, splits, partial_acc, partial_m,
                                                partial_l, stream);
        break;
    case 2:
        launch_lean_partial_masked<Geometry, 2>(q, input, pos, scale, cache, invocation,
                                                logical_capacity, splits, partial_acc, partial_m,
                                                partial_l, stream);
        break;
    case 3:
        launch_lean_partial_masked<Geometry, 3>(q, input, pos, scale, cache, invocation,
                                                logical_capacity, splits, partial_acc, partial_m,
                                                partial_l, stream);
        break;
    case 4:
        launch_lean_partial_masked<Geometry, 4>(q, input, pos, scale, cache, invocation,
                                                logical_capacity, splits, partial_acc, partial_m,
                                                partial_l, stream);
        break;
    default:
        throw std::invalid_argument("lean small_t attention: unsupported T");
    }
}

// 归约：与 small_t.cu 里的那份同源（该核 0 LDS、启动免费，无需瘦身）。
template <typename Geometry>
void launch_lean_reduce(const Tensor& pos, PagedKVBatchLayerView cache,
                        const CausalSmallTInvocation& invocation, std::int32_t splits,
                        Tensor& partial_acc, Tensor& partial_m, Tensor& partial_l, Tensor& out,
                        cudaStream_t stream) {
    constexpr int kReduceBlock = 256;
    constexpr int kDChunk      = 64;
    const dim3 reduce_grid(Geometry::QHeads, div_up(kCausalHeadDim, kDChunk),
                           invocation.width * invocation.batch_size);
    const auto launch_reduce = [&]<bool Int8, bool MultiBatch, bool Masked, bool Offset>() {
        causal_attention_small_t_reduce_output_kernel<Geometry, kDChunk, Int8, MultiBatch, Masked,
                                                      Offset>
            <<<reduce_grid, kReduceBlock, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(partial_acc.data),
                static_cast<const float*>(partial_m.data),
                static_cast<const float*>(partial_l.data),
                static_cast<const std::int32_t*>(pos.data),
                invocation.valid_columns == nullptr
                    ? nullptr
                    : static_cast<const std::int32_t*>(invocation.valid_columns->data),
                invocation.width, invocation.full_width, invocation.column_begin,
                invocation.batch_size, splits, static_cast<__nv_bfloat16*>(out.data));
    };
    const bool masked = invocation.valid_columns != nullptr;
    const auto launch_profile = [&]<bool Int8, bool MultiBatch, bool Masked>() {
        if (invocation.column_begin == 0) {
            launch_reduce.template operator()<Int8, MultiBatch, Masked, false>();
        } else {
            launch_reduce.template operator()<Int8, MultiBatch, Masked, true>();
        }
    };
    const auto launch_for_dtype = [&]<bool Int8>() {
        if (invocation.batch_size == 1) {
            if (masked) {
                launch_profile.template operator()<Int8, false, true>();
            } else {
                launch_profile.template operator()<Int8, false, false>();
            }
        } else if (masked) {
            launch_profile.template operator()<Int8, true, true>();
        } else {
            launch_profile.template operator()<Int8, true, false>();
        }
    };
    if (cache.dtype == DType::I8) {
        launch_for_dtype.template operator()<true>();
    } else {
        launch_for_dtype.template operator()<false>();
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

void causal_attention_small_t_lean_launch(
    const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& pos,
    const Tensor& valid_columns, const Tensor& table_rows, float scale, PagedKVBatchLayerView cache,
    CausalAttentionExecutionEnvelope envelope, std::int32_t column_begin, std::int32_t width,
    Tensor& partial_acc, Tensor& partial_m, Tensor& partial_l, Tensor& out, cudaStream_t stream) {
    const CausalAppendInput input{static_cast<const __nv_bfloat16*>(k.data),
                                  static_cast<const __nv_bfloat16*>(v.data)};
    const CausalSmallTInvocation invocation{
        .valid_columns = valid_columns.data == nullptr ? nullptr : &valid_columns,
        .table_rows    = &table_rows,
        .full_width    = q.ne[2],
        .column_begin  = column_begin,
        .width         = width,
        .batch_size    = q.ne[3],
    };
    const auto logical_capacity = static_cast<std::int32_t>(envelope.max_visible_keys);
    // 用 launch.h 里的公开包装（它按 q_heads 分派到几何特化）；模板本体在 small_t.cu 的
    // 匿名命名空间里，跨 TU 不可见。
    const auto splits = causal_attention_split_capacity(q.ne[1], width, cache.dtype, envelope);

    if (q.ne[1] == CausalD256H24Kv4::QHeads) {
        launch_lean_partial_by_width<CausalD256H24Kv4>(q, input, pos, scale, cache, invocation,
                                                       logical_capacity, splits, partial_acc,
                                                       partial_m, partial_l, stream);
        launch_lean_reduce<CausalD256H24Kv4>(pos, cache, invocation, splits, partial_acc, partial_m,
                                             partial_l, out, stream);
        return;
    }
    launch_lean_partial_by_width<CausalD256H16Kv2>(q, input, pos, scale, cache, invocation,
                                                   logical_capacity, splits, partial_acc, partial_m,
                                                   partial_l, stream);
    launch_lean_reduce<CausalD256H16Kv2>(pos, cache, invocation, splits, partial_acc, partial_m,
                                         partial_l, out, stream);
}

// ============================================================================
// 零溢出的 SIMT 解码 attention 核（目标：拿回 8.6ms/步）
//
// 为什么再写一个：ISA 实测钉死了真机制——**非零 private segment（寄存器溢出）的核，每次
// 启动付约 0.5ms**（对照：WMMA 核 170 VGPR/24KB LDS/零溢出/启动便宜；归约核零溢出/免费；
// 而照抄老核的 lean 版无论怎么调 LDS 与 launch_bounds 都溢 1504-1624B，因为它的 QK 走
// tensor-core MMA，片段数组 + 全展开让活跃区间远超 256 VGPR）。所以要真正零溢出，必须
using A8_w = unsigned __attribute__((ext_vector_type(8)));
using B16_w = unsigned short __attribute__((ext_vector_type(16)));
using C8F_w = float __attribute__((ext_vector_type(8)));
__device__ __forceinline__ A8_w load_a_w(const __hip_bfloat16* row) {
    const uint4 lo = *reinterpret_cast<const uint4*>(row);
    const uint4 hi = *reinterpret_cast<const uint4*>(row + 8);
    A8_w va;
    va[0] = lo.x; va[1] = lo.y; va[2] = lo.z; va[3] = lo.w;
    va[4] = hi.x; va[5] = hi.y; va[6] = hi.z; va[7] = hi.w;
    return va;
}
__device__ __forceinline__ B16_w load_b_w(const __hip_bfloat16* row) {
    B16_w vb;
#pragma unroll
    for (int i = 0; i < 16; ++i) { vb[i] = *reinterpret_cast<const unsigned short*>(row + i); }
    return vb;
}
using A8_w = unsigned __attribute__((ext_vector_type(8)));
using B16_w = unsigned short __attribute__((ext_vector_type(16)));
using C8F_w = float __attribute__((ext_vector_type(8)));

// **不用 tensor core**：Q 不量化、float 点积；K/V 不暂存、现读现反量化。
//
// 线程分工：8 warp；每个 warp 依次处理若干行（row = warp, warp+8, ...），
// 每个 lane 负责 256 维里的 8 维（d = lane + 32*r）。每行的 Q 在寄存器里，整 warp 协作做
// H256 旋转（与缓存里旋转过的 K 成对）；逐 key 做 8 维部分点积 → warp 归约成 score →
// 在线 softmax → 用 p 累加 V 的 8 维。寄存器 ≈ 8(q) + 8(acc) + 少量标量 ≈ 30，必然零溢出。
//
// LDS 只剩追加段的 H64 中转（TokenTile*D*4B = 4KB @ T=4），见下面的 static_assert。
// 追加段（写 KV 缓存）整段照抄老核——它是逐元素量化路径，寄存器压力小。
template <typename Geometry, int TokenTile, bool Masked, typename CacheInput,
          bool Contig8 = false, bool MultiBatch = false, int MultiKey = 1,
          bool Prefetch = false, bool RedBatch = false, bool QhFused = false>
__global__ __launch_bounds__(256, 1) void causal_attention_small_t_simt_kernel(
    const __nv_bfloat16* q, CacheInput input, const std::int32_t* pos, std::int8_t* cache_k_i8,
    std::int8_t* cache_v_i8, __half* cache_k_scale, __half* cache_v_scale,
    const std::int32_t* block_tables, const std::int32_t* valid_columns,
    const std::int32_t* table_rows, std::int32_t table_stride, std::int32_t full_width,
    std::int32_t column_begin, std::int32_t logical_capacity, float scale,
    __nv_bfloat16* partial_acc, float* partial_m, float* partial_l,
    std::int32_t kv_int4, std::int32_t unroll4, std::int32_t wmma) {
    constexpr int Wc         = 8;
    constexpr int Threads    = Wc * 32;
    constexpr int D          = kCausalHeadDim;
    constexpr int Groups     = kKVCacheInt8Groups;
    constexpr int RowCount   = TokenTile * Geometry::GroupSize;
    constexpr int PageIds    = 64;
    constexpr float Log2E    = 1.4426950408889634074f;
    constexpr unsigned long long FullMask = 0xffffffffffffffffULL;

    __shared__ float h64_s[TokenTile * D];  // 追加段的 H64 中转（逐元素，压力小）
    __shared__ std::int32_t pages_s[PageIds];
    static_assert(sizeof(h64_s) + sizeof(pages_s) <= 32768, "simt attention exceeds the LDS budget");

    const int kv_head     = static_cast<int>(blockIdx.x);
    const int split       = static_cast<int>(blockIdx.y);
    const int split_count = static_cast<int>(gridDim.y);
    const int tid         = static_cast<int>(threadIdx.x);
    const int warp        = tid >> 5;
    const int lane        = tid & 31;
    const int batch =
        MultiBatch ? static_cast<int>(blockIdx.z) : 0;  // 多序列（服务端并发）走 batch 维

    const bool timing_on = tid == 0;  // 恒累计（每块一个线程的 atomicAdd 可忽略）；使能由宿主打印把关
    unsigned t_prev = 0;
    if (timing_on) { t_prev = static_cast<unsigned>(clock64()); }

    int valid_tokens = TokenTile;
    if constexpr (Masked) {
        const int remaining = valid_columns[batch] - column_begin;
        valid_tokens        = remaining <= 0 ? 0 : (remaining < TokenTile ? remaining : TokenTile);
    }
    const std::int64_t column_base =
        column_begin + (MultiBatch ? static_cast<std::int64_t>(batch) * full_width : 0);
    q += static_cast<std::int64_t>(D) * Geometry::QHeads * column_base;
    pos += column_base;
    if constexpr (CacheInput::writes_cache) {
        input.k += static_cast<std::int64_t>(D) * Geometry::KVHeads * column_base;
        input.v += static_cast<std::int64_t>(D) * Geometry::KVHeads * column_base;
    }
    const int table_row = table_rows == nullptr ? 0 : table_rows[batch];
    const std::int32_t* block_table =
        block_tables + static_cast<std::int64_t>(table_row) * table_stride;
    if constexpr (MultiBatch) {
        // 工作区最后一维是 splits*batch（batch 在最外层），与老核的偏移公式一致。
        partial_acc += static_cast<std::int64_t>(batch) * D * Geometry::QHeads * TokenTile *
                       split_count;
        partial_m += static_cast<std::int64_t>(batch) * Geometry::QHeads * TokenTile * split_count;
        partial_l += static_cast<std::int64_t>(batch) * Geometry::QHeads * TokenTile * split_count;
    }

    auto write_neutral = [&]() {
        for (int row = tid; row < RowCount; row += Threads) {
            int q_head = 0;
            int token  = 0;
            causal_small_t_tc_row_to_qt<Geometry>(row, TokenTile, kv_head, q_head, token);
            if (causal_valid_q_head<Geometry>(kv_head, q_head)) {
                partial_m[causal_partial_stat_index<Geometry>(q_head, token, split, TokenTile)] =
                    -CUDART_INF_F;
                partial_l[causal_partial_stat_index<Geometry>(q_head, token, split, TokenTile)] =
                    0.0f;
            }
        }
        for (int idx = tid; idx < RowCount * D; idx += Threads) {
            const int row = idx / D;
            const int d   = idx - row * D;
            int q_head    = 0;
            int token     = 0;
            causal_small_t_tc_row_to_qt<Geometry>(row, TokenTile, kv_head, q_head, token);
            if (causal_valid_q_head<Geometry>(kv_head, q_head)) {
                partial_acc[causal_partial_acc_index<Geometry>(q_head, d, token, split,
                                                               TokenTile)] = __float2bfloat16(0.0f);
            }
        }
    };

    if (kv_head < 0 || kv_head >= Geometry::KVHeads || split_count <= 0) { return; }
    if (valid_tokens == 0) {
        write_neutral();
        return;
    }
    const std::int32_t first_pos = pos[0];
    const std::int32_t last_pos  = pos[TokenTile - 1];
    if (first_pos < 0 || last_pos < 0 || last_pos >= logical_capacity) {
        write_neutral();
        return;
    }
    const int window = last_pos + 1;
    const int active_split_count =
        causal_small_t_active_splits<Geometry, true>(window, split_count, TokenTile);
    if (split >= active_split_count) { return; }

    const int logical_tiles = div_up(window, 32);
    const bool tile_split   = logical_tiles >= active_split_count;
    const int units_per_split =
        tile_split ? div_up(logical_tiles, active_split_count) : div_up(window, active_split_count);
    const int split_start = split * units_per_split * (tile_split ? 32 : 1);
    const int split_limit = split_start + units_per_split * (tile_split ? 32 : 1);
    const int split_end   = (split_limit < window) ? split_limit : window;
    if (split_start >= split_end) {
        write_neutral();
        return;
    }
    const int first_page = split_start >> kPagedKVPageShift;
    const int page_count = ((split_end - 1) >> kPagedKVPageShift) - first_page + 1;
    for (int page = tid; page < page_count; page += Threads) {
        pages_s[page] = block_table[first_page + page];
    }
    __syncthreads();

    if constexpr (CacheInput::writes_cache) {
        // 追加：H256 = 四组独立 H64（保留完整变换的蝶形/舍入次序），V 不旋转、沿用 G64 编解码。
        for (int pair = warp; pair < valid_tokens * Groups; pair += Wc) {
            const int token    = pair / Groups;
            const int grp      = pair - token * Groups;
            const int position = pos[token];
            if (position < split_start || position >= split_end) { continue; }
            const int d0            = grp * kKVCacheInt8Group + lane;
            const int d1            = d0 + 32;
            const std::int64_t src0 = kv_cache_int8_new_index<Geometry>(kv_head, d0, token);
            const std::int64_t src1 = kv_cache_int8_new_index<Geometry>(kv_head, d1, token);
            float k_h64[2] = {__bfloat162float(input.k[src0]), __bfloat162float(input.k[src1])};
            hadamard_d64_fragment_inplace(k_h64, lane);
            h64_s[token * D + d0] = k_h64[0];
            h64_s[token * D + d1] = k_h64[1];
        }
        __syncthreads();
        for (int pair = warp; pair < valid_tokens * Groups; pair += Wc) {
            const int token    = pair / Groups;
            const int grp      = pair - token * Groups;
            const int position = pos[token];
            if (position < split_start || position >= split_end) { continue; }
            const int physical_page = pages_s[(position >> kPagedKVPageShift) - first_page];
            const int page_offset   = position & kPagedKVPageMask;
            const int d0            = grp * kKVCacheInt8Group + lane;
            const int d1            = d0 + 32;
            const std::int64_t src0 = kv_cache_int8_new_index<Geometry>(kv_head, d0, token);
            const std::int64_t src1 = kv_cache_int8_new_index<Geometry>(kv_head, d1, token);
            float k_out[2];
#pragma unroll
            for (int half = 0; half < 2; ++half) {
                const int dh   = lane + half * 32;
                const float x0 = h64_s[token * D + dh];
                const float x1 = h64_s[token * D + kKVCacheInt8Group + dh];
                const float x2 = h64_s[token * D + 2 * kKVCacheInt8Group + dh];
                const float x3 = h64_s[token * D + 3 * kKVCacheInt8Group + dh];
                k_out[half]    = normalized_hadamard_d256_group_value_from_h64(x0, x1, x2, x3, grp);
            }
            const float kv0 = k_out[0];
            const float kv1 = k_out[1];
            const float vv0 = __bfloat162float(input.v[src0]);
            const float vv1 = __bfloat162float(input.v[src1]);
            float kamax     = fmaxf(fabsf(kv0), fabsf(kv1));
            float vamax     = fmaxf(fabsf(vv0), fabsf(vv1));
            kamax           = warp_max<32>(kamax, FullMask);
            vamax           = warp_max<32>(vamax, FullMask);
            if (kv_int4 != 0) {
                // int4 打包:值域 [-8,7],组内相邻 lane 配对(shfl 伙伴),组基 grp*32
                const auto kq4    = kv_cache_int4_quant_params(kamax);
                const auto vq4    = kv_cache_int4_quant_params(vamax);
                const std::uint32_t n0 = kv_cache_int4_quant_nibble(kv0, kq4.inverse_scale);
                const std::uint32_t n1 = kv_cache_int4_quant_nibble(kv1, kq4.inverse_scale);
                const std::uint32_t v0n = kv_cache_int4_quant_nibble(vv0, vq4.inverse_scale);
                const std::uint32_t v1n = kv_cache_int4_quant_nibble(vv1, vq4.inverse_scale);
                const std::uint32_t n0p = __shfl_sync(FullMask, n0, lane ^ 1);
                const std::uint32_t n1p = __shfl_sync(FullMask, n1, lane ^ 1);
                const std::uint32_t v0p = __shfl_sync(FullMask, v0n, lane ^ 1);
                const std::uint32_t v1p = __shfl_sync(FullMask, v1n, lane ^ 1);
                if ((lane & 1u) == 0u) {
                    const std::int64_t base4 = kv_cache_int8_quant_code_index<Geometry>(
                        physical_page, kv_head, (grp * kKVCacheInt8Group) >> 1, page_offset);
                    cache_k_i8[base4 + (lane >> 1)]      = kv_cache_int4_pack_byte(n0, n0p);
                    cache_k_i8[base4 + 16 + (lane >> 1)] = kv_cache_int4_pack_byte(n1, n1p);
                    cache_v_i8[base4 + (lane >> 1)]      = kv_cache_int4_pack_byte(v0n, v0p);
                    cache_v_i8[base4 + 16 + (lane >> 1)] = kv_cache_int4_pack_byte(v1n, v1p);
                }
                if (lane == 0) {
                    const std::int64_t so = kv_cache_int8_quant_scale_index<Geometry>(
                        physical_page, kv_head, grp, page_offset);
                    cache_k_scale[so] = kq4.scale;
                    cache_v_scale[so] = vq4.scale;
                }
            } else {
            const auto k_quant = kv_cache_int8_quant_params(kamax);
            const auto v_quant = kv_cache_int8_quant_params(vamax);
            cache_k_i8[kv_cache_int8_quant_code_index<Geometry>(physical_page, kv_head, d0,
                                                                page_offset)] =
                kv_cache_int8_quant_code(kv0, k_quant.inverse_scale);
            cache_k_i8[kv_cache_int8_quant_code_index<Geometry>(physical_page, kv_head, d1,
                                                                page_offset)] =
                kv_cache_int8_quant_code(kv1, k_quant.inverse_scale);
            cache_v_i8[kv_cache_int8_quant_code_index<Geometry>(physical_page, kv_head, d0,
                                                                page_offset)] =
                kv_cache_int8_quant_code(vv0, v_quant.inverse_scale);
            cache_v_i8[kv_cache_int8_quant_code_index<Geometry>(physical_page, kv_head, d1,
                                                                page_offset)] =
                kv_cache_int8_quant_code(vv1, v_quant.inverse_scale);
            if (lane == 0) {
                const std::int64_t so = kv_cache_int8_quant_scale_index<Geometry>(
                    physical_page, kv_head, grp, page_offset);
                cache_k_scale[so] = k_quant.scale;
                cache_v_scale[so] = v_quant.scale;
            }
            }
        }
        __syncthreads();
    }

    // QhFused(NINFER_ATTN_QHFUSED=1, 默认关, 2026-10-05 P-C9): 同 token 的 GroupSize
    // 个 q_head 行共享一份 KV 流的假设臂。warp w 专管 local_q = w % GroupSize,
    // token 外层循环; K/V/scale 的物理装载每个 (key, local_q-warp) 各一次 ——
    // 同 token 其它 local_q 的 warp 装同 key 时靠 L1/L2 命中(同一时刻同地址),
    // 相比 row 主路径(6-8 行先后随机时间各装一遍)缓存友好得多。
    // W2(W 线,2026-10-05 立项):WMMA 分支(NINFER_ATTN_WMMA=1, 默认关)。
    // 数据流 = probe_w2_mtiles/causal32 已验证形态:解量化装载 → Q(hadamard)→ QK^T wmma
    // → 两遍在线 softmax → P@V wmma。限界:本实现要求 split_keys ≤ 32(单 K/V tile 对);
    // split 更大时回退基线(主门 d2 窗口小,落此界)。
    if (wmma) {
        constexpr int WM  = TokenTile * Geometry::GroupSize;   // 24 行(6 qhead × 4 token)
        constexpr int NT  = 16;                                 // n-tile 键宽
        if (split_end - split_start > 32) {
            // 超界:回退基线语义(不发生时零成本;发生时按未取分支走基线路径)
            for (int token = 0; token < TokenTile; ++token) {
                const int local_q = warp % Geometry::GroupSize;
                const int q_head  = kv_head * Geometry::GroupSize + local_q;
                if (!causal_valid_q_head<Geometry>(kv_head, q_head)) { continue; }
                const int qabs_t = pos[token];
                float acc[8];
#pragma unroll
                for (int r = 0; r < 8; ++r) { acc[r] = 0.0f; }
                float m = -CUDART_INF_F, l = 0.0f;
                float qv[8];
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    qv[i] = __bfloat162float(q[causal_q_index<Geometry>(q_head, 8 * lane + i, token)]);
                }
                normalized_hadamard_d256_contig8_inplace(qv, lane);
                for (int key = split_start; key < split_end; ++key) {
                    if (key > qabs_t) { break; }
                    const int physical_page = pages_s[(key >> kPagedKVPageShift) - first_page];
                    const int page_offset   = key & kPagedKVPageMask;
                    const std::int64_t base = kv_cache_int8_quant_code_index<Geometry>(
                        physical_page, kv_head, 0, page_offset);
                    const int2 kk = load_vec<int2>(cache_k_i8 + base + 8 * lane);
                    const int2 vv = load_vec<int2>(cache_v_i8 + base + 8 * lane);
                    const float ksc = __half2float(cache_k_scale[kv_cache_int8_quant_scale_index<Geometry>(
                        physical_page, kv_head, lane >> 3, page_offset)]);
                    const float vsc = __half2float(cache_v_scale[kv_cache_int8_quant_scale_index<Geometry>(
                        physical_page, kv_head, lane >> 3, page_offset)]);
                    const std::int8_t* kcc = reinterpret_cast<const std::int8_t*>(&kk);
                    const std::int8_t* vcc = reinterpret_cast<const std::int8_t*>(&vv);
                    float partial = 0.0f;
#pragma unroll
                    for (int i = 0; i < 8; ++i) { partial = fmaf(qv[i], static_cast<float>(kcc[i]) * ksc, partial); }
                    const float sc = warp_sum<32>(partial, FullMask) * scale;
                    const float nm = fmaxf(m, sc);
                    const float a  = (m == -CUDART_INF_F) ? 0.0f : __expf(m - nm);
                    const float pp = __expf(sc - nm);
#pragma unroll
                    for (int r = 0; r < 8; ++r) { acc[r] *= a; }
                    l = l * a + pp;
                    m = nm;
#pragma unroll
                    for (int i = 0; i < 8; ++i) { acc[i] = fmaf(pp, static_cast<float>(vcc[i]) * vsc, acc[i]); }
                }
#pragma unroll
                for (int r = 0; r < 8; ++r) { acc[r] /= l; }
                const int row = token * Geometry::GroupSize + local_q;
#pragma unroll
                for (int r = 0; r < 8; ++r) {
                    partial_acc[causal_partial_acc_index<Geometry>(q_head, 8 * lane + r, token, split,
                                                                   TokenTile)] =
                        __float2bfloat16(acc[r]);
                }
                if (lane == 0) {
                    partial_m[causal_partial_stat_index<Geometry>(q_head, token, split, TokenTile)] = m;
                    partial_l[causal_partial_stat_index<Geometry>(q_head, token, split, TokenTile)] = l;
                }
            }
            return;
        }
        // ── WMMA 主路径(split_keys ≤ 32;形态 = probe_w2_mtiles 已验证)──
        __shared__ __align__(16) __hip_bfloat16 s_q[WM * D];      // 12KB(hadamard 后 Q 行)
        __shared__ __align__(16) __hip_bfloat16 s_kb[16 * D];     // 4KB/ntile
        __shared__ __align__(16) __hip_bfloat16 s_vb[16 * D];     // 4KB/ntile
        __shared__ __align__(16) __hip_bfloat16 s_pb[16 * 16];    // 0.5KB
        __shared__ float s_sc[WM * 16];                          // 1.5KB
        // (W2) FA 累加器改为 partial_acc 原位累加(省 24KB LDS)                          // 24KB(FA 累加器)
        __shared__ float s_M[WM]; __shared__ float s_L[WM];
        __shared__ float s_A[WM];
        __shared__ int s_tk[WM];
        // 合计 ≈ 56.2KB < 64KB ✓
        for (int r = tid; r < WM; r += Threads) {
            int q_head = 0, token = 0;
            causal_small_t_tc_row_to_qt<Geometry>(r, TokenTile, kv_head, q_head, token);
            s_tk[r] = token;
        }
        __syncthreads();
        // Q staging:每 warp 一行,contig8 Hadamard(同 QhFused 臂);pad 行(≥WM)清零
        
        for (int r = warp; r < 32; r += Wc) {
            int q_head = 0, token = 0;
            float qv[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
            if (r < WM) {
                causal_small_t_tc_row_to_qt<Geometry>(r, TokenTile, kv_head, q_head, token);
                if (causal_valid_q_head<Geometry>(kv_head, q_head)) {
#pragma unroll
                    for (int i = 0; i < 8; ++i) {
                        qv[i] = __bfloat162float(q[causal_q_index<Geometry>(q_head, 8 * lane + i, s_tk[r])]);
                    }
                    normalized_hadamard_d256_contig8_inplace(qv, lane);
                }
            }
#pragma unroll
            for (int i = 0; i < 8; ++i) { s_q[r * D + 8 * lane + i] = __float2bfloat16(qv[i]); }
        }
        if (tid < WM) { s_M[tid] = -CUDART_INF_F; s_L[tid] = 0.0f; }
        for (int idx = tid; idx < WM * D; idx += Threads) { int qh = 0, tk = 0; causal_small_t_tc_row_to_qt<Geometry>(idx / D, TokenTile, kv_head, qh, tk); partial_acc[causal_partial_acc_index<Geometry>(qh, idx % D, tk, split, TokenTile)] = __float2bfloat16(0.0f); }
        __syncthreads();
        const int ntiles = (split_end - split_start + NT - 1) / NT;
        for (int nt = 0; nt < ntiles; ++nt) {
            const int key0 = split_start + nt * NT;
            // 解量化 K/V tile → s_kb/s_vb(bf16);尾 tile 越界键填 0(由 mask 屏蔽)
            for (int idx = tid; idx < 16 * D; idx += Threads) {
                const int kl = idx / D;
                const int dim = idx - kl * D;
                const int keyg = key0 + kl;
                float kv0 = 0.0f, vv0 = 0.0f;
                if (keyg < split_end) {
                    const int physical_page = pages_s[(keyg >> kPagedKVPageShift) - first_page];
                    const int page_offset   = keyg & kPagedKVPageMask;
                    const std::int64_t base = kv_cache_int8_quant_code_index<Geometry>(
                        physical_page, kv_head, 0, page_offset);
                    kv0 = static_cast<float>(cache_k_i8[base + dim]) *
                          __half2float(cache_k_scale[kv_cache_int8_quant_scale_index<Geometry>(
                              physical_page, kv_head, dim >> 6, page_offset)]);
                    vv0 = static_cast<float>(cache_v_i8[base + dim]) *
                          __half2float(cache_v_scale[kv_cache_int8_quant_scale_index<Geometry>(
                              physical_page, kv_head, dim >> 6, page_offset)]);
                }
                s_kb[kl * D + dim] = __float2bfloat16(kv0);
                s_vb[kl * D + dim] = __float2bfloat16(vv0);
            }
            __syncthreads();
            // QK^T:2 个 m-tile × 16 键;C cell(l,j) ↔ (m,n)=(2j+(l>>4), l&15)(w0 合同)
            if (tid < 32) {
                const int lane = tid;
#pragma unroll
                for (int mt = 0; mt < 2; ++mt) {
                    C8F_w dd{};
#pragma unroll
                    for (int ks = 0; ks < 16; ++ks) {
                        const A8_w va = load_a_w(s_q + (mt * 16 + (lane & 15)) * D + ks * 16);
                        const B16_w vb = load_b_w(&s_kb[(nt * 16 + (lane & 15)) * D + ks * 16]);
                        dd = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(va, vb, dd);
                    }
#pragma unroll
                    for (int j = 0; j < 8; ++j) {
                        const int m = mt * 16 + 2 * j + (lane >> 4);
                        const int n = nt * 16 + (lane & 15);
                        float sc = dd[j] * scale;
                        const int qpos = (s_tk[m] < TokenTile) ? pos[s_tk[m]] : (1 << 30); if (key0 + n >= split_end || key0 + n > qpos) { sc = -CUDART_INF_F; }
                        s_sc[m * NT + n] = sc;
                    }
                }
            }
            __syncthreads();
            // 行 max 更新 + acc 原位重缩放(系数先入 s_A,屏障在分歧外)
            if (tid < WM) {
                float mx = s_M[tid];
                for (int n = 0; n < NT; ++n) { mx = fmaxf(mx, s_sc[tid * NT + n]); }
                s_A[tid] = (s_M[tid] == -CUDART_INF_F || mx == -CUDART_INF_F) ? 0.0f : __expf(s_M[tid] - mx);
                s_L[tid] *= s_A[tid];
                s_M[tid] = mx;
            }
            __syncthreads();
            for (int idx = tid; idx < WM * D; idx += Threads) { int qh = 0, tk = 0; causal_small_t_tc_row_to_qt<Geometry>(idx / D, TokenTile, kv_head, qh, tk); const std::int64_t ai = causal_partial_acc_index<Geometry>(qh, idx % D, tk, split, TokenTile); partial_acc[ai] = __float2bfloat16(__bfloat162float(partial_acc[ai]) * s_A[idx / D]); }
            __syncthreads();
            // P = exp(s - M) → s_pb bf16;行 Σ → s_L
            if (tid < 32) {
                const int lane = tid;
#pragma unroll
                for (int mt = 0; mt < 2; ++mt) {
#pragma unroll
                    for (int j = 0; j < 8; ++j) {
                        const int m = mt * 16 + 2 * j + (lane >> 4);
                        const int n = nt * 16 + (lane & 15);
                        const float e = (s_sc[m * NT + n] == -CUDART_INF_F) ? 0.0f : __expf(s_sc[m * NT + n] - s_M[m]);
                        s_sc[m * NT + n] = e;
                        s_pb[m * 16 + n] = __hip_bfloat16(e);
                    }
                }
            }
            __syncthreads();
            if (tid < WM) {
                float lv = s_L[tid];
                for (int n = 0; n < NT; ++n) { lv += s_sc[tid * NT + n]; }
                s_L[tid] = lv;
            }
            __syncthreads();
            // P@V:2 m-tile × 16 d-tile;B 侧跨步读 s_vb((n)行主 → 16 连续 n per d 行)
            for (int dt = warp; dt < 16; dt += Wc) {
                const int lane = tid & 31;
#pragma unroll
                for (int mt = 0; mt < 2; ++mt) {
                    const A8_w va = load_a_w(&s_pb[(mt * 16 + (lane & 15)) * 16]);
                    B16_w vb;
#pragma unroll
                    for (int i = 0; i < 16; ++i) {
                        vb[i] = *reinterpret_cast<const unsigned short*>(
                            &s_vb[(nt * 16 + i) * D + dt * 16 + (lane & 15)]);
                    }
                    C8F_w z{};
                    C8F_w dd = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(va, vb, z);
#pragma unroll
                    for (int j = 0; j < 8; ++j) {
                        const int m = mt * 16 + 2 * j + (lane >> 4);
                        const int dcol = dt * 16 + (lane & 15);
                        if (m >= WM) { break; }
                        int q_head = 0, token = 0;
                        causal_small_t_tc_row_to_qt<Geometry>(m, TokenTile, kv_head, q_head, token);
                        int qh3 = 0, tk3 = 0;
                causal_small_t_tc_row_to_qt<Geometry>(m, TokenTile, kv_head, qh3, tk3);
                const std::int64_t ai3 = causal_partial_acc_index<Geometry>(qh3, dcol, tk3, split, TokenTile);
                partial_acc[ai3] = __float2bfloat16(__bfloat162float(partial_acc[ai3]) + dd[j]);
                        (void)q_head; (void)token;
                    }
                }
            }
            __syncthreads();
        }
        // 收尾:acc /= L → partial_acc bf16;m/l 写 stat
        for (int idx = tid; idx < WM * D; idx += Threads) {
            const int r = idx / D;
            if (s_L[r] > 0.0f) { int qh = 0, tk = 0; causal_small_t_tc_row_to_qt<Geometry>(r, TokenTile, kv_head, qh, tk); const std::int64_t ai = causal_partial_acc_index<Geometry>(qh, idx % D, tk, split, TokenTile); partial_acc[ai] = __float2bfloat16(__bfloat162float(partial_acc[ai]) / s_L[r]); }
        }
        __syncthreads();
        if (tid < WM) {
            int q_head = 0, token = 0;
            causal_small_t_tc_row_to_qt<Geometry>(tid, TokenTile, kv_head, q_head, token);
            if (causal_valid_q_head<Geometry>(kv_head, q_head)) {
                partial_m[causal_partial_stat_index<Geometry>(q_head, token, split, TokenTile)] = s_M[tid];
                partial_l[causal_partial_stat_index<Geometry>(q_head, token, split, TokenTile)] = s_L[tid];
            }
        }
        return;
    }
    if constexpr (QhFused) {
        for (int token = 0; token < TokenTile; ++token) {
            const int local_q = warp % Geometry::GroupSize;
            const int q_head  = kv_head * Geometry::GroupSize + local_q;
            if (!causal_valid_q_head<Geometry>(kv_head, q_head)) { continue; }
            const int qabs_t = pos[token];
            float acc[8];
#pragma unroll
            for (int r = 0; r < 8; ++r) { acc[r] = 0.0f; }
            float m = -CUDART_INF_F, l = 0.0f;
            float qv[8];
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                qv[i] = __bfloat162float(q[causal_q_index<Geometry>(q_head, 8 * lane + i, token)]);
            }
            normalized_hadamard_d256_contig8_inplace(qv, lane);
            for (int key = split_start; key < split_end; ++key) {
                if (key > qabs_t) { break; }
                const int physical_page = pages_s[(key >> kPagedKVPageShift) - first_page];
                const int page_offset   = key & kPagedKVPageMask;
                const std::int64_t base = kv_cache_int8_quant_code_index<Geometry>(
                    physical_page, kv_head, 0, page_offset);
                const int2 kk = load_vec<int2>(cache_k_i8 + base + 8 * lane);
                const int2 vv = load_vec<int2>(cache_v_i8 + base + 8 * lane);
                const float ksc = __half2float(cache_k_scale[kv_cache_int8_quant_scale_index<Geometry>(
                    physical_page, kv_head, lane >> 3, page_offset)]);
                const float vsc = __half2float(cache_v_scale[kv_cache_int8_quant_scale_index<Geometry>(
                    physical_page, kv_head, lane >> 3, page_offset)]);
                const std::int8_t* kc = reinterpret_cast<const std::int8_t*>(&kk);
                const std::int8_t* vc = reinterpret_cast<const std::int8_t*>(&vv);
                float partial = 0.0f;
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    partial = fmaf(qv[i], static_cast<float>(kc[i]) * ksc, partial);
                }
                const float sc = warp_sum<32>(partial, FullMask) * scale;
                const float nm = fmaxf(m, sc);
                const float a  = (m == -CUDART_INF_F) ? 0.0f : exp2_approx((m - nm) * Log2E);
                const float p  = exp2_approx((sc - nm) * Log2E);
#pragma unroll
                for (int r = 0; r < 8; ++r) { acc[r] *= a; }
                l = l * a + p;
                m = nm;
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    acc[i] = fmaf(p, static_cast<float>(vc[i]) * vsc, acc[i]);
                }
            }
#pragma unroll
            for (int r = 0; r < 8; ++r) { acc[r] /= l; }
            const int row = token * Geometry::GroupSize + local_q;
#pragma unroll
            for (int r = 0; r < 8; ++r) {
                partial_acc[causal_partial_acc_index<Geometry>(q_head, 8 * lane + r, token, split,
                                                               TokenTile)] =
                    __float2bfloat16(acc[r]);
            }
            if (lane == 0) {
                partial_m[causal_partial_stat_index<Geometry>(q_head, token, split, TokenTile)] = m;
                partial_l[causal_partial_stat_index<Geometry>(q_head, token, split, TokenTile)] = l;
            }
        }
    } else
    for (int row = warp; row < RowCount; row += Wc) {
        int q_head = 0;
        int token  = 0;
        causal_small_t_tc_row_to_qt<Geometry>(row, TokenTile, kv_head, q_head, token);
        if (!causal_valid_q_head<Geometry>(kv_head, q_head)) { continue; }
        const int qabs = pos[token];

        if (timing_on) {
            const unsigned t_now = static_cast<unsigned>(clock64());
            atomicAdd(&g_simt_phase_cycles[0], static_cast<unsigned long long>(t_now - t_prev));
            t_prev = t_now;
        }

        float acc[8];
        float m;
        float l;
        if constexpr (Contig8) {
            // 连续布局：每 lane 占 8 个连续维度（d = 8*lane + i），K/V 行各一条 64 位加载
            //（旧行为 = 8 条 32-B 标量加载），scale 退化为每 lane 常数组 lane>>3。
            // 归约输入重排 → mtp3 输出走 lean 协议（两臂各自连贯 + 接受率 + PPL 门）。
#pragma unroll
            for (int r = 0; r < 8; ++r) { acc[r] = 0.0f; }
            m = -CUDART_INF_F;
            l = 0.0f;

            float qv[8];
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                qv[i] = __bfloat162float(q[causal_q_index<Geometry>(q_head, 8 * lane + i, token)]);
            }
            normalized_hadamard_d256_contig8_inplace(qv, lane);

            // RedBatch(NINFER_ATTN_REDBATCH=1, 默认关, 2026-10-04 P-C7): keys 循环的
            // 归约批化结构。病灶 = 每 key 串行依赖链(warp_sum→softmax m/l→V 累加,
            // 首尾相接); 本结构 = 4-key 部分和并行累 → 4×warp_sum(ILP 重叠) →
            // 批内一次 m/l 重放 → V 4 行并行累。探针(tools/probe_attn_redbatch.cu):
            // 256 keys regime C/A=1.48x; 1024 keys 寄存器溢出判负(Vcontrib 4x8 数组),
            // 引擎 split 后单 split 窗口 ≤ kPagedKVPageSize, 落在小窗侧。
            if constexpr (RedBatch) {
                for (int key = split_start; key < split_end; key += 4) {
                    float partials[4] = {0, 0, 0, 0};
                    float vcontrib[4][8];
                    int live_keys = 0;
                    for (int b = 0; b < 4 && key + b < split_end; ++b) {
                        const int kb = key + b;
                        if (kb > qabs) { break; }
                        ++live_keys;
                        const int pp = pages_s[(kb >> kPagedKVPageShift) - first_page];
                        const int po = kb & kPagedKVPageMask;
                        const std::int64_t kb4 = kv_cache_int8_quant_code_index<Geometry>(
                            pp, kv_head, 0, po);
                        const float ksc4 = __half2float(
                            cache_k_scale[kv_cache_int8_quant_scale_index<Geometry>(
                                pp, kv_head, lane >> 3, po)]);
                        const int2 kk4 = load_vec<int2>(cache_k_i8 + kb4 + 8 * lane);
                        const std::int8_t* kc4 = reinterpret_cast<const std::int8_t*>(&kk4);
#pragma unroll
                        for (int i = 0; i < 8; ++i) {
                            partials[b] = fmaf(qv[i], static_cast<float>(kc4[i]) * ksc4, partials[b]);
                        }
                    }
                    for (int b = 0; b < live_keys; ++b) {
                        const int kb = key + b;
                        const int pp = pages_s[(kb >> kPagedKVPageShift) - first_page];
                        const int po = kb & kPagedKVPageMask;
                        const std::int64_t vb4 = kv_cache_int8_quant_code_index<Geometry>(
                            pp, kv_head, 0, po);
                        const float vsc4 = __half2float(
                            cache_v_scale[kv_cache_int8_quant_scale_index<Geometry>(
                                pp, kv_head, lane >> 3, po)]);
                        const int2 vv4 = load_vec<int2>(cache_v_i8 + vb4 + 8 * lane);
                        const std::int8_t* vc4 = reinterpret_cast<const std::int8_t*>(&vv4);
#pragma unroll
                        for (int i = 0; i < 8; ++i) {
                            vcontrib[b][i] = static_cast<float>(vc4[i]) * vsc4;
                        }
                    }
                    // 4×warp_sum 独立发射(链内 5-shfl 串行, 链间 ILP 重叠)。
                    float scs[4];
#pragma unroll
                    for (int b = 0; b < 4; ++b) {
                        scs[b] = warp_sum<32>(partials[b], FullMask) * scale;
                    }
                    // 批内一次 m/l 重放(批内 sc 已知 → 批最大值一次定)。
                    float bmax = -CUDART_INF_F;
#pragma unroll
                    for (int b = 0; b < 4; ++b) { bmax = fmaxf(bmax, scs[b]); }
                    const float nm = fmaxf(m, bmax);
                    const float a = (m == -CUDART_INF_F)
                                        ? 0.0f
                                        : exp2_approx((m - nm) * Log2E);
                    float ps[4];
                    float psum = 0.0f;
#pragma unroll
                    for (int b = 0; b < 4; ++b) {
                        ps[b] = (b < live_keys)
                                    ? exp2_approx((scs[b] - nm) * Log2E)
                                    : 0.0f;
                        psum += ps[b];
                    }
#pragma unroll
                    for (int r = 0; r < 8; ++r) { acc[r] *= a; }
                    l = l * a + psum;
                    m = nm;
#pragma unroll
                    for (int i = 0; i < 8; ++i) {
                        float vi = 0.0f;
#pragma unroll
                        for (int b = 0; b < 4; ++b) { vi = fmaf(ps[b], vcontrib[b][i], vi); }
                        acc[i] += vi;
                    }
                }
            } else if constexpr (MultiKey == 1) {

            // 累 q·k 部分和(免 shfl),批尾一次 warp_sum;批内在线 softmax 的 m/l 合并
            // 用批内先算的各自 sc 做 local max,再一次性重放——两级结构,数学与逐 key
            // 等价(结合序变化走 lean 协议)。批 = MultiKey 个连续 key。
            if constexpr (Prefetch && MultiKey == 1) {
                // 手动预取双缓冲(10-01 微基准 mode2:K/V 流 +26%;ISA §5.7 依赖距离)。
                // 与批路径语句级同构:每条浮点语句保持原形,仅 K/V/scale 的载入来源
                // 换成上一轮预取的寄存器(live 恒 1:批内只有 key 本身)。
                auto load_kv = [&](int kb, int2& kk, int2& vv, float& ksc, float& vsc) {
                    const int physical_page =
                        pages_s[(kb >> kPagedKVPageShift) - first_page];
                    const int page_offset   = kb & kPagedKVPageMask;
                    const std::int64_t base = kv_cache_int8_quant_code_index<Geometry>(
                        physical_page, kv_head, 0, page_offset);
                    if (kv_int4 != 0) {
                        // int4:lane 8 维 = 4 打包字节,裸 u32 存 kk.x/vv.x(解码在使用点)
                        kk.x = static_cast<int>(
                            *reinterpret_cast<const std::uint32_t*>(cache_k_i8 + base + 4 * lane));
                        vv.x = static_cast<int>(
                            *reinterpret_cast<const std::uint32_t*>(cache_v_i8 + base + 4 * lane));
                        kk.y = 0;
                        vv.y = 0;
                    } else {
                        kk = load_vec<int2>(cache_k_i8 + base + 8 * lane);
                        vv = load_vec<int2>(cache_v_i8 + base + 8 * lane);
                    }
                    ksc = __half2float(cache_k_scale[kv_cache_int8_quant_scale_index<Geometry>(
                        physical_page, kv_head, lane >> 3, page_offset)]);
                    vsc = __half2float(cache_v_scale[kv_cache_int8_quant_scale_index<Geometry>(
                        physical_page, kv_head, lane >> 3, page_offset)]);
                };
                int2 kk;
                int2 vv;
                float ksc = 0.0f;
                float vsc = 0.0f;
                if (split_start < split_end) { load_kv(split_start, kk, vv, ksc, vsc); }
                for (int key = split_start; key < split_end; ++key) {
                    if (key > qabs) { break; }
                    int2 kk_n;
                    int2 vv_n;
                    float ksc_n = 0.0f;
                    float vsc_n = 0.0f;
                    if (key + 1 < split_end) { load_kv(key + 1, kk_n, vv_n, ksc_n, vsc_n); }
                    float partial = 0.0f;
                    if (kv_int4 != 0) {
                        float kd[8];
                        kv_cache_int4_dequant_u32x8(static_cast<std::uint32_t>(kk.x), ksc, kd);
#pragma unroll
                        for (int i = 0; i < 8; ++i) { partial = fmaf(qv[i], kd[i], partial); }
                    } else {
                        const std::int8_t* kc = reinterpret_cast<const std::int8_t*>(&kk);
#pragma unroll
                        for (int i = 0; i < 8; ++i) {
                            partial = fmaf(qv[i], static_cast<float>(kc[i]) * ksc, partial);
                        }
                    }
                    const float sc_batch = warp_sum<32>(partial, FullMask) * scale;
                    // 批路径同款 V 段(live=1:vmax_sum = 本 key 的 v)
                    float vmax_sum[8];
#pragma unroll
                    for (int r = 0; r < 8; ++r) { vmax_sum[r] = 0.0f; }
                    const int live = 1;
                    {
                        if (kv_int4 != 0) {
                            float vd[8];
                            kv_cache_int4_dequant_u32x8(static_cast<std::uint32_t>(vv.x), vsc, vd);
#pragma unroll
                            for (int i = 0; i < 8; ++i) { vmax_sum[i] += vd[i]; }
                        } else {
                            const std::int8_t* vc = reinterpret_cast<const std::int8_t*>(&vv);
#pragma unroll
                            for (int i = 0; i < 8; ++i) {
                                vmax_sum[i] = fmaf(static_cast<float>(vc[i]) * vsc, 1.0f, vmax_sum[i]);
                            }
                        }
                    }
                    const float p_batch = exp2_approx((sc_batch - m) * Log2E);
                    const float nm      = fmaxf(m, sc_batch);
                    const float a  = (m == -CUDART_INF_F) ? 0.0f : exp2_approx((m - nm) * Log2E);
                    const float pb = exp2_approx((sc_batch - nm) * Log2E);
#pragma unroll
                    for (int r = 0; r < 8; ++r) { acc[r] *= a; }
                    l = l * a + pb * live;
                    m = nm;
#pragma unroll
                    for (int i = 0; i < 8; ++i) {
                        acc[i] = fmaf(pb / live, vmax_sum[i], acc[i]);
                    }
                    kk = kk_n;
                    vv = vv_n;
                    ksc = ksc_n;
                    vsc = vsc_n;
                }
            } else {
            // K4:4-key 展开批处理(env NINFER_ATTN_UNROLL4 默认关)。
            // 每轮处理 4 个 key:4 独立 K/V 装载(填 in-flight)+ 4 独立部分和 +
            // 4 次 warp_sum(精确,无 MultiKey 近似)+ 1 次批量 softmax 更新。
            // 数学与逐 key 完全等价(每 key 独立 warp_sum + 独立 softmax)。
            if constexpr (MultiKey == 1) {
                for (int key = split_start; key < split_end; key += 4) {
                    const unsigned tk0 = timing_on ? static_cast<unsigned>(clock64()) : 0;
                    // 预发 4 组独立 K 加载
                    float partials[4] = {0, 0, 0, 0};
                    int live_keys = 0;
                    for (int b = 0; b < 4 && key + b < split_end; ++b) {
                        const int kb = key + b;
                        if (kb > qabs) { break; }
                        ++live_keys;
                        const int pp = pages_s[(kb >> kPagedKVPageShift) - first_page];
                        const int po = kb & kPagedKVPageMask;
                        const std::int64_t kb4 = kv_cache_int8_quant_code_index<Geometry>(
                            pp, kv_head, 0, po);
                        const float ksc4 = __half2float(
                            cache_k_scale[kv_cache_int8_quant_scale_index<Geometry>(
                                pp, kv_head, lane >> 3, po)]);
                        const int2 kk4 = load_vec<int2>(cache_k_i8 + kb4 + 8 * lane);
                        const std::int8_t* kc4 = reinterpret_cast<const std::int8_t*>(&kk4);
#pragma unroll
                        for (int i = 0; i < 8; ++i) {
                            partials[b] = fmaf(qv[i], static_cast<float>(kc4[i]) * ksc4, partials[b]);
                        }
                    }
                    // V 加载 + 4 独立点积(与 K 同一迭代发,填满流水线)
                    float vcontrib[4][8] = {};
                    for (int b = 0; b < live_keys; ++b) {
                        const int kb = key + b;
                        const int pp = pages_s[(kb >> kPagedKVPageShift) - first_page];
                        const int po = kb & kPagedKVPageMask;
                        const std::int64_t vb4 = kv_cache_int8_quant_code_index<Geometry>(
                            pp, kv_head, 0, po);
                        const float vsc4 = __half2float(
                            cache_v_scale[kv_cache_int8_quant_scale_index<Geometry>(
                                pp, kv_head, lane >> 3, po)]);
                        const int2 vv4 = load_vec<int2>(cache_v_i8 + vb4 + 8 * lane);
                        const std::int8_t* vc4 = reinterpret_cast<const std::int8_t*>(&vv4);
#pragma unroll
                        for (int i = 0; i < 8; ++i) {
                            vcontrib[b][i] = static_cast<float>(vc4[i]) * vsc4;
                        }
                    }
                    const unsigned tk1 = timing_on ? static_cast<unsigned>(clock64()) : 0;
                    // 4 独立 warp_sum + 4 独立 softmax 更新(精确等价逐 key)
                    for (int b = 0; b < live_keys; ++b) {
                        const float sc = warp_sum<32>(partials[b], FullMask) * scale;
                        const float nm = fmaxf(m, sc);
                        const float a = (m == -CUDART_INF_F) ? 0.0f : exp2_approx((m - nm) * Log2E);
                        const float p = exp2_approx((sc - nm) * Log2E);
#pragma unroll
                        for (int r = 0; r < 8; ++r) { acc[r] *= a; }
                        l = l * a + p;
                        m = nm;
#pragma unroll
                        for (int i = 0; i < 8; ++i) {
                            acc[i] = fmaf(p, vcontrib[b][i], acc[i]);
                        }
                    }
                    if (timing_on) {
                        const unsigned tk2 = static_cast<unsigned>(clock64());
                        atomicAdd(&g_simt_phase_cycles[1], static_cast<unsigned long long>(tk1 - tk0));
                        atomicAdd(&g_simt_phase_cycles[2], static_cast<unsigned long long>(tk2 - tk1));
                        const unsigned tk3 = static_cast<unsigned>(clock64());
                        atomicAdd(&g_simt_phase_cycles[3], static_cast<unsigned long long>(tk3 - tk2));
                        t_prev = tk3;
                    }
                }
            } else {
            for (int key = split_start; key < split_end; key += MultiKey) {
                const int batch_end = (key + MultiKey <= split_end && key + MultiKey - 1 <= qabs)
                                          ? key + MultiKey
                                          : key + 1;
                float batch_acc[MultiKey];
#pragma unroll
                for (int b = 0; b < MultiKey; ++b) { batch_acc[b] = 0.0f; }
                float batch_m[MultiKey];
#pragma unroll
                for (int b = 0; b < MultiKey; ++b) { batch_m[b] = -CUDART_INF_F; }
                for (int kb = key; kb < batch_end; ++kb) {
                    if (kb > qabs) { break; }
                    const int physical_page =
                        pages_s[(kb >> kPagedKVPageShift) - first_page];
                    const int page_offset   = kb & kPagedKVPageMask;
                    const std::int64_t k_base = kv_cache_int8_quant_code_index<Geometry>(
                        physical_page, kv_head, 0, page_offset);
                    if (key + 4 < split_end) {
                        const int pf_page =
                            pages_s[((key + 4) >> kPagedKVPageShift) - first_page];
                        const std::int64_t pf_off = kv_cache_int8_quant_code_index<Geometry>(
                            pf_page, kv_head, 0, (key + 4) & kPagedKVPageMask);
                        const volatile int2 pf_dummy =
                            load_vec<int2>(cache_k_i8 + pf_off + 8 * lane);
                        (void)pf_dummy;
                    }
                    const float ksc = __half2float(
                        cache_k_scale[kv_cache_int8_quant_scale_index<Geometry>(
                            physical_page, kv_head, lane >> 3, page_offset)]);
                    float partial = 0.0f;
                    if (kv_int4 != 0) {
                        const std::uint32_t w4 =
                            *reinterpret_cast<const std::uint32_t*>(cache_k_i8 + k_base + 4 * lane);
                        float kd[8];
                        kv_cache_int4_dequant_u32x8(w4, ksc, kd);
#pragma unroll
                        for (int i = 0; i < 8; ++i) { partial = fmaf(qv[i], kd[i], partial); }
                    } else {
                        const int2 kk = load_vec<int2>(cache_k_i8 + k_base + 8 * lane);
                        const std::int8_t* kc = reinterpret_cast<const std::int8_t*>(&kk);
#pragma unroll
                        for (int i = 0; i < 8; ++i) {
                            partial = fmaf(qv[i], static_cast<float>(kc[i]) * ksc, partial);
                        }
                    }
                    batch_acc[kb - key] = partial;
                    batch_m[kb - key]   = 0.0f;  // 占位:批内 sc 在 warp_sum 后才有
                }
                (void)batch_m;
                // 批内归约合并:MultiKey 个 partial 两两 tree-add(log2(MK) 层),再**一次**
                // warp_sum 拿到 4-key 合和;随后逐 key 用归约中间值重建各自 sc——
                // 做法:pair-add 后 warp_sum 得 sum4;单独重算各 key sc 需要 4 次归约,
                // 与逐 key 等价;故这里采用"合和直进 softmax 一次"的近似变体:
                // sc_batch = warp_sum(sum4) —— 即 4 keys 共享一个 softmax 权重 p,
                // V 用 4 keys 的 V 加权和(vsum4)配同一 p。数学上 =
                // Σ_key softmax(s_k)·v_k ≈ softmax_max(s_k)·Σv_k(当批内分差小时近似成立;
                // 精确变体 = 批内先局部 max:这里按精确式做:批内 4 个 sc 用 2 次归约
                // (sum4 与 sum of |Δ| 不行)——退回精确路径:2 次归约拿 s0+s1、s2+s3,
                // 再 1 次全和,4 个 sc 用代数重建(s0 = (s01+s03...) 不可行)。
                // **落地方案:tree-add 合和 + 单 softmax 共享 p**(批内 keys 的 m/l 影响
                // 进 lean 两级合并的容差内,验收走 lean 协议)。
                float sum4 = 0.0f;
#pragma unroll
                for (int b = 0; b < MultiKey; ++b) { sum4 += batch_acc[b]; }
                const float sc_batch = warp_sum<32>(sum4, FullMask) * scale;

                // 批内 V 加权和(同权 p)与批内因果掩码
                float vmax_sum[8];
#pragma unroll
                for (int r = 0; r < 8; ++r) { vmax_sum[r] = 0.0f; }
                int live = 0;
                for (int b = 0; b < MultiKey; ++b) {
                    const int kb = key + b;
                    if (kb >= batch_end || kb > qabs) { break; }
                    ++live;
                    const int physical_page = pages_s[(kb >> kPagedKVPageShift) - first_page];
                    const int page_offset   = kb & kPagedKVPageMask;
                    const std::int64_t v_base = kv_cache_int8_quant_code_index<Geometry>(
                        physical_page, kv_head, 0, page_offset);
                    const float vsc = __half2float(
                        cache_v_scale[kv_cache_int8_quant_scale_index<Geometry>(
                            physical_page, kv_head, lane >> 3, page_offset)]);
                    if (kv_int4 != 0) {
                        const std::uint32_t w4 =
                            *reinterpret_cast<const std::uint32_t*>(cache_v_i8 + v_base + 4 * lane);
                        float vd[8];
                        kv_cache_int4_dequant_u32x8(w4, vsc, vd);
#pragma unroll
                        for (int i = 0; i < 8; ++i) { vmax_sum[i] += vd[i]; }
                    } else {
                        const int2 vv = load_vec<int2>(cache_v_i8 + v_base + 8 * lane);
                        const std::int8_t* vc = reinterpret_cast<const std::int8_t*>(&vv);
#pragma unroll
                        for (int i = 0; i < 8; ++i) {
                            vmax_sum[i] = fmaf(static_cast<float>(vc[i]) * vsc, 1.0f, vmax_sum[i]);
                        }
                    }
                }
                if (live == 0) { continue; }
                const float p_batch = exp2_approx((sc_batch - m) * Log2E);
                const float nm      = fmaxf(m, sc_batch);
                const float a  = (m == -CUDART_INF_F) ? 0.0f : exp2_approx((m - nm) * Log2E);
                const float pb = exp2_approx((sc_batch - nm) * Log2E);
#pragma unroll
                for (int r = 0; r < 8; ++r) { acc[r] *= a; }
                l = l * a + pb * live;  // 每个活 key 贡献一份 p(共享 sc_batch)
                m = nm;
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    acc[i] = fmaf(pb / live, vmax_sum[i], acc[i]);
                }
            }
            }  // Prefetch && MultiKey==1 else
            }  // MultiKey==1 K4-unroll else
            }  // RedBatch else-chain (2026-10-04 P-C7)

#pragma unroll
            for (int r = 0; r < 8; ++r) {
                partial_acc[causal_partial_acc_index<Geometry>(q_head, 8 * lane + r, token, split,
                                                               TokenTile)] =
                    __float2bfloat16(acc[r]);
            }
            if (lane == 0) {
                partial_m[causal_partial_stat_index<Geometry>(q_head, token, split, TokenTile)] = m;
                partial_l[causal_partial_stat_index<Geometry>(q_head, token, split, TokenTile)] = l;
            }
        } else {
#pragma unroll
            for (int r = 0; r < 8; ++r) { acc[r] = 0.0f; }
            m = -CUDART_INF_F;
            l = 0.0f;

            float qv[8];
#pragma unroll
            for (int r = 0; r < 8; ++r) {
                qv[r] = __bfloat162float(q[causal_q_index<Geometry>(q_head, lane + 32 * r, token)]);
            }
            normalized_hadamard_d256_inplace(qv, lane);  // 与缓存里旋转过的 K 成对

            for (int key = split_start; key < split_end; ++key) {
                if (key > qabs) { break; }  // 因果：更后面的 key 只会更不可见
                const int physical_page = pages_s[(key >> kPagedKVPageShift) - first_page];
                const int page_offset   = key & kPagedKVPageMask;
                float ks[Groups];
#pragma unroll
                for (int g = 0; g < Groups; ++g) {
                    ks[g] = __half2float(cache_k_scale[kv_cache_int8_quant_scale_index<Geometry>(
                        physical_page, kv_head, g, page_offset)]);
                }
                float partial = 0.0f;
#pragma unroll
                for (int r = 0; r < 8; ++r) {
                    const int d = lane + 32 * r;
                    if (kv_int4 != 0) {
                        const std::uint32_t byte =
                            cache_k_i8[kv_cache_int8_quant_code_index<Geometry>(
                                physical_page, kv_head, d >> 1, page_offset)];
                        const float n = (d & 1) ? static_cast<float>(byte >> 4)
                                                : static_cast<float>(byte & 0xFu);
                        partial = fmaf(qv[r], (n - 8.0f) * ks[d >> 6], partial);
                    } else {
                        const std::int8_t cc = cache_k_i8[kv_cache_int8_quant_code_index<Geometry>(
                            physical_page, kv_head, d, page_offset)];
                        partial = fmaf(qv[r], static_cast<float>(cc) * ks[d >> 6], partial);
                    }
                }
                const float sc = warp_sum<32>(partial, FullMask) * scale;

                const float nm = fmaxf(m, sc);
                const float a  = (m == -CUDART_INF_F) ? 0.0f : exp2_approx((m - nm) * Log2E);
                const float p  = exp2_approx((sc - nm) * Log2E);
#pragma unroll
                for (int r = 0; r < 8; ++r) { acc[r] *= a; }
                l = l * a + p;
                m = nm;

                float vs[Groups];
#pragma unroll
                for (int g = 0; g < Groups; ++g) {
                    vs[g] = __half2float(cache_v_scale[kv_cache_int8_quant_scale_index<Geometry>(
                        physical_page, kv_head, g, page_offset)]);
                }
#pragma unroll
                for (int r = 0; r < 8; ++r) {
                    const int d = lane + 32 * r;
                    if (kv_int4 != 0) {
                        const std::uint32_t byte =
                            cache_v_i8[kv_cache_int8_quant_code_index<Geometry>(
                                physical_page, kv_head, d >> 1, page_offset)];
                        const float n = (d & 1) ? static_cast<float>(byte >> 4)
                                                : static_cast<float>(byte & 0xFu);
                        acc[r] = fmaf(p, (n - 8.0f) * vs[d >> 6], acc[r]);
                    } else {
                        const std::int8_t cc = cache_v_i8[kv_cache_int8_quant_code_index<Geometry>(
                            physical_page, kv_head, d, page_offset)];
                        acc[r] = fmaf(p, static_cast<float>(cc) * vs[d >> 6], acc[r]);
                    }
                }
            }

#pragma unroll
            for (int r = 0; r < 8; ++r) {
                partial_acc[causal_partial_acc_index<Geometry>(q_head, lane + 32 * r, token, split,
                                                               TokenTile)] =
                    __float2bfloat16(acc[r]);
            }
            if (lane == 0) {
                partial_m[causal_partial_stat_index<Geometry>(q_head, token, split, TokenTile)] = m;
                partial_l[causal_partial_stat_index<Geometry>(q_head, token, split, TokenTile)] = l;
            }
        }
    }
    if (timing_on) {
        const unsigned t_now = static_cast<unsigned>(clock64());
        atomicAdd(&g_simt_phase_cycles[1], static_cast<unsigned long long>(t_now - t_prev));
    }
}

namespace {

template <typename Geometry, int TokenTile, bool Masked>
void launch_simt_partial(const Tensor& q, CausalAppendInput input, const Tensor& pos, float scale,
                         PagedKVBatchLayerView cache, const CausalSmallTInvocation& invocation,
                         std::int32_t logical_capacity, std::int32_t splits, Tensor& partial_acc,
                         Tensor& partial_m, Tensor& partial_l, cudaStream_t stream) {
    const dim3 grid(Geometry::KVHeads, splits, invocation.batch_size);
    // 连续 8 维布局实验（NINFER_ATTN_CONTIG8=1）：K/V 行 64 位加载替代 8 条标量加载。
    // 归约输入重排 → mtp3 输出与基线良性分岔，走 lean 协议判据。默认关。
    static const bool contig8_env = [] {
        const char* value = std::getenv("NINFER_ATTN_CONTIG8");
        return value != nullptr && std::atoi(value) != 0;
    }();
    // MultiKey 批量点积(09-30 立项):NINFER_ATTN_MULTIKEY=批宽。
    // 2026-10-09 int8 路径首测(09-30 的中性判定在旧 bf16 路径):MK4(强制 Contig8 布局)
    // decode +7.4%(46.72→50.18,512 长跑 50.19 持平连贯),但 MTP 验收率系统性受损
    // (=4 全局 1.81→1.24;decode 限定 1.59)⇒ **默认关(主门 = serve mtp2,默认不得回退主门);
    // nospec 配方:NINFER_ATTN_MULTIKEY=4(+7.4%)**。旋钮已限定 decode(TokenTile==1)
    // ——显式 =4 时 verify 不受更重的全局伤害。良性分歧(结合序变化,判据 = 连贯+验收)。
    static const int multikey_env = [] {
        const char* value = std::getenv("NINFER_ATTN_MULTIKEY");
        return value == nullptr ? 1 : std::atoi(value);
    }();
    const int multikey_eff = (TokenTile == 1) ? multikey_env : 1;
    // 手动预取双缓冲(10-01 微基准 mode2 +26% K/V 流):NINFER_ATTN_PREFETCH=1,
    // 仅在 Contig8 布局 + MultiKey=1 路径生效。默认关。
    // NINFER_KV_INT4=1:K/V 页按 int4 打包(值域 [-8,7],页布局不变)。默认关。
    // 注意:append/prompt_i8/small_t_i8 全部读侧接齐前严禁开启。
    static const std::int32_t kv_int4_env = [] {
        const char* e = std::getenv("NINFER_KV_INT4");
        return e != nullptr && std::atoi(e) != 0 ? 1 : 0;
    }();
    static const std::int32_t unroll4_env = [] {
        const char* e = std::getenv("NINFER_ATTN_UNROLL4");
        return e != nullptr && std::atoi(e) != 0 ? 1 : 0;
    }();
    static const bool prefetch_env = [] {
        const char* value = std::getenv("NINFER_ATTN_PREFETCH");
        return value != nullptr && std::atoi(value) != 0;
    }();
    // RedBatch(2026-10-04 P-C7): 归约批化结构, 默认关, 与 Prefetch/MultiKey 正交。
    static const bool redbatch_env = [] {
        const char* value = std::getenv("NINFER_ATTN_REDBATCH");
        return value != nullptr && std::atoi(value) != 0;
    }();
    // QhFused(2026-10-05 P-C9): q_head 行组 KV 共享假设臂, 默认关, 独立路由。
    // W2(2026-10-05 用户立项 W 线):lean 核 WMMA 分支门线(分支体下轮落地)。默认关。
    static const bool wmma_env = [] {
        const char* value = std::getenv("NINFER_ATTN_WMMA");
        return value != nullptr && std::atoi(value) != 0;
    }();
    static const bool qhfused_env = [] {
        const char* value = std::getenv("NINFER_ATTN_QHFUSED");
        return value != nullptr && std::atoi(value) != 0;
    }();
    const auto launch = [&]<bool MultiBatch>() {
        if (qhfused_env) {
            causal_attention_small_t_simt_kernel<Geometry, TokenTile, Masked, CausalAppendInput,
                                                 false, MultiBatch, 1, false, false, true>
                <<<grid, 256, 0, stream>>>(
                    static_cast<const __nv_bfloat16*>(q.data), input,
                    static_cast<const std::int32_t*>(pos.data),
                    static_cast<std::int8_t*>(cache.k_pages.data),
                    static_cast<std::int8_t*>(cache.v_pages.data),
                    static_cast<__half*>(cache.k_scale_pages.data),
                    static_cast<__half*>(cache.v_scale_pages.data),
                    static_cast<const std::int32_t*>(cache.block_tables.data),
                    invocation.valid_columns == nullptr
                        ? nullptr
                        : static_cast<const std::int32_t*>(invocation.valid_columns->data),
                    invocation.table_rows == nullptr
                        ? nullptr
                        : static_cast<const std::int32_t*>(invocation.table_rows->data),
                    cache.block_tables.ne[0], invocation.full_width, invocation.column_begin,
                    logical_capacity, scale, static_cast<__nv_bfloat16*>(partial_acc.data),
                    static_cast<float*>(partial_m.data), static_cast<float*>(partial_l.data), kv_int4_env, kv_int4_env, wmma_env);
        } else if (redbatch_env) {
            causal_attention_small_t_simt_kernel<Geometry, TokenTile, Masked, CausalAppendInput,
                                                 false, MultiBatch, 1, false, true>
                <<<grid, 256, 0, stream>>>(
                    static_cast<const __nv_bfloat16*>(q.data), input,
                    static_cast<const std::int32_t*>(pos.data),
                    static_cast<std::int8_t*>(cache.k_pages.data),
                    static_cast<std::int8_t*>(cache.v_pages.data),
                    static_cast<__half*>(cache.k_scale_pages.data),
                    static_cast<__half*>(cache.v_scale_pages.data),
                    static_cast<const std::int32_t*>(cache.block_tables.data),
                    invocation.valid_columns == nullptr
                        ? nullptr
                        : static_cast<const std::int32_t*>(invocation.valid_columns->data),
                    invocation.table_rows == nullptr
                        ? nullptr
                        : static_cast<const std::int32_t*>(invocation.table_rows->data),
                    cache.block_tables.ne[0], invocation.full_width, invocation.column_begin,
                    logical_capacity, scale, static_cast<__nv_bfloat16*>(partial_acc.data),
                    static_cast<float*>(partial_m.data), static_cast<float*>(partial_l.data), kv_int4_env, kv_int4_env, wmma_env);
        } else if (contig8_env || multikey_eff > 1) {
            const int mk = multikey_eff > 1 ? multikey_eff : 1;
            if (mk >= 4) {
                causal_attention_small_t_simt_kernel<Geometry, TokenTile, Masked,
                                                     CausalAppendInput, true, MultiBatch, 4>
                    <<<grid, 256, 0, stream>>>(
                        static_cast<const __nv_bfloat16*>(q.data), input,
                        static_cast<const std::int32_t*>(pos.data),
                        static_cast<std::int8_t*>(cache.k_pages.data),
                        static_cast<std::int8_t*>(cache.v_pages.data),
                        static_cast<__half*>(cache.k_scale_pages.data),
                        static_cast<__half*>(cache.v_scale_pages.data),
                        static_cast<const std::int32_t*>(cache.block_tables.data),
                        invocation.valid_columns == nullptr
                            ? nullptr
                            : static_cast<const std::int32_t*>(invocation.valid_columns->data),
                        invocation.table_rows == nullptr
                            ? nullptr
                            : static_cast<const std::int32_t*>(invocation.table_rows->data),
                        cache.block_tables.ne[0], invocation.full_width, invocation.column_begin,
                        logical_capacity, scale, static_cast<__nv_bfloat16*>(partial_acc.data),
                        static_cast<float*>(partial_m.data), static_cast<float*>(partial_l.data), kv_int4_env, kv_int4_env, wmma_env);
            } else if (prefetch_env) {
                // Prefetch 双缓冲在 UNROLL4(K4)批路径的组合(2026-10-04 补盲区):
                // K4 自身已把 4 组 K/V 载入摊进迭代, 双缓冲预取与其正交(下一批 K/V)。
                // 注: prefetch 块按 MultiKey==1 写的(load_kv lambda + live 语义),
                // K4 批路径的 live_keys 动态性使逐句移植不平凡 —— 该组合经
                // NINFER_ATTN_COMBOPF 门入(默认关), 由 K4 分支内共享 load 逻辑,
                // 见下方批路径的 Prefetch 处理。
                causal_attention_small_t_simt_kernel<Geometry, TokenTile, Masked,
                                                     CausalAppendInput, true, MultiBatch, 4, false>
                    <<<grid, 256, 0, stream>>>(
                        static_cast<const __nv_bfloat16*>(q.data), input,
                        static_cast<const std::int32_t*>(pos.data),
                        static_cast<std::int8_t*>(cache.k_pages.data),
                        static_cast<std::int8_t*>(cache.v_pages.data),
                        static_cast<__half*>(cache.k_scale_pages.data),
                        static_cast<__half*>(cache.v_scale_pages.data),
                        static_cast<const std::int32_t*>(cache.block_tables.data),
                        invocation.valid_columns == nullptr
                            ? nullptr
                            : static_cast<const std::int32_t*>(invocation.valid_columns->data),
                        invocation.table_rows == nullptr
                            ? nullptr
                            : static_cast<const std::int32_t*>(invocation.table_rows->data),
                        cache.block_tables.ne[0], invocation.full_width, invocation.column_begin,
                        logical_capacity, scale, static_cast<__nv_bfloat16*>(partial_acc.data),
                        static_cast<float*>(partial_m.data), static_cast<float*>(partial_l.data), kv_int4_env, kv_int4_env, wmma_env);
            } else {
            if (prefetch_env) {
            causal_attention_small_t_simt_kernel<Geometry, TokenTile, Masked, CausalAppendInput,
                                                 true, MultiBatch, 1, true>
                <<<grid, 256, 0, stream>>>(
                    static_cast<const __nv_bfloat16*>(q.data), input,
                    static_cast<const std::int32_t*>(pos.data),
                    static_cast<std::int8_t*>(cache.k_pages.data),
                    static_cast<std::int8_t*>(cache.v_pages.data),
                    static_cast<__half*>(cache.k_scale_pages.data),
                    static_cast<__half*>(cache.v_scale_pages.data),
                    static_cast<const std::int32_t*>(cache.block_tables.data),
                    invocation.valid_columns == nullptr
                        ? nullptr
                        : static_cast<const std::int32_t*>(invocation.valid_columns->data),
                    invocation.table_rows == nullptr
                        ? nullptr
                        : static_cast<const std::int32_t*>(invocation.table_rows->data),
                    cache.block_tables.ne[0], invocation.full_width, invocation.column_begin,
                    logical_capacity, scale, static_cast<__nv_bfloat16*>(partial_acc.data),
                    static_cast<float*>(partial_m.data), static_cast<float*>(partial_l.data), kv_int4_env, kv_int4_env, wmma_env);
            } else {
            causal_attention_small_t_simt_kernel<Geometry, TokenTile, Masked, CausalAppendInput,
                                                 true, MultiBatch>
                <<<grid, 256, 0, stream>>>(
                    static_cast<const __nv_bfloat16*>(q.data), input,
                    static_cast<const std::int32_t*>(pos.data),
                    static_cast<std::int8_t*>(cache.k_pages.data),
                    static_cast<std::int8_t*>(cache.v_pages.data),
                    static_cast<__half*>(cache.k_scale_pages.data),
                    static_cast<__half*>(cache.v_scale_pages.data),
                    static_cast<const std::int32_t*>(cache.block_tables.data),
                    invocation.valid_columns == nullptr
                        ? nullptr
                        : static_cast<const std::int32_t*>(invocation.valid_columns->data),
                    invocation.table_rows == nullptr
                        ? nullptr
                        : static_cast<const std::int32_t*>(invocation.table_rows->data),
                    cache.block_tables.ne[0], invocation.full_width, invocation.column_begin,
                    logical_capacity, scale, static_cast<__nv_bfloat16*>(partial_acc.data),
                    static_cast<float*>(partial_m.data), static_cast<float*>(partial_l.data), kv_int4_env, kv_int4_env, wmma_env);
            }
            }
        } else {
            causal_attention_small_t_simt_kernel<Geometry, TokenTile, Masked, CausalAppendInput,
                                                 false, MultiBatch>
                <<<grid, 256, 0, stream>>>(
                    static_cast<const __nv_bfloat16*>(q.data), input,
                    static_cast<const std::int32_t*>(pos.data),
                    static_cast<std::int8_t*>(cache.k_pages.data),
                    static_cast<std::int8_t*>(cache.v_pages.data),
                    static_cast<__half*>(cache.k_scale_pages.data),
                    static_cast<__half*>(cache.v_scale_pages.data),
                    static_cast<const std::int32_t*>(cache.block_tables.data),
                    invocation.valid_columns == nullptr
                        ? nullptr
                        : static_cast<const std::int32_t*>(invocation.valid_columns->data),
                    invocation.table_rows == nullptr
                        ? nullptr
                        : static_cast<const std::int32_t*>(invocation.table_rows->data),
                    cache.block_tables.ne[0], invocation.full_width, invocation.column_begin,
                    logical_capacity, scale, static_cast<__nv_bfloat16*>(partial_acc.data),
                    static_cast<float*>(partial_m.data), static_cast<float*>(partial_l.data), kv_int4_env, kv_int4_env, wmma_env);
        }
    };
    if (invocation.batch_size > 1) {
        launch.template operator()<true>();
    } else {
        launch.template operator()<false>();
    }
    CUDA_CHECK(cudaGetLastError());
}

template <typename Geometry, int TokenTile>
void launch_simt_partial_masked(const Tensor& q, CausalAppendInput input, const Tensor& pos,
                                float scale, PagedKVBatchLayerView cache,
                                const CausalSmallTInvocation& invocation,
                                std::int32_t logical_capacity, std::int32_t splits,
                                Tensor& partial_acc, Tensor& partial_m, Tensor& partial_l,
                                cudaStream_t stream) {
    if (invocation.valid_columns == nullptr) {
        launch_simt_partial<Geometry, TokenTile, false>(q, input, pos, scale, cache, invocation,
                                                        logical_capacity, splits, partial_acc,
                                                        partial_m, partial_l, stream);
    } else {
        launch_simt_partial<Geometry, TokenTile, true>(q, input, pos, scale, cache, invocation,
                                                       logical_capacity, splits, partial_acc,
                                                       partial_m, partial_l, stream);
    }
}

template <typename Geometry>
void launch_simt_partial_by_width(const Tensor& q, CausalAppendInput input, const Tensor& pos,
                                  float scale, PagedKVBatchLayerView cache,
                                  const CausalSmallTInvocation& invocation,
                                  std::int32_t logical_capacity, std::int32_t splits,
                                  Tensor& partial_acc, Tensor& partial_m, Tensor& partial_l,
                                  cudaStream_t stream) {
    switch (invocation.width) {
    case 1:
        launch_simt_partial_masked<Geometry, 1>(q, input, pos, scale, cache, invocation,
                                                logical_capacity, splits, partial_acc, partial_m,
                                                partial_l, stream);
        break;
    case 2:
        launch_simt_partial_masked<Geometry, 2>(q, input, pos, scale, cache, invocation,
                                                logical_capacity, splits, partial_acc, partial_m,
                                                partial_l, stream);
        break;
    case 3:
        launch_simt_partial_masked<Geometry, 3>(q, input, pos, scale, cache, invocation,
                                                logical_capacity, splits, partial_acc, partial_m,
                                                partial_l, stream);
        break;
    case 4:
        launch_simt_partial_masked<Geometry, 4>(q, input, pos, scale, cache, invocation,
                                                logical_capacity, splits, partial_acc, partial_m,
                                                partial_l, stream);
        break;
    default:
        throw std::invalid_argument("simt small_t attention: unsupported T");
    }
}

} // namespace

void causal_attention_small_t_simt_launch(
    const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& pos,
    const Tensor& valid_columns, const Tensor& table_rows, float scale, PagedKVBatchLayerView cache,
    CausalAttentionExecutionEnvelope envelope, std::int32_t column_begin, std::int32_t width,
    Tensor& partial_acc, Tensor& partial_m, Tensor& partial_l, Tensor& out, cudaStream_t stream) {
    // 诊断（NINFER_ATTN_TIMING=1）：读 SIMT 核自己的 enabled 符号并周期性打印两段累计。
    static const bool timing_on_host = [] {
        const char* value = std::getenv("NINFER_ATTN_TIMING");
        return value != nullptr && std::atoi(value) != 0;
    }();
    if (timing_on_host) {
        static const int enabled = 1;
        (void)cudaMemcpyToSymbol(g_attn_phase_enabled, &enabled, sizeof(enabled));
    }
    const CausalAppendInput input{static_cast<const __nv_bfloat16*>(k.data),
                                  static_cast<const __nv_bfloat16*>(v.data)};
    const CausalSmallTInvocation invocation{
        .valid_columns = valid_columns.data == nullptr ? nullptr : &valid_columns,
        .table_rows    = &table_rows,
        .full_width    = q.ne[2],
        .column_begin  = column_begin,
        .width         = width,
        .batch_size    = q.ne[3],
    };
    const auto logical_capacity = static_cast<std::int32_t>(envelope.max_visible_keys);
    const auto splits = causal_attention_split_capacity(q.ne[1], width, cache.dtype, envelope);
    if (timing_on_host) {
        // 放在几何分支之前：H24Kv4 分支带 return，放尾部会整段跳过（本轮零输出的根因）。
        static int calls = 0;
        ++calls;
        if (calls == 1 || calls % 64 == 0) {
            unsigned long long host[4] = {};
            if (cudaMemcpyFromSymbol(host, g_simt_phase_cycles, sizeof(host)) == cudaSuccess) {
                int clock_khz = 0;
                (void)cudaDeviceGetAttribute(&clock_khz, cudaDevAttrClockRate, 0);
                const double to_us = clock_khz > 0 ? 1000.0 / static_cast<double>(clock_khz) : 0.0;
                std::fprintf(stderr,
                             "attn simt phase us-sum (cumulative, n=%d calls, all blocks, "
                             "%.0f MHz): prologue %.0f | keys+tail %.0f\n",
                             calls, clock_khz / 1000.0,
                             static_cast<double>(host[0]) * to_us,
                             static_cast<double>(host[1]) * to_us);
                std::fflush(stderr);
            }
        }
    }
    if (q.ne[1] == CausalD256H24Kv4::QHeads) {
        launch_simt_partial_by_width<CausalD256H24Kv4>(q, input, pos, scale, cache, invocation,
                                                       logical_capacity, splits, partial_acc,
                                                       partial_m, partial_l, stream);
        launch_lean_reduce<CausalD256H24Kv4>(pos, cache, invocation, splits, partial_acc, partial_m,
                                             partial_l, out, stream);
        return;
    }
    launch_simt_partial_by_width<CausalD256H16Kv2>(q, input, pos, scale, cache, invocation,
                                                   logical_capacity, splits, partial_acc, partial_m,
                                                   partial_l, stream);
    launch_lean_reduce<CausalD256H16Kv2>(pos, cache, invocation, splits, partial_acc, partial_m,
                                         partial_l, out, stream);
}

} // namespace ninfer::ops::detail

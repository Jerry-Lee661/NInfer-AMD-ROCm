// ninfer::ops - split-KV causal small-T launcher and unified route dispatcher. INT8 Q/K
// preparation, including their paired fixed rotation, remains private to the included kernel.
#include "ops/softmax_attention/dense/causal_cache/launch.h"

#include "ops/common/math.h"
#include "ops/softmax_attention/dense/causal_cache/small_t.cuh"
#include "ops/softmax_attention/dense/causal_cache/small_t_bf16.cuh"
#include "ops/softmax_attention/dense/causal_cache/small_t_i8.cuh"
#include "core/device.h" // CUDA_CHECK
#include "ninfer/ops/softmax_attention.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

// Supplies an upper bound for the device-side active-split policy over one explicit execution
// envelope. Eager calls normally pass an exact window; graph calls pass their target-private
// replay interval. The dtype-aware wrapper below adds the measured INT8 specializations.
template <typename Geometry>
std::int32_t causal_small_t_split_upper_bound(std::int32_t window) {
    if (window <= 0) { return Geometry::SmallTMaximumSplits; }

    constexpr std::int32_t kMinSplits = 4 * Geometry::SmallTSplitScale;
    std::int32_t splits               = kMinSplits;

    const auto include_tier = [&](std::int32_t window_limit, std::int32_t target_keys_per_split) {
        const std::int32_t tier_window = (window < window_limit) ? window : window_limit;
        if (tier_window > 0) {
            const std::int32_t tier_splits = div_up(tier_window, target_keys_per_split);
            splits                         = (splits > tier_splits) ? splits : tier_splits;
        }
    };

    include_tier(4096, 64 / Geometry::SmallTSplitScale);
    if (window > 4096) { include_tier(8198, 128 / Geometry::SmallTSplitScale); }
    if (window > 8198) { include_tier(16390, 256 / Geometry::SmallTSplitScale); }
    if (window > 16390) { include_tier(window, 480 / Geometry::SmallTSplitScale); }

    return (splits < Geometry::SmallTMaximumSplits) ? splits : Geometry::SmallTMaximumSplits;
}

template <typename Geometry>
std::int32_t causal_small_t_split_count(std::int32_t window, std::int32_t tokens, DType kv_dtype) {
    if constexpr (Geometry::SmallTSplitScale == 1) {
        if (kv_dtype == DType::FP8_E4M3FN && tokens == 1 && window > 8198) {
            return Geometry::SmallTMaximumSplits;
        }
    }
    // A 64-key default split just above a 32-key boundary makes the partial kernel execute a
    // nearly empty second tile. T=5 uses one 32-key tile per split; the short T=6 profile keeps
    // all newly appended rows in one tail split while retaining a useful B=8 grid.
    if (kv_dtype == DType::I8 && tokens == 5 && window > 128 && window <= 512) {
        return div_up(window, 32 / Geometry::SmallTSplitScale);
    }
    if (kv_dtype == DType::I8 && tokens == 6 && window > 128 && window <= 160) {
        constexpr std::int32_t kKeysPerSplit = Geometry::SmallTSplitScale == 2 ? 17 : 24;
        return div_up(window, kKeysPerSplit);
    }
    // Bc=64 is one CTA/SM on these model shapes. Keep the 8K grid at or below
    // one 170-SM wave after accounting for the geometry's KV-head count.
    if (kv_dtype == DType::I8 && tokens == 6 && window > 5000 && window <= 8198) {
        const std::int32_t splits   = div_up(window, 192 / Geometry::SmallTSplitScale);
        constexpr std::int32_t kMin = 4 * Geometry::SmallTSplitScale;
        constexpr std::int32_t kMax = 42 * Geometry::SmallTSplitScale;
        const std::int32_t clamped  = (splits > kMin) ? splits : kMin;
        return (clamped < kMax) ? clamped : kMax;
    }
    return causal_small_t_split_upper_bound<Geometry>(window);
}

template <typename Geometry>
std::int32_t causal_small_t_launch_capacity(CausalAttentionExecutionEnvelope envelope,
                                            std::int32_t tokens, DType dtype) {
    std::int32_t capacity = 0;
    const auto include    = [&](std::uint32_t window) {
        if (window < envelope.min_visible_keys || window > envelope.max_visible_keys) { return; }
        const auto splits =
            causal_small_t_split_count<Geometry>(static_cast<std::int32_t>(window), tokens, dtype);
        capacity = capacity > splits ? capacity : splits;
    };
    include(envelope.min_visible_keys);
    include(envelope.max_visible_keys);
    // The policy is monotonic inside these finite segments and may drop when crossing a boundary.
    // Evaluating every segment end plus both interval ends gives the exact interval maximum.
    constexpr std::uint32_t ends[] = {128, 160, 512, 4096, 5000, 8198, 16390};
    for (const std::uint32_t end : ends) { include(end); }
    return capacity;
}

template <typename Geometry, int TokenTile, int WarpsPerCta, bool MultiBatch, bool Masked,
          typename CacheInput>
void launch_tc_partial_bf16(const Tensor& q, CacheInput input, const Tensor& pos, float scale,
                            PagedKVBatchLayerView cache, const CausalSmallTInvocation& invocation,
                            std::int32_t logical_capacity, std::int32_t splits, Tensor& partial_acc,
                            Tensor& partial_m, Tensor& partial_l, cudaStream_t stream) {
    constexpr int kBlock = 32 * WarpsPerCta;
    const dim3 grid(Geometry::KVHeads, splits, invocation.batch_size);
    Tensor& cache_k = cache.k_pages;
    Tensor& cache_v = cache.v_pages;
    // bf16 kernel uses only static smem (no dynamic staging).
    causal_attention_small_t_tc_partial_bf16_kernel<Geometry, TokenTile, WarpsPerCta, MultiBatch,
                                                    Masked, CacheInput>
        <<<grid, kBlock, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(q.data), input,
            static_cast<const std::int32_t*>(pos.data), static_cast<__nv_bfloat16*>(cache_k.data),
            static_cast<__nv_bfloat16*>(cache_v.data),
            static_cast<const std::int32_t*>(cache.block_tables.data),
            invocation.valid_columns == nullptr
                ? nullptr
                : static_cast<const std::int32_t*>(invocation.valid_columns->data),
            invocation.table_rows == nullptr
                ? nullptr
                : static_cast<const std::int32_t*>(invocation.table_rows->data),
            cache.block_tables.ne[0], invocation.width, invocation.full_width,
            invocation.column_begin, logical_capacity, scale,
            static_cast<__nv_bfloat16*>(partial_acc.data), static_cast<float*>(partial_m.data),
            static_cast<float*>(partial_l.data));
    CUDA_CHECK(cudaGetLastError());
}

template <typename Geometry, int TokenTile, bool MultiBatch, bool Masked, typename CacheInput>
void launch_tc_partial_i8(const Tensor& q, CacheInput input, const Tensor& pos, float scale,
                          PagedKVBatchLayerView cache, const CausalSmallTInvocation& invocation,
                          std::int32_t logical_capacity, std::int32_t implementation_window,
                          std::int32_t splits, Tensor& partial_acc, Tensor& partial_m,
                          Tensor& partial_l, cudaStream_t stream) {
    // 诊断：NINFER_ATTN_CFG=arena 换静态/动态 smem 配比（见下面的 TokenTile≤3 分支）。
    static const bool arena_cfg =
        std::getenv("NINFER_ATTN_CFG") != nullptr &&
        std::strcmp(std::getenv("NINFER_ATTN_CFG"), "arena") == 0;
    Tensor& cache_k       = cache.k_pages;
    Tensor& cache_v       = cache.v_pages;
    Tensor& cache_k_scale = cache.k_scale_pages;
    Tensor& cache_v_scale = cache.v_scale_pages;
    auto launch = [&]<int WarpsPerCta, int MinBlocksPerSm, int KeyBlock, bool DynamicArena>() {
        const dim3 grid(Geometry::KVHeads, splits, invocation.batch_size);
        // NINFER_KV_INT4=1:K/V 页 int4 打包,装载时展开(默认关;读侧已接齐)
        static const std::int32_t kv_int4_env = [] {
            const char* e = std::getenv("NINFER_KV_INT4");
            return e != nullptr && std::atoi(e) != 0 ? 1 : 0;
        }();
        constexpr std::size_t kDynamicBytes =
            DynamicArena ? static_cast<std::size_t>(4 * KeyBlock * kCausalHeadDim) : 0u;
        if constexpr (DynamicArena) {
            static const cudaError_t attr = cudaFuncSetAttribute(
                causal_attention_small_t_i8_tiled_kernel<Geometry, TokenTile, WarpsPerCta,
                                                         MinBlocksPerSm, KeyBlock, DynamicArena,
                                                         MultiBatch, Masked, CacheInput>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(kDynamicBytes));
            CUDA_CHECK(attr);
        }
        causal_attention_small_t_i8_tiled_kernel<Geometry, TokenTile, WarpsPerCta, MinBlocksPerSm,
                                                 KeyBlock, DynamicArena, MultiBatch, Masked,
                                                 CacheInput>
            <<<grid, WarpsPerCta * 32, kDynamicBytes, stream>>>(
                static_cast<const __nv_bfloat16*>(q.data), input,
                static_cast<const std::int32_t*>(pos.data), static_cast<std::int8_t*>(cache_k.data),
                static_cast<std::int8_t*>(cache_v.data), static_cast<__half*>(cache_k_scale.data),
                static_cast<__half*>(cache_v_scale.data),
                static_cast<const std::int32_t*>(cache.block_tables.data),
                invocation.valid_columns == nullptr
                    ? nullptr
                    : static_cast<const std::int32_t*>(invocation.valid_columns->data),
                invocation.table_rows == nullptr
                    ? nullptr
                    : static_cast<const std::int32_t*>(invocation.table_rows->data),
                cache.block_tables.ne[0], invocation.full_width, invocation.column_begin,
                logical_capacity, scale, static_cast<__nv_bfloat16*>(partial_acc.data),
                static_cast<float*>(partial_m.data), static_cast<float*>(partial_l.data),
                kv_int4_env);
    };
    if constexpr (TokenTile == 6) {
        // Small grids need more warps per CTA. From 2K to 8K, Bc=64 halves key
        // loop iterations; dynamic smem avoids penalizing the long-context path.
        if (implementation_window > 128 && implementation_window <= 160) {
            launch.template operator()<24, 1, 32, false>();
        } else if (implementation_window <= 2054) {
            launch.template operator()<12, 1, 32, false>();
        } else if (implementation_window <= 8198) {
            // 2026-10-07: KeyBlock 64→32。64KB 动态 arena(4*64*256)加静态段超过
            // gfx1100 的 64KB 上限,cudaFuncSetAttribute 报 hipErrorInvalidValue
            // 直接 abort(int8 d5 崩因);32 是 ≤2054 分支的已验证值。
            launch.template operator()<12, 1, 32, true>();
        } else {
            launch.template operator()<6, 2, 32, false>();
        }
    } else if constexpr (TokenTile == 5) {
        if constexpr (Geometry::GroupSize == 6) {
            // Two Q row tiles for the 27B group of six.
            if (implementation_window > 128 && implementation_window <= 512) {
                launch.template operator()<32, 1, 32, false>();
            } else if (implementation_window <= 1029) {
                launch.template operator()<16, 1, 32, false>();
            } else {
                launch.template operator()<8, 2, 32, false>();
            }
        } else {
            // Three Q row tiles for the 35B group of eight. The 24/12-warp
            // routes retain eight/four consumer warps per tile; the 6-warp
            // route is reserved for long windows where CTA residency wins.
            if (implementation_window > 128 && implementation_window <= 512) {
                launch.template operator()<24, 1, 32, false>();
            } else if (implementation_window <= 1029) {
                launch.template operator()<24, 1, 32, false>();
            } else if (implementation_window <= 4096) {
                launch.template operator()<12, 1, 32, false>();
            } else {
                launch.template operator()<6, 2, 32, false>();
            }
        }
    } else if constexpr (TokenTile == 4) {
        if (implementation_window <= 1029) {
            launch.template operator()<16, 1, 32, false>();
        } else {
            launch.template operator()<8, 2, 32, false>();
        }
    } else if (arena_cfg) {
        // 诊断（NINFER_ATTN_CFG=arena）：同总 LDS 但静态/动态配比不同——静态 r_s 从 32KB 降到 16B，
        // 32KB 改走 dynamic。用来判断那 ~560µs/次的启动固定成本是否与**静态** smem 分配有关。
        launch.template operator()<8, 2, 32, true>();
    } else {
        launch.template operator()<8, 2, 32, false>();
    }
    CUDA_CHECK(cudaGetLastError());
}

PagedKVBatchLayerView single_row_batch_view(const PagedKVLayerView& cache) {
    return {
        .k_pages       = cache.k_pages,
        .v_pages       = cache.v_pages,
        .k_scale_pages = cache.k_scale_pages,
        .v_scale_pages = cache.v_scale_pages,
        .block_tables  = cache.block_table.view({cache.block_table.ne[0], 1}),
        .head_dim      = cache.head_dim,
        .num_kv_heads  = cache.num_kv_heads,
        .dtype         = cache.dtype,
        .quant_group   = cache.quant_group,
    };
}

} // namespace

bool causal_attention_uses_small_t(std::int32_t tokens) { return tokens >= 1 && tokens <= 6; }

std::int32_t causal_attention_split_capacity(std::int32_t q_heads, std::int32_t tokens,
                                             DType cache_dtype,
                                             CausalAttentionExecutionEnvelope envelope) {
    if (tokens < 1 || tokens > 6 ||
        (cache_dtype != DType::BF16 && cache_dtype != DType::I8 &&
         cache_dtype != DType::FP8_E4M3FN) ||
        envelope.min_visible_keys == 0 || envelope.min_visible_keys > envelope.max_visible_keys) {
        throw std::invalid_argument("causal_softmax_attention split capacity: invalid profile");
    }
    if (q_heads == CausalD256H24Kv4::QHeads) {
        return causal_small_t_launch_capacity<CausalD256H24Kv4>(envelope, tokens, cache_dtype);
    }
    if (q_heads == CausalD256H16Kv2::QHeads) {
        return causal_small_t_launch_capacity<CausalD256H16Kv2>(envelope, tokens, cache_dtype);
    }
    throw std::invalid_argument(
        "causal_softmax_attention split capacity: unsupported head geometry");
}

template <typename Geometry, typename CacheInput>
void causal_attention_small_t_launch_for(const Tensor& q, CacheInput input, const Tensor& pos,
                                         float scale, PagedKVBatchLayerView cache,
                                         const CausalSmallTInvocation& invocation,
                                         CausalAttentionExecutionEnvelope envelope,
                                         Tensor& partial_acc, Tensor& partial_m, Tensor& partial_l,
                                         Tensor& out, cudaStream_t stream) {
    // 核内分段计时的宿主侧开关（诊断用，默认关）。每 64 次调用把累积的 cycle 数打一行到 stderr。
    static const bool timing_on = std::getenv("NINFER_ATTN_TIMING") != nullptr;
    if (timing_on) {
        static const int enabled = 1;
        (void)cudaMemcpyToSymbol(g_attn_phase_enabled, &enabled, sizeof(enabled));
    }
    const auto logical_capacity      = static_cast<std::int32_t>(envelope.max_visible_keys);
    const auto implementation_window = static_cast<std::int32_t>(envelope.max_visible_keys);
    const auto splits =
        causal_small_t_launch_capacity<Geometry>(envelope, invocation.width, cache.dtype);

    // BF16 keeps its row-tile warp count; INT8 selects its producer/consumer
    // geometry inside launch_tc_partial_i8.
#define NINFER_CAUSAL_SMALL_T_DISPATCH(TOKENS, WARPS)                                              \
    do {                                                                                           \
        const auto launch_profile = [&]<bool MultiBatch, bool Masked>() {                          \
            if (cache.dtype == DType::I8) {                                                        \
                launch_tc_partial_i8<Geometry, (TOKENS), MultiBatch, Masked>(                      \
                    q, input, pos, scale, cache, invocation, logical_capacity,                     \
                    implementation_window, splits, partial_acc, partial_m, partial_l, stream);     \
            } else {                                                                               \
                launch_tc_partial_bf16<Geometry, (TOKENS), (WARPS), MultiBatch, Masked>(           \
                    q, input, pos, scale, cache, invocation, logical_capacity, splits,             \
                    partial_acc, partial_m, partial_l, stream);                                    \
            }                                                                                      \
        };                                                                                         \
        const bool masked = invocation.valid_columns != nullptr;                                   \
        if (invocation.batch_size == 1) {                                                          \
            if (masked) {                                                                          \
                launch_profile.template operator()<false, true>();                                 \
            } else {                                                                               \
                launch_profile.template operator()<false, false>();                                \
            }                                                                                      \
        } else if (masked) {                                                                       \
            launch_profile.template operator()<true, true>();                                      \
        } else {                                                                                   \
            launch_profile.template operator()<true, false>();                                     \
        }                                                                                          \
    } while (0)

    // 诊断开关（默认关）：把 partial 与 reduce 分开跳过，用来二分这 ~0.58ms/层的固定开销
    // 落在哪个核上。整 op 的跳过已在 launcher 侧用 NINFER_ATTN_SKIP 验过（跳过它 decode
    // 从 25.66 涨到 33.61 t/s，即该 op 真实占 9.2ms/步），这两个开关是往下一层切。
    const bool skip_partial = std::getenv("NINFER_ATTN_SKIP_PARTIAL") != nullptr;
    const bool skip_reduce  = std::getenv("NINFER_ATTN_SKIP_REDUCE") != nullptr;
    // 诊断：NINFER_ATTN_REPEAT=n 把 partial 核连发 n 次。用来判别那 ~560µs/次是"每次启动的固定成本"
    // （成本随 n 线性涨）还是"每次调用的等待/依赖"（成本与 n 无关）。默认 1，不改语义。
    static const int repeat = [] {
        const char* value = std::getenv("NINFER_ATTN_REPEAT");
        const int parsed  = value == nullptr ? 1 : std::atoi(value);
        return parsed < 1 ? 1 : (parsed > 8 ? 8 : parsed);
    }();
    for (int rep = 0; rep < repeat && !skip_partial; ++rep) {
        switch (invocation.width) {
        case 1:
            NINFER_CAUSAL_SMALL_T_DISPATCH(1, 2);
            break;
        case 2:
            NINFER_CAUSAL_SMALL_T_DISPATCH(2, 4);
            break;
        case 3:
            NINFER_CAUSAL_SMALL_T_DISPATCH(3, 4);
            break;
        case 4:
            NINFER_CAUSAL_SMALL_T_DISPATCH(4, 4);
            break;
        case 5:
            NINFER_CAUSAL_SMALL_T_DISPATCH(5, 4);
            break;
        case 6:
            NINFER_CAUSAL_SMALL_T_DISPATCH(6, 4);
            break;
        default:
            throw std::invalid_argument("causal_attention_small_t_launch: unsupported T");
        }
    }
#undef NINFER_CAUSAL_SMALL_T_DISPATCH

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
    const bool masked         = invocation.valid_columns != nullptr;
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
    if (!skip_reduce) {
        if (cache.dtype == DType::I8) {
            launch_for_dtype.template operator()<true>();
        } else {
            launch_for_dtype.template operator()<false>();
        }
    }
    if (timing_on) {
        static int calls = 0;
        if (++calls % 64 == 0) {
            unsigned long long host[3][64] = {};
            if (cudaMemcpyFromSymbol(host, g_attn_phase_cycles, sizeof(host)) == cudaSuccess) {
                double sum[3] = {0.0, 0.0, 0.0};
                int used = 0;
                for (int b = 0; b < 64; ++b) {
                    if (host[0][b] == 0 && host[1][b] == 0 && host[2][b] == 0) { continue; }
                    ++used;
                    for (int p = 0; p < 3; ++p) { sum[p] += static_cast<double>(host[p][b]); }
                }
                if (used > 0) {
                    int clock_khz = 0;
                    (void)cudaDeviceGetAttribute(&clock_khz, cudaDevAttrClockRate, 0);
                    const double to_us = clock_khz > 0 ? 1000.0 / static_cast<double>(clock_khz) : 0.0;
                    std::fprintf(stderr,
                                 "attn phase us/block (n=%d, %.0f MHz): prologue %.2f | key_loop %.2f | "
                                 "epilogue %.2f\n",
                                 used, clock_khz / 1000.0, sum[0] / used * to_us,
                                 sum[1] / used * to_us, sum[2] / used * to_us);
                    std::fflush(stderr);
                }
            }
            const unsigned long long zero[3][64] = {};
            (void)cudaMemcpyToSymbol(g_attn_phase_cycles, zero, sizeof(zero));
        }
    }
    CUDA_CHECK(cudaGetLastError());
}

void causal_attention_small_t_launch(
    const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& pos,
    const Tensor& valid_columns, const Tensor& table_rows, float scale, PagedKVBatchLayerView cache,
    CausalAttentionExecutionEnvelope envelope, std::int32_t column_begin, std::int32_t width,
    Tensor& partial_acc, Tensor& partial_m, Tensor& partial_l, Tensor& out, cudaStream_t stream) {
    if (cache.dtype == DType::FP8_E4M3FN) {
        causal_attention_small_t_fp8_launch(q, k, v, pos, valid_columns, table_rows, scale, cache,
                                            envelope, column_begin, width, partial_acc, partial_m,
                                            partial_l, out, stream);
        return;
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
    if (q.ne[1] == CausalD256H24Kv4::QHeads) {
        causal_attention_small_t_launch_for<CausalD256H24Kv4>(q, input, pos, scale, cache,
                                                              invocation, envelope, partial_acc,
                                                              partial_m, partial_l, out, stream);
        return;
    }
    causal_attention_small_t_launch_for<CausalD256H16Kv2>(q, input, pos, scale, cache, invocation,
                                                          envelope, partial_acc, partial_m,
                                                          partial_l, out, stream);
}

void causal_attention_cached_small_t_launch(const Tensor& q, const Tensor& pos, float scale,
                                            const PagedKVLayerView& cache,
                                            CausalAttentionExecutionEnvelope envelope,
                                            Tensor& partial_acc, Tensor& partial_m,
                                            Tensor& partial_l, Tensor& out, cudaStream_t stream) {
    if (cache.dtype == DType::FP8_E4M3FN) {
        causal_attention_cached_small_t_fp8_launch(q, pos, scale, cache, envelope, partial_acc,
                                                   partial_m, partial_l, out, stream);
        return;
    }
    const CausalCachedInput input{};
    const CausalSmallTInvocation invocation{
        .valid_columns = nullptr,
        .table_rows    = nullptr,
        .full_width    = q.ne[2],
        .column_begin  = 0,
        .width         = q.ne[2],
        .batch_size    = 1,
    };
    const PagedKVBatchLayerView batch_cache = single_row_batch_view(cache);
    if (q.ne[1] == CausalD256H24Kv4::QHeads) {
        causal_attention_small_t_launch_for<CausalD256H24Kv4>(q, input, pos, scale, batch_cache,
                                                              invocation, envelope, partial_acc,
                                                              partial_m, partial_l, out, stream);
        return;
    }
    causal_attention_small_t_launch_for<CausalD256H16Kv2>(q, input, pos, scale, batch_cache,
                                                          invocation, envelope, partial_acc,
                                                          partial_m, partial_l, out, stream);
}

} // namespace ninfer::ops::detail

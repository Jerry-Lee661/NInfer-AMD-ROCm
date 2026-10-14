// ninfer::ops - causal cached Softmax Attention validation and finite route dispatch.
#include "ninfer/ops/softmax_attention.h"

#include "core/layout.h"
#include "ops/kv_cache/d256_profile.h"
#include "ops/softmax_attention/dense/causal_cache/launch.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

constexpr std::int32_t kHeadDim                      = 256;
constexpr float kExpectedScale                       = 0.0625f;
constexpr std::int32_t kSmallTChunkTokens            = 6;
constexpr std::int32_t kMaximumVerifyTokens          = 16;
constexpr std::int32_t kMaximumBatchSize             = 8;
constexpr std::uint32_t kTwoChunkPromptVisibleKeys   = 512;
constexpr std::uint32_t kThreeChunkPromptVisibleKeys = 1024;

void require_causal_geometry(AttentionHeadGeometry geometry, const char* op) {
    if (!valid_attention_head_geometry(geometry) || geometry.head_dim != kHeadDim ||
        !((geometry.query_heads == 24 && geometry.kv_heads == 4) ||
          (geometry.query_heads == 16 && geometry.kv_heads == 2))) {
        throw std::invalid_argument(std::string(op) + ": unsupported head geometry");
    }
}

void require_shape(const Tensor& tensor, std::int32_t n0, std::int32_t n1, std::int32_t n2,
                   std::int32_t n3, const char* op, const char* name) {
    if (tensor.ne[0] != n0 || tensor.ne[1] != n1 || tensor.ne[2] != n2 || tensor.ne[3] != n3) {
        throw std::invalid_argument(std::string(op) + ": invalid shape for " + name);
    }
}

void require_contiguous_nonnull(const Tensor& tensor, const char* op, const char* name) {
    if (!tensor.is_contiguous()) {
        throw std::invalid_argument(std::string(op) + ": " + name + " must be contiguous");
    }
    if (tensor.data == nullptr) {
        throw std::invalid_argument(std::string(op) + ": " + name + " data must be non-null");
    }
}

std::uint32_t validate_cache(const PagedKVLayerView& cache, std::int32_t kv_heads, const char* op) {
    D256KVCacheProfile profile{};
    try {
        profile = d256_kv_cache_profile(cache.dtype);
    } catch (const std::invalid_argument&) {
        throw std::invalid_argument(std::string(op) + ": invalid KV cache geometry or dtype");
    }
    if (cache.num_kv_heads != kv_heads || cache.head_dim != kHeadDim ||
        cache.quant_group != profile.quant_group) {
        throw std::invalid_argument(std::string(op) + ": invalid KV cache geometry or dtype");
    }

    const std::int32_t physical_pages = cache.k_pages.ne[3];
    const std::int32_t logical_pages  = cache.block_table.ne[0];
    const std::int64_t capacity       = static_cast<std::int64_t>(logical_pages) * kPagedKVPageSize;
    if (physical_pages <= 0 || logical_pages <= 0 ||
        capacity > std::numeric_limits<std::int32_t>::max()) {
        throw std::invalid_argument(std::string(op) + ": invalid KV cache capacity");
    }

    if (cache.k_pages.dtype != profile.code_dtype || cache.v_pages.dtype != profile.code_dtype) {
        throw std::invalid_argument(std::string(op) + ": invalid KV cache code dtype");
    }
    require_shape(cache.k_pages, kHeadDim, kPagedKVPageSize, kv_heads, physical_pages, op,
                  "cache k pages");
    require_shape(cache.v_pages, kHeadDim, kPagedKVPageSize, kv_heads, physical_pages, op,
                  "cache v pages");
    require_contiguous_nonnull(cache.k_pages, op, "cache k pages");
    require_contiguous_nonnull(cache.v_pages, op, "cache v pages");
    if (cache.block_table.dtype != DType::I32) {
        throw std::invalid_argument(std::string(op) + ": block table must be I32");
    }
    require_shape(cache.block_table, logical_pages, 1, 1, 1, op, "block table");
    require_contiguous_nonnull(cache.block_table, op, "block table");

    if (cache.dtype == DType::BF16) {
        if (cache.k_scale_pages.data != nullptr || cache.v_scale_pages.data != nullptr) {
            throw std::invalid_argument(std::string(op) + ": BF16 KV cache must not have scales");
        }
        return static_cast<std::uint32_t>(capacity);
    }

    if (cache.k_scale_pages.dtype != DType::FP16 || cache.v_scale_pages.dtype != DType::FP16) {
        throw std::invalid_argument(std::string(op) + ": invalid KV cache scale dtype");
    }
    require_shape(cache.k_scale_pages, profile.scale_leading_extent, kPagedKVPageSize, kv_heads,
                  physical_pages, op, "cache k scale pages");
    require_shape(cache.v_scale_pages, profile.scale_leading_extent, kPagedKVPageSize, kv_heads,
                  physical_pages, op, "cache v scale pages");
    require_contiguous_nonnull(cache.k_scale_pages, op, "cache k scale pages");
    require_contiguous_nonnull(cache.v_scale_pages, op, "cache v scale pages");
    return static_cast<std::uint32_t>(capacity);
}

std::uint32_t validate_batch_cache(const PagedKVBatchLayerView& cache, std::int32_t kv_heads,
                                   const char* op) {
    D256KVCacheProfile profile{};
    try {
        profile = d256_kv_cache_profile(cache.dtype);
    } catch (const std::invalid_argument&) {
        throw std::invalid_argument(std::string(op) + ": invalid KV cache geometry or dtype");
    }
    if (cache.num_kv_heads != kv_heads || cache.head_dim != kHeadDim ||
        cache.quant_group != profile.quant_group) {
        throw std::invalid_argument(std::string(op) + ": invalid KV cache geometry or dtype");
    }

    const std::int32_t physical_pages = cache.k_pages.ne[3];
    const std::int32_t logical_pages  = cache.block_tables.ne[0];
    const std::int32_t table_rows     = cache.block_tables.ne[1];
    const std::int64_t capacity       = static_cast<std::int64_t>(logical_pages) * kPagedKVPageSize;
    if (physical_pages <= 0 || logical_pages <= 0 || table_rows <= 0 ||
        capacity > std::numeric_limits<std::int32_t>::max()) {
        throw std::invalid_argument(std::string(op) + ": invalid KV cache capacity");
    }

    if (cache.k_pages.dtype != profile.code_dtype || cache.v_pages.dtype != profile.code_dtype) {
        throw std::invalid_argument(std::string(op) + ": invalid KV cache code dtype");
    }
    require_shape(cache.k_pages, kHeadDim, kPagedKVPageSize, kv_heads, physical_pages, op,
                  "cache k pages");
    require_shape(cache.v_pages, kHeadDim, kPagedKVPageSize, kv_heads, physical_pages, op,
                  "cache v pages");
    require_contiguous_nonnull(cache.k_pages, op, "cache k pages");
    require_contiguous_nonnull(cache.v_pages, op, "cache v pages");
    if (cache.block_tables.dtype != DType::I32) {
        throw std::invalid_argument(std::string(op) + ": block tables must be I32");
    }
    require_shape(cache.block_tables, logical_pages, table_rows, 1, 1, op, "block tables");
    require_contiguous_nonnull(cache.block_tables, op, "block tables");

    if (cache.dtype == DType::BF16) {
        if (cache.k_scale_pages.data != nullptr || cache.v_scale_pages.data != nullptr) {
            throw std::invalid_argument(std::string(op) + ": BF16 KV cache must not have scales");
        }
        return static_cast<std::uint32_t>(capacity);
    }

    if (cache.k_scale_pages.dtype != DType::FP16 || cache.v_scale_pages.dtype != DType::FP16) {
        throw std::invalid_argument(std::string(op) + ": invalid KV cache scale dtype");
    }
    require_shape(cache.k_scale_pages, profile.scale_leading_extent, kPagedKVPageSize, kv_heads,
                  physical_pages, op, "cache k scale pages");
    require_shape(cache.v_scale_pages, profile.scale_leading_extent, kPagedKVPageSize, kv_heads,
                  physical_pages, op, "cache v scale pages");
    require_contiguous_nonnull(cache.k_scale_pages, op, "cache k scale pages");
    require_contiguous_nonnull(cache.v_scale_pages, op, "cache v scale pages");
    return static_cast<std::uint32_t>(capacity);
}

void validate_envelope(CausalAttentionExecutionEnvelope envelope, const PagedKVLayerView& cache,
                       std::int32_t tokens, const char* op) {
    const std::uint32_t capacity = validate_cache(cache, cache.num_kv_heads, op);
    if (envelope.min_visible_keys == 0 || envelope.min_visible_keys > envelope.max_visible_keys ||
        envelope.max_visible_keys > kCausalAttentionMaximumVisibleKeys ||
        envelope.max_visible_keys > capacity) {
        throw std::invalid_argument(std::string(op) + ": invalid execution envelope");
    }
    if (envelope.max_visible_keys < static_cast<std::uint32_t>(tokens)) {
        throw std::invalid_argument(std::string(op) + ": execution envelope is shorter than T");
    }
}

void validate_attention_tensors(const Tensor& q, const Tensor& positions, const Tensor& out,
                                AttentionHeadGeometry geometry, const PagedKVLayerView& cache,
                                CausalAttentionExecutionEnvelope envelope, float scale,
                                const char* op) {
    require_causal_geometry(geometry, op);
    if (q.dtype != DType::BF16 || out.dtype != DType::BF16) {
        throw std::invalid_argument(std::string(op) + ": q/out must be BF16");
    }
    if (positions.dtype != DType::I32) {
        throw std::invalid_argument(std::string(op) + ": positions must be I32");
    }
    if (!std::isfinite(scale) || std::abs(scale - kExpectedScale) > 1.0e-6f) {
        throw std::invalid_argument(std::string(op) + ": scale must be 1/sqrt(256)");
    }
    const std::int32_t q_heads  = geometry.query_heads;
    const std::int32_t kv_heads = geometry.kv_heads;
    const std::int32_t tokens   = q.ne[2];
    if (tokens <= 0) { throw std::invalid_argument(std::string(op) + ": T must be positive"); }
    require_shape(q, kHeadDim, q_heads, tokens, 1, op, "q");
    require_shape(positions, tokens, 1, 1, 1, op, "positions");
    require_shape(out, kHeadDim, q_heads, tokens, 1, op, "out");
    require_contiguous_nonnull(q, op, "q");
    require_contiguous_nonnull(positions, op, "positions");
    require_contiguous_nonnull(out, op, "out");
    if (cache.num_kv_heads != kv_heads) {
        throw std::invalid_argument(std::string(op) + ": invalid KV cache head geometry");
    }
    validate_envelope(envelope, cache, tokens, op);
}

void validate_batched_attention_tensors(const Tensor& q, const Tensor& positions,
                                        const Tensor& valid_columns, const Tensor& kv_table_rows,
                                        const Tensor& out, const PagedKVBatchLayerView& cache,
                                        AttentionHeadGeometry geometry,
                                        CausalAttentionExecutionEnvelope envelope, float scale,
                                        const char* op) {
    require_causal_geometry(geometry, op);
    if (q.dtype != DType::BF16 || out.dtype != DType::BF16) {
        throw std::invalid_argument(std::string(op) + ": q/out must be BF16");
    }
    const bool masked = valid_columns.data != nullptr;
    if (positions.dtype != DType::I32 || kv_table_rows.dtype != DType::I32 ||
        (masked && valid_columns.dtype != DType::I32)) {
        throw std::invalid_argument(std::string(op) + ": batch metadata must be I32");
    }
    if (!std::isfinite(scale) || std::abs(scale - kExpectedScale) > 1.0e-6f) {
        throw std::invalid_argument(std::string(op) + ": scale must be 1/sqrt(256)");
    }
    const std::int32_t q_heads  = geometry.query_heads;
    const std::int32_t kv_heads = geometry.kv_heads;
    const std::int32_t width    = q.ne[2];
    const std::int32_t batch    = q.ne[3];
    if (width <= 0 || batch <= 0 || batch > kMaximumBatchSize ||
        (batch > 1 && width > kMaximumVerifyTokens)) {
        throw std::invalid_argument(std::string(op) + ": unsupported B/W domain");
    }
    require_shape(q, kHeadDim, q_heads, width, batch, op, "q");
    require_shape(positions, width, batch, 1, 1, op, "positions");
    if (masked) { require_shape(valid_columns, batch, 1, 1, 1, op, "valid columns"); }
    require_shape(kv_table_rows, batch, 1, 1, 1, op, "KV table rows");
    require_shape(out, kHeadDim, q_heads, width, batch, op, "out");
    require_contiguous_nonnull(q, op, "q");
    require_contiguous_nonnull(positions, op, "positions");
    if (masked) { require_contiguous_nonnull(valid_columns, op, "valid columns"); }
    require_contiguous_nonnull(kv_table_rows, op, "KV table rows");
    require_contiguous_nonnull(out, op, "out");
    if (cache.num_kv_heads != kv_heads) {
        throw std::invalid_argument(std::string(op) + ": invalid KV cache head geometry");
    }
    const std::uint32_t capacity = validate_batch_cache(cache, kv_heads, op);
    if (cache.block_tables.ne[1] < batch || envelope.min_visible_keys == 0 ||
        envelope.min_visible_keys > envelope.max_visible_keys ||
        envelope.max_visible_keys > kCausalAttentionMaximumVisibleKeys ||
        envelope.max_visible_keys > capacity ||
        envelope.max_visible_keys < static_cast<std::uint32_t>(width)) {
        throw std::invalid_argument(std::string(op) + ": invalid execution envelope or table");
    }
}

struct SmallTWorkspace {
    Tensor acc;
    Tensor m;
    Tensor l;
};

template <class Allocator>
SmallTWorkspace allocate_small_t_workspace(Allocator& workspace, std::int32_t q_heads,
                                           std::int32_t tokens, std::int32_t splits,
                                           std::int32_t batch_size, DType cache_dtype) {
    return {
        workspace.alloc(cache_dtype == DType::FP8_E4M3FN ? DType::FP32 : DType::BF16,
                        {kHeadDim, q_heads, tokens, splits * batch_size}),
        workspace.alloc(DType::FP32, {q_heads, tokens, splits * batch_size}),
        workspace.alloc(DType::FP32, {q_heads, tokens, splits * batch_size}),
    };
}

template <typename Launch>
void for_each_small_t_chunk(const Tensor& q, const Tensor& positions, WorkspaceArena& workspace,
                            DType cache_dtype, CausalAttentionExecutionEnvelope envelope,
                            Tensor& out, Launch&& launch) {
    for (std::int32_t begin = 0; begin < q.ne[2]; begin += kSmallTChunkTokens) {
        const std::int32_t count = std::min(kSmallTChunkTokens, q.ne[2] - begin);
        auto chunk_scope         = workspace.scope();
        const std::int32_t splits =
            detail::causal_attention_split_capacity(q.ne[1], count, cache_dtype, envelope);
        SmallTWorkspace partial =
            allocate_small_t_workspace(workspace, q.ne[1], count, splits, 1, cache_dtype);
        Tensor q_chunk        = q.slice(2, begin, count);
        Tensor position_chunk = positions.slice(0, begin, count);
        Tensor out_chunk      = out.slice(2, begin, count);
        launch(begin, count, q_chunk, position_chunk, partial, out_chunk);
    }
}

void launch_chunked_small_t(const Tensor& q, const Tensor& k, const Tensor& v,
                            const Tensor& positions, const Tensor& valid_columns,
                            const Tensor& table_rows, float scale, PagedKVBatchLayerView cache,
                            CausalAttentionExecutionEnvelope envelope, WorkspaceArena& workspace,
                            Tensor& out, cudaStream_t stream) {
    for (std::int32_t begin = 0; begin < q.ne[2]; begin += kSmallTChunkTokens) {
        const std::int32_t count = std::min(kSmallTChunkTokens, q.ne[2] - begin);
        auto chunk_scope         = workspace.scope();
        const std::int32_t splits =
            detail::causal_attention_split_capacity(q.ne[1], count, cache.dtype, envelope);
        SmallTWorkspace partial =
            allocate_small_t_workspace(workspace, q.ne[1], count, splits, q.ne[3], cache.dtype);
        detail::causal_attention_small_t_launch(q, k, v, positions, valid_columns, table_rows,
                                                scale, cache, envelope, begin, count, partial.acc,
                                                partial.m, partial.l, out, stream);
    }
}

void launch_cached_chunked_small_t(const Tensor& q, const Tensor& positions, float scale,
                                   const PagedKVLayerView& cache,
                                   CausalAttentionExecutionEnvelope envelope,
                                   WorkspaceArena& workspace, Tensor& out, cudaStream_t stream) {
    for_each_small_t_chunk(
        q, positions, workspace, cache.dtype, envelope, out,
        [&](std::int32_t, std::int32_t, const Tensor& q_chunk, const Tensor& position_chunk,
            SmallTWorkspace& partial, Tensor& out_chunk) {
            detail::causal_attention_cached_small_t_launch(q_chunk, position_chunk, scale, cache,
                                                           envelope, partial.acc, partial.m,
                                                           partial.l, out_chunk, stream);
        });
}

} // namespace

namespace detail {

CausalAttentionRoute causal_attention_resolve_route(std::int32_t q_heads, std::int32_t width,
                                                    std::int32_t batch_size,
                                                    CausalAttentionExecutionEnvelope envelope) {
    if (width >= 1 && width <= kSmallTChunkTokens) { return CausalAttentionRoute::SmallT; }
    if (batch_size > 1) { return CausalAttentionRoute::ChunkedSmallT; }
    const std::uint32_t prompt_visible_keys =
        width <= 2 * kSmallTChunkTokens ? kTwoChunkPromptVisibleKeys : kThreeChunkPromptVisibleKeys;
    if (q_heads == 16 && width <= kMaximumVerifyTokens &&
        envelope.max_visible_keys > prompt_visible_keys) {
        return CausalAttentionRoute::ChunkedSmallT;
    }
    return CausalAttentionRoute::Prompt;
}

const char* causal_attention_route_name(CausalAttentionRoute route) {
    switch (route) {
    case CausalAttentionRoute::SmallT:
        return "small_t";
    case CausalAttentionRoute::ChunkedSmallT:
        return "chunked_small_t";
    case CausalAttentionRoute::Prompt:
        return "prompt";
    }
    return "unknown";
}

} // namespace detail

std::size_t causal_softmax_attention_workspace_capacity_bytes(
    AttentionHeadGeometry geometry, DType cache_dtype, CausalAttentionExecutionEnvelope envelope,
    std::int32_t batch_size, std::int32_t min_width, std::int32_t max_width) {
    require_causal_geometry(geometry, "causal_softmax_attention workspace");
    const std::int32_t q_heads = geometry.query_heads;
    bool supported_dtype       = true;
    try {
        (void)d256_kv_cache_profile(cache_dtype);
    } catch (const std::invalid_argument&) { supported_dtype = false; }
    if (!supported_dtype || batch_size <= 0 || batch_size > kMaximumBatchSize || min_width <= 0 ||
        max_width < min_width || (batch_size > 1 && max_width > kMaximumVerifyTokens) ||
        envelope.min_visible_keys == 0 || envelope.min_visible_keys > envelope.max_visible_keys ||
        envelope.max_visible_keys > kCausalAttentionMaximumVisibleKeys ||
        envelope.max_visible_keys < static_cast<std::uint32_t>(max_width)) {
        throw std::invalid_argument(
            "causal_softmax_attention workspace: invalid profile or interval");
    }

    const auto chunk_capacity = [&](std::int32_t width) {
        const std::int32_t splits =
            detail::causal_attention_split_capacity(q_heads, width, cache_dtype, envelope);
        WorkspaceLayoutBuilder layout;
        (void)allocate_small_t_workspace(layout, q_heads, width, splits, batch_size, cache_dtype);
        return layout.peak_bytes(1);
    };
    const auto exact_capacity = [&](std::int32_t width) {
        const detail::CausalAttentionRoute route =
            detail::causal_attention_resolve_route(q_heads, width, batch_size, envelope);
        if (route == detail::CausalAttentionRoute::Prompt) { return std::size_t{0}; }
        if (route == detail::CausalAttentionRoute::SmallT) { return chunk_capacity(width); }
        std::size_t maximum = 0;
        for (std::int32_t begin = 0; begin < width; begin += kSmallTChunkTokens) {
            maximum =
                std::max(maximum, chunk_capacity(std::min(kSmallTChunkTokens, width - begin)));
        }
        return maximum;
    };

    std::size_t maximum = 0;
    if (min_width <= kMaximumVerifyTokens) {
        const std::int32_t last = std::min(max_width, kMaximumVerifyTokens);
        for (std::int32_t width = min_width; width <= last; ++width) {
            maximum = std::max(maximum, exact_capacity(width));
        }
    }
    return maximum;
}

void causal_softmax_attention(const Tensor& q, const Tensor& k, const Tensor& v,
                              const Tensor& positions, const Tensor& valid_columns,
                              const Tensor& kv_table_rows, AttentionHeadGeometry geometry,
                              float scale, PagedKVBatchLayerView cache,
                              CausalAttentionExecutionEnvelope envelope, WorkspaceArena& workspace,
                              Tensor& out, cudaStream_t stream) {
    constexpr const char* op = "causal_softmax_attention";
    // 诊断开关：NINFER_ATTN_SKIP=1 时整个 op 不落任何 kernel，用来量它在整步里的**真实**占比。
    // 逐区 profiler 靠 stream 同步切段，存在把"等待"记到相邻区上的风险，所以需要一个不依赖
    // 同步的独立校验。代价：out 不写，后续 token 是垃圾——只看计时，不看输出。
    if (std::getenv("NINFER_ATTN_SKIP") != nullptr) { return; }
    validate_batched_attention_tensors(q, positions, valid_columns, kv_table_rows, out, cache,
                                       geometry, envelope, scale, op);
    if (k.dtype != DType::BF16 || v.dtype != DType::BF16) {
        throw std::invalid_argument("causal_softmax_attention: k/v must be BF16");
    }
    const std::int32_t width    = q.ne[2];
    const std::int32_t batch    = q.ne[3];
    const std::int32_t kv_heads = geometry.kv_heads;
    require_shape(k, kHeadDim, kv_heads, width, batch, op, "k");
    require_shape(v, kHeadDim, kv_heads, width, batch, op, "v");
    require_contiguous_nonnull(k, op, "k");
    require_contiguous_nonnull(v, op, "v");

    auto scope = workspace.scope();
    detail::CausalAttentionRoute route =
        detail::causal_attention_resolve_route(q.ne[1], width, batch, envelope);
    // 诊断(2026-10-09, NINFER_ATTN_ROUTE_TRACE=1): 每次调用的路由与键窗(megakernel M1c)。
    static const bool route_trace = std::getenv("NINFER_ATTN_ROUTE_TRACE") != nullptr;
    if (route_trace) {
        static int route_lines = 0;
        if (route_lines < 40) {
            ++route_lines;
            std::fprintf(stderr, "[attn-route] %s width=%d batch=%d keys<=%u\n",
                         detail::causal_attention_route_name(route), width, batch,
                         envelope.max_visible_keys);
            std::fflush(stderr);
        }
    }
    // 诊断对照（默认关）：NINFER_ATTN_FORCE_ROUTE=prompt 强制走 prompt 路由，用来判断
    // "T=1 下 ~0.6ms/次"是 small_t 这条路径本身的开销，还是这套分页/追加机制的固有代价。
    // prompt 核在小 T 下语义未必等价，只作性能对照，不要拿来出数。
    if (const char* forced = std::getenv("NINFER_ATTN_FORCE_ROUTE"); forced != nullptr) {
        if (std::strcmp(forced, "prompt") == 0) {
            route = detail::CausalAttentionRoute::Prompt;
        } else if (std::strcmp(forced, "small_t") == 0) {
            route = detail::CausalAttentionRoute::SmallT;
        }
    }
    if (route == detail::CausalAttentionRoute::ChunkedSmallT) {
        launch_chunked_small_t(q, k, v, positions, valid_columns, kv_table_rows, scale, cache,
                               envelope, workspace, out, stream);
        return;
    }
    if (route == detail::CausalAttentionRoute::SmallT) {
        const std::int32_t splits =
            detail::causal_attention_split_capacity(q.ne[1], width, cache.dtype, envelope);
        SmallTWorkspace partial =
            allocate_small_t_workspace(workspace, q.ne[1], width, splits, batch, cache.dtype);
        // decode 的极小 T 走瘦版核：单 workgroup LDS 从 ~37.8KB 压到 ~29.8KB，绕开
        // 「>32KB 的核每次启动约 0.54ms」（decode 每步 16 次 → 约 8.6ms/步）。
        // **窗口长度不能作为条件**：那 0.54ms 与窗口无关，而 decode 的窗口随序列增长很快越过
        // 32（一次 128 token 生成里大半步都在 32 以上）——卡窗口等于大半时间没走瘦版（首版实测
        // lean/off 完全同速就是这么来的）。条件只按 T 与 batch：T ≤ 4 是 decode（含 MTP 验证趟），
        // 5/6 是 prefill 分块，留给老核。V 改读全局对长窗口只是每层多 ~2MB/步的读，可忽略。
        // 多序列（服务端并发）扩展：核与归约都已支持 batch 维（MultiBatch 实例），
        // 之前只是路由把 batch 卡在 1——C8 实测 attention 每步 33.6ms（C1 的 25.5 倍，应 8 倍），
        // 因为 batch>1 回落到带私有段的老核。已默认开：C8 实测 80.0 → 109.1 t/s（+36%）、
        // 数值门 PASS（2026-09-23 validate-pending）；NINFER_ATTN_SIMT_MULTI=0 回退老核。
        static const int simt_multi_env = [] {
            const char* value = std::getenv("NINFER_ATTN_SIMT_MULTI");
            return value == nullptr ? 1 : std::atoi(value);
        }();
        const bool lean_shape = cache.dtype == DType::I8 && width >= 1 && width <= 4 &&
                                (batch == 1 || simt_multi_env != 0);
        static const int lean_env = [] {
            const char* value = std::getenv("NINFER_ATTN_LEAN");
            return value == nullptr ? -1 : std::atoi(value);
        }();
        // 零溢出的 SIMT 核现在是默认路径（+27.6%：25.97 → 33.13 t/s）；NINFER_ATTN_SIMT=0 可回退。
        // 数值上比老核更准（不把 Q 量化成 int8），实测三个 prompt 输出连贯、前 18 个 token 逐位相同，
        // 之后因贪心在 top-2 logits 接近处翻转而分岔——这是预期的良性差异，不是错。
        static const int simt_env = [] {
            const char* value = std::getenv("NINFER_ATTN_SIMT");
            return value == nullptr ? 1 : std::atoi(value);
        }();
        // 一次性诊断（NINFER_ATTN_TRACE=1）：把这条路的真实运行值打出来，确认瘦版到底走没走。
        if (std::getenv("NINFER_ATTN_TRACE") != nullptr) {
            static int traced = 0;
            if (traced < 4) {
                ++traced;
                std::fprintf(stderr,
                             "attn trace: width=%d batch=%d dtype=%d min/max_keys=%u/%u lean=%d "
                             "simt=%d\n",
                             width, batch, static_cast<int>(cache.dtype),
                             envelope.min_visible_keys, envelope.max_visible_keys,
                             (lean_shape && lean_env != 0) ? 1 : 0, simt_env);
                std::fflush(stderr);
            }
        }
        if (lean_shape && simt_env != 0) {
            detail::causal_attention_small_t_simt_launch(
                q, k, v, positions, valid_columns, kv_table_rows, scale, cache, envelope, 0, width,
                partial.acc, partial.m, partial.l, out, stream);
        } else if (lean_shape && lean_env != 0) {
            detail::causal_attention_small_t_lean_launch(
                q, k, v, positions, valid_columns, kv_table_rows, scale, cache, envelope, 0, width,
                partial.acc, partial.m, partial.l, out, stream);
        } else {
            detail::causal_attention_small_t_launch(q, k, v, positions, valid_columns,
                                                    kv_table_rows, scale, cache, envelope, 0, width,
                                                    partial.acc, partial.m, partial.l, out, stream);
        }
        return;
    }
    detail::causal_attention_prompt_launch(q, k, v, positions, valid_columns, kv_table_rows, scale,
                                           cache, out, stream);
}

void causal_softmax_attention_cached(const Tensor& q, const Tensor& positions,
                                     AttentionHeadGeometry geometry, float scale,
                                     const PagedKVLayerView& cache,
                                     CausalAttentionExecutionEnvelope envelope,
                                     WorkspaceArena& workspace, Tensor& out, cudaStream_t stream) {
    constexpr const char* op = "causal_softmax_attention_cached";
    validate_attention_tensors(q, positions, out, geometry, cache, envelope, scale, op);

    auto scope = workspace.scope();
    const detail::CausalAttentionRoute route =
        detail::causal_attention_resolve_route(q.ne[1], q.ne[2], 1, envelope);
    if (route == detail::CausalAttentionRoute::ChunkedSmallT) {
        launch_cached_chunked_small_t(q, positions, scale, cache, envelope, workspace, out, stream);
        return;
    }
    if (detail::causal_attention_uses_small_t(q.ne[2])) {
        const std::int32_t splits =
            detail::causal_attention_split_capacity(q.ne[1], q.ne[2], cache.dtype, envelope);
        SmallTWorkspace partial =
            allocate_small_t_workspace(workspace, q.ne[1], q.ne[2], splits, 1, cache.dtype);
        detail::causal_attention_cached_small_t_launch(
            q, positions, scale, cache, envelope, partial.acc, partial.m, partial.l, out, stream);
        return;
    }
    detail::causal_attention_prompt_attention_launch(q, positions, scale, cache, out, stream);
}

} // namespace ninfer::ops

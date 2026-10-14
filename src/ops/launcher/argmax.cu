// Implements: include/ninfer/ops/argmax.h
// Match: validated contiguous BF16 logits and I32 output.
// Algorithm assumptions: one tile uses a direct reduction; larger domains use
// zero-initialized atomic winners across route-selected row tiles.
#include "ops/launcher/argmax.h"

#include "ops/common/math.h"
#include "ops/common/token_slices.h"
#include "ops/kernel/argmax.cuh"
#include "core/device.h" // CUDA_CHECK

#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace ninfer::ops {
namespace {

// Debug aid for the mod16 logits fingerprint: NINFER_DUMP_LOGITS=1 dumps the first two
// argmax invocations' row-0 head plus a window around a known winner index.
void debug_dump_logits_row(const Tensor& logits, std::int32_t valid_rows) {
    static int dumped = 0;
    if (dumped >= 2 || std::getenv("NINFER_DUMP_LOGITS") == nullptr) { return; }
    ++dumped;
    const auto* row = static_cast<const __nv_bfloat16*>(logits.data);
    std::fprintf(stderr, "[logits-dump %d] ne={%d,%d} valid_rows=%d\n", dumped, logits.ne[0],
                 logits.ne[1], valid_rows);
    __nv_bfloat16 tmp[24];
    cudaMemcpy(tmp, row, sizeof(tmp), cudaMemcpyDeviceToHost);
    for (int i = 0; i < 24; ++i) {
        std::fprintf(stderr, "%.4g ", __bfloat162float(tmp[i]));
    }
    std::fprintf(stderr, "\n[logits-dump] around 67616: ");
    cudaMemcpy(tmp, row + 67608, sizeof(tmp), cudaMemcpyDeviceToHost);
    for (int i = 0; i < 24; ++i) {
        std::fprintf(stderr, "%.4g ", __bfloat162float(tmp[i]));
    }
    std::fprintf(stderr, "\n[logits-dump] around 131072: ");
    cudaMemcpy(tmp, row + 131064, sizeof(tmp), cudaMemcpyDeviceToHost);
    for (int i = 0; i < 24; ++i) {
        std::fprintf(stderr, "%.4g ", __bfloat162float(tmp[i]));
    }
    std::fprintf(stderr, "\n");
}

} // namespace

namespace detail {
namespace {

constexpr std::int32_t kFullPhysicalRows      = 248320;
constexpr std::int32_t kFullValidRows         = 248077;
constexpr std::int32_t kShortlistRows         = 131072;
constexpr std::int32_t kSmallAggregateColumns = 8;
constexpr int kFullAggregateBlock             = 128;
constexpr int kShortlistAggregateBlock        = 256;

int tiled_block_for(std::int32_t physical_rows, std::int32_t valid_rows, std::int32_t t_count) {
    if (t_count <= kSmallAggregateColumns) { return kArgmaxBlock; }
    if (physical_rows == kFullPhysicalRows && valid_rows == kFullValidRows) {
        return kFullAggregateBlock;
    }
    if (physical_rows == kShortlistRows && valid_rows == kShortlistRows) {
        return kShortlistAggregateBlock;
    }
    return kArgmaxBlock;
}

void argmax_tiled_atomic_launch(const Tensor& logits, Tensor& out, std::int32_t valid_rows,
                                int block, cudaStream_t stream);

} // namespace

void argmax_launch(const Tensor& logits, Tensor& out, std::int32_t valid_rows,
                   cudaStream_t stream) {
    debug_dump_logits_row(logits, valid_rows);
    const std::int32_t physical_rows = logits.ne[0];
    const std::int32_t t_count       = logits.ne[1];
    if (t_count == 0) { return; }

    constexpr int kTileElems = kArgmaxBlock * kArgmaxItemsPerThread;
    const int tiled_blocks   = div_up(valid_rows, kTileElems);
    if (tiled_blocks < 2) {
        argmax_kernel<<<static_cast<unsigned int>(t_count), kArgmaxBlock, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(logits.data), static_cast<std::int32_t*>(out.data),
            valid_rows, physical_rows);
        CUDA_CHECK(cudaGetLastError());
        return;
    }

    argmax_tiled_atomic_launch(logits, out, valid_rows,
                               tiled_block_for(physical_rows, valid_rows, t_count), stream);
}

namespace {

void argmax_tiled_atomic_launch(const Tensor& logits, Tensor& out, std::int32_t valid_rows,
                                int block, cudaStream_t stream) {
    const std::int32_t physical_rows = logits.ne[0];
    const std::int32_t t_count       = logits.ne[1];
    const int tiled_blocks           = div_up(valid_rows, block * kArgmaxItemsPerThread);
    for_each_token_slice(t_count, 1, [&](int token_offset, int token_count) {
        const Tensor logits_slice = logits.slice(1, token_offset, token_count);
        Tensor out_slice          = out.slice(0, token_offset, token_count);
        CUDA_CHECK(cudaMemsetAsync(out_slice.data, 0,
                                   static_cast<std::size_t>(token_count) * sizeof(std::int32_t),
                                   stream));
        const dim3 grid(static_cast<unsigned int>(tiled_blocks),
                        static_cast<unsigned int>(token_count));
        argmax_tiled_atomic_kernel<<<grid, block, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(logits_slice.data),
            static_cast<std::int32_t*>(out_slice.data), valid_rows, physical_rows);
        CUDA_CHECK(cudaGetLastError());
    });
}

} // namespace
} // namespace ninfer::ops::detail
} // namespace ninfer::ops

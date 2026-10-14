// Implements: include/ninfer/ops/sampling.h
// Match: validated contiguous BF16/I32 tensors and a shared-layout workspace.
// Algorithm assumptions: launcher and kernels use sampler_multiblock_ok() from
// the same layout authority, so exactly one finite route owns each shape.
#include "ops/launcher/sampling.h"

#include "ops/common/math.h"
#include "ops/kernel/sampling.cuh"
#include "core/device.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace ninfer::ops::detail {

namespace {

// Debug aid for the mod16 logits fingerprint: NINFER_DUMP_LOGITS=1 dumps the first two
// sample_batch_launch invocations. Call 1 adds a full host scan (top-8 values/indices),
// which decides whether the lattice lives in the logits buffer or in the sampler.
void debug_dump_sample_row(const Tensor& logits, std::int32_t token_domain) {
    static int dumped = 0;
    if (dumped >= 2 || std::getenv("NINFER_DUMP_LOGITS") == nullptr) { return; }
    ++dumped;
    const auto* row = static_cast<const __nv_bfloat16*>(logits.data);
    const std::int32_t rows = logits.ne[0];
    std::fprintf(stderr, "[sample-dump %d] ne={%d,%d} token_domain=%d\n", dumped, rows,
                 logits.ne[1], token_domain);
    const int head = rows < 24 ? rows : 24;
    std::vector<__nv_bfloat16> tmp(static_cast<std::size_t>(head));
    cudaMemcpy(tmp.data(), row, tmp.size() * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);
    for (int i = 0; i < head; ++i) {
        std::fprintf(stderr, "%.4g ", __bfloat162float(tmp[i]));
    }
    std::fprintf(stderr, "\n");
    if (dumped != 1) { return; }
    const int scan = rows < token_domain ? rows : token_domain;
    std::vector<__nv_bfloat16> full(static_cast<std::size_t>(scan));
    if (cudaMemcpy(full.data(), row, full.size() * sizeof(__nv_bfloat16),
                   cudaMemcpyDeviceToHost) != cudaSuccess) {
        std::fprintf(stderr, "[sample-dump] full copy failed\n");
        return;
    }
    struct Top {
        float value;
        int index;
    };
    Top top[8];
    for (auto& t : top) {
        t = {-1e30f, -1};
    }
    long nonfinite = 0;
    for (int i = 0; i < scan; ++i) {
        const float v = __bfloat162float(full[static_cast<std::size_t>(i)]);
        if (!std::isfinite(v)) {
            ++nonfinite;
            continue;
        }
        for (int k = 0; k < 8; ++k) {
            if (v > top[k].value || (v == top[k].value && i < top[k].index)) {
                for (int m = 7; m > k; --m) { top[m] = top[m - 1]; }
                top[k] = {v, i};
                break;
            }
        }
    }
    std::fprintf(stderr, "[sample-dump] scanned=%d nonfinite=%ld top8:", scan, nonfinite);
    for (const auto& t : top) {
        std::fprintf(stderr, " (%d:%.4g)", t.index, t.value);
    }
    std::fprintf(stderr, "\n");
}

} // namespace

__global__ void increment_token_counts_kernel(const std::int32_t* token_ids, std::int32_t count,
                                              std::int32_t* token_counts) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) { atomicAdd(&token_counts[token_ids[index]], 1); }
}

std::size_t sampling_workspace_exact_bytes(std::int32_t token_domain, std::int32_t columns) {
    return make_sampling_workspace_layout(token_domain, columns).bytes;
}

void sample_batch_launch(const Tensor& logits, Tensor& out, std::int32_t token_domain,
                         const SamplingConfig* configs, const Tensor& logical_positions,
                         std::int32_t purpose, DeviceSpan workspace, cudaStream_t stream) {
    debug_dump_sample_row(logits, token_domain);
    const std::int32_t physical_rows     = logits.ne[0];
    const std::int32_t batch             = logits.ne[1];
    const auto* positions                = static_cast<const std::int32_t*>(logical_positions.data);
    const SamplingWorkspaceLayout layout = make_sampling_workspace_layout(token_domain, batch);
    if (!layout.multiblock) {
        sample_row_kernel<<<static_cast<unsigned int>(batch), kSamplerBlock, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(logits.data), static_cast<std::int32_t*>(out.data),
            configs, positions, purpose, token_domain, physical_rows);
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    const std::int32_t partial_blocks = div_up(token_domain, kSamplerPartialTileItems);
    const std::int32_t groups         = sampler_group_count(partial_blocks);
    const SamplingWorkspace scratch   = layout.bind(workspace);
    const dim3 partial_grid(static_cast<unsigned int>(partial_blocks),
                            static_cast<unsigned int>(batch));
    sampling_partial_topk_kernel<<<partial_grid, kSamplerBlock, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(logits.data), configs, token_domain, physical_rows,
        scratch);
    CUDA_CHECK(cudaGetLastError());
    const dim3 group_grid(static_cast<unsigned int>(groups), static_cast<unsigned int>(batch));
    sampling_group_finalize_sample_kernel<<<group_grid, kSamplerGroupBlock, 0, stream>>>(
        static_cast<std::int32_t*>(out.data), configs, positions, purpose, token_domain,
        partial_blocks, groups, scratch);
    CUDA_CHECK(cudaGetLastError());
}

void increment_token_counts_launch(const Tensor& token_ids, Tensor& token_counts,
                                   cudaStream_t stream) {
    constexpr int kBlock = 256;
    const int count      = token_ids.ne[0];
    increment_token_counts_kernel<<<div_up(count, kBlock), kBlock, 0, stream>>>(
        static_cast<const std::int32_t*>(token_ids.data), count,
        static_cast<std::int32_t*>(token_counts.data));
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail

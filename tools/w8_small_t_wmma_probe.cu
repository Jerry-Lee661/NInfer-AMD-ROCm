// w8 small_t: FMA-emulation vs v_wmma, same structure, real geometry (k=5120), timed.
//
// Purpose: give the WMMA main-work item a hard first number for the w8 family (the
// lm_head is the single largest matrix) before any kernel surgery: ms per 16-row CTA
// on each path, plus a numeric cross-check of the WMMA path against the CPU reference.
//
// Adapter shapes follow the hardware-decoded layout from tools/hip_wmma_selftest.cu:
//   builtin(va, vb) computes (2nd arg) x (1st arg) = vb_matrix x va_matrix;
//   for M*N feed N through fill_a (k-major/n-contiguous), M through fill_b ([m][k]).
//   C: lane row = lane & 15, register j -> column 2j + (lane>>4).
//
// This probe stages per-warp operands once per k-step from global memory; the real
// kernel will amortize packing over its own LDS staging, so treat the WMMA number as
// an upper bound on cost (lower bound on speedup).
//
// Build (X99):
//   hipcc -O3 -std=c++20 -DNINFER_HIP=1 -I compat/cuda -I src -I include \
//     --offload-arch=gfx1100 tools/w8_small_t_wmma_probe.cu -o /tmp/w8wmma && /tmp/w8wmma

#include "ops/linear/w8/w8_config.h"
#include "ops/linear/w8/w8_rowsplit_output.cuh"
#include "ops/linear/w8/w8_small_t_mma.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

namespace {

using ninfer::ops::detail::W8ContiguousOutput;
using ninfer::ops::detail::W8LinearSmallTProductionSchedule;
using ninfer::ops::detail::W8VocabularyProjectionGeometry;
using ninfer::ops::detail::w8_small_t_bf16_pair_from_s8;

constexpr int kHidden = W8VocabularyProjectionGeometry::kInputRows;  // 5120
constexpr int kRows   = W8VocabularyProjectionGeometry::kOutputRows; // 248320, engine grid

using A8   = unsigned __attribute__((ext_vector_type(8)));
using B16S = short __attribute__((ext_vector_type(16)));
using C8F  = float __attribute__((ext_vector_type(8)));

// WMMA path: one warp computes a 16-row x 16-col tile; per k-step (16 wide) it packs
// B-16x16 (weights [m][k], dequantized) and A-16x16 (activations k-major).
__global__ __launch_bounds__(32) void w8_wmma_probe_kernel(
    const __nv_bfloat16* __restrict__ x, const std::uint8_t* __restrict__ codes,
    __nv_bfloat16* __restrict__ out, int rows, int k_steps) {
    const int cta  = static_cast<int>(blockIdx.x);
    const int lane = static_cast<int>(threadIdx.x);
    const int row0 = cta * 16;
    if (row0 >= rows) { return; }

    C8F acc{};
    unsigned d_va0 = 0, d_va1 = 0, d_vb0 = 0, d_vb1 = 0;
    for (int step = 0; step < k_steps; ++step) {
        const int k0 = step * 16;
        // B operand: weights [16 rows][16 k], bf16 codes from int8 via the production dequant.
        B16S vb;
#pragma unroll
        for (int j = 0; j < 16; ++j) {
            const int m = row0 + (lane & 15);
            const int k = k0 + j;
            const unsigned pair =
                w8_small_t_bf16_pair_from_s8(static_cast<unsigned>(
                    codes[static_cast<std::int64_t>(m) * kHidden + k]) & 0xFFu);
            vb[j] = static_cast<short>(pair & 0xFFFFu);
        }
        // A operand: activations k-major [k][n]; n = 16 columns.
        A8 va;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const int kk = k0 + 2 * j;
            const int n  = lane & 15;
            const unsigned short lo =
                __bfloat16_as_ushort(x[static_cast<std::int64_t>(n) * kHidden + kk]);
            const unsigned short hi =
                __bfloat16_as_ushort(x[static_cast<std::int64_t>(n) * kHidden + kk + 1]);
            va[j] = static_cast<unsigned>(lo) | (static_cast<unsigned>(hi) << 16);
        }
        if (step == 0) {
            d_va0 = va[0];
            d_va1 = va[1];
            d_vb0 = static_cast<unsigned>(static_cast<unsigned short>(vb[0]));
            d_vb1 = static_cast<unsigned>(static_cast<unsigned short>(vb[1]));
        }
        acc = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(va, vb, acc);
    }
    // C: lane row = lane & 15, register j -> column 2j + (lane>>4).
    const int row  = row0 + (lane & 15);
    const int cpar = lane >> 4;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const int n = 2 * j + cpar;
        if (n < 16) {
            out[static_cast<std::int64_t>(n) * rows + row] =
                __float2bfloat16_rn(acc[j]);
        }
    }
    // Debug: CTA 0 dumps raw per-lane operand/accumulator registers (float bit patterns).
    if (cta == 0) {
        unsigned* dbg_g = reinterpret_cast<unsigned*>(out + 16 * rows);  // past the tile area
        dbg_g[lane * 12 + 0] = d_va0;
        dbg_g[lane * 12 + 1] = d_va1;
        dbg_g[lane * 12 + 2] = d_vb0;
        dbg_g[lane * 12 + 3] = d_vb1;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            dbg_g[lane * 12 + 4 + j] = __float_as_uint(acc[j]);
        }
    }
}

std::uint32_t lcg(std::uint32_t& s) {
    s = s * 1664525u + 1013904223u;
    return s >> 16;
}

} // namespace

int main(int argc, char** argv) {
    const int k_steps = (argc > 1) ? std::atoi(argv[1]) : kHidden / 16;
    std::vector<__nv_bfloat16> h_x(static_cast<std::size_t>(16) * kHidden);
    std::uint32_t seed = 7u;
    for (auto& v : h_x) {
        v = __float2bfloat16_rn(static_cast<float>(static_cast<int>(lcg(seed) % 9) - 4) * 0.5f);
    }
    std::vector<std::uint8_t> h_codes(static_cast<std::size_t>(kRows) * kHidden);
    for (auto& c : h_codes) { c = static_cast<std::uint8_t>(lcg(seed) % 17); }

    __nv_bfloat16* d_x = nullptr;
    std::uint8_t* d_codes = nullptr;
    __nv_bfloat16* d_out = nullptr;
    constexpr int kDumpU32 = 32 * 12;  // per-lane raw registers, CTA 0 only
    constexpr std::size_t kOutElems = static_cast<std::size_t>(16) * kRows;
    constexpr std::size_t kAllocBytes = kOutElems * sizeof(__nv_bfloat16) + kDumpU32 * 4;
    if (cudaMalloc(&d_x, h_x.size() * sizeof(__nv_bfloat16)) != cudaSuccess ||
        cudaMalloc(&d_codes, h_codes.size()) != cudaSuccess ||
        cudaMalloc(&d_out, kAllocBytes) != cudaSuccess) {
        std::fprintf(stderr, "cudaMalloc failed\n");
        return 2;
    }
    cudaMemset(d_out, 0, kAllocBytes);
    cudaMemcpy(d_x, h_x.data(), h_x.size() * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
    cudaMemcpy(d_codes, h_codes.data(), h_codes.size(), cudaMemcpyHostToDevice);

    // Warm-up + timed WMMA run.
    w8_wmma_probe_kernel<<<kRows / 16, 32>>>(d_x, d_codes, d_out, kRows, k_steps);
    if (hipDeviceSynchronize() != cudaSuccess) { std::fprintf(stderr, "wmma err\n"); return 2; }
    hipEvent_t t0, t1;
    hipEventCreate(&t0);
    hipEventCreate(&t1);
    hipEventRecord(t0);
    w8_wmma_probe_kernel<<<kRows / 16, 32>>>(d_x, d_codes, d_out, kRows, k_steps);
    hipEventRecord(t1);
    hipEventSynchronize(t1);
    float wmma_ms = 0.0f;
    hipEventElapsedTime(&wmma_ms, t0, t1);

    // Debug dump: CTA 0's per-lane raw registers.
    std::vector<unsigned> dbg(kDumpU32);
    cudaMemcpy(dbg.data(), reinterpret_cast<const unsigned char*>(d_out) + kOutElems * 2,
               kDumpU32 * sizeof(unsigned), cudaMemcpyDeviceToHost);
    for (int lane = 0; lane < 40; lane += (lane < 4 ? 1 : (lane < 32 ? 12 : 1))) {
        const unsigned* d = &dbg[static_cast<std::size_t>(lane) * 12];
        std::printf("lane %2d: va=%08x %08x vb=%04x %04x acc=", lane, d[0], d[1], d[2] & 0xFFFF,
                    d[3] & 0xFFFF);
        for (int j = 0; j < 8; ++j) {
            std::printf("%08x ", d[4 + j]);
        }
        std::printf("\n");
    }

    // Numeric check: rows 0..7, cols 0..7 against the CPU reference (scales = 1).
    // The store layout is out[n * rows + m], so copy BOTH full columns we score.
    std::vector<__nv_bfloat16> h_out(static_cast<std::size_t>(16) * kRows);
    cudaMemcpy(h_out.data(), d_out, 8 * kRows * sizeof(__nv_bfloat16),
               cudaMemcpyDeviceToHost);
    long mismatches = 0;
    double worst = 0.0;
    const int k_used = k_steps * 16;
    for (int m = 0; m < 8; ++m) {
        for (int n = 0; n < 8; ++n) {
            double ref = 0.0;
            for (int k = 0; k < k_used; ++k) {
                ref += static_cast<double>(__bfloat162float(h_x[static_cast<std::size_t>(n) *
                                                                    kHidden + k])) *
                       static_cast<double>(h_codes[static_cast<std::size_t>(m) * kHidden + k]);
            }
            const double got =
                __bfloat162float(h_out[static_cast<std::size_t>(n) * kRows + m]);
            const double tol = 8.0 * std::abs(ref) * 0.01 + 1.0;
            if (std::abs(got - ref) > tol) {
                ++mismatches;
                if (mismatches <= 5) {
                    std::printf("m=%d n=%d: got %.1f ref %.1f\n", m, n, got, ref);
                }
            }
            worst = std::max(worst, std::abs(got - ref));
        }
    }

    // Emulated path: the production kernel on the same rows (scales = 1), timed.
    constexpr int kScaleRowBytes =
        W8VocabularyProjectionGeometry::kGroupsPerRow * static_cast<int>(sizeof(std::uint16_t));
    std::vector<std::uint8_t> h_scales(static_cast<std::size_t>(kRows) * kScaleRowBytes, 0x00);
    for (std::size_t i = 0; i + 1 < h_scales.size(); i += 2) { h_scales[i + 1] = 0x3C; }
    std::uint8_t* d_scales = nullptr;
    cudaMalloc(&d_scales, h_scales.size());
    cudaMemcpy(d_scales, h_scales.data(), h_scales.size(), cudaMemcpyHostToDevice);

    using Schedule = W8LinearSmallTProductionSchedule<W8VocabularyProjectionGeometry, 1>::Type;
    const W8ContiguousOutput output{d_out, kRows};
    auto run_emulated = [&]() {
        ninfer::ops::detail::w8_small_t_mma_kernel<W8VocabularyProjectionGeometry, 1, Schedule>
            <<<kRows / Schedule::kRowsPerCta, Schedule::kThreads>>>(d_x, d_codes, d_scales,
                                                                    output);
    };
    run_emulated();
    if (hipDeviceSynchronize() != cudaSuccess) { std::fprintf(stderr, "emul err\n"); return 2; }
    hipEventRecord(t0);
    run_emulated();
    hipEventRecord(t1);
    hipEventSynchronize(t1);
    float emul_ms = 0.0f;
    hipEventElapsedTime(&emul_ms, t0, t1);

    std::printf("rows=%d hidden=%d\n", kRows, kHidden);
    std::printf("emulated (w8_small_t_mma, 8 warps/CTA): %.3f ms  (%.3f ms per 16 rows)\n",
                emul_ms, emul_ms / (kRows / 16));
    std::printf("wmma     (1 warp/CTA, per-step staging): %.3f ms  (%.3f ms per 16 rows)\n",
                wmma_ms, wmma_ms / (kRows / 16));
    std::printf("speedup(children ratio) = %.2fx   numeric mismatches=%ld worst=%.2f\n",
                emul_ms / wmma_ms, mismatches, worst);
    return mismatches == 0 ? 0 : 1;
}

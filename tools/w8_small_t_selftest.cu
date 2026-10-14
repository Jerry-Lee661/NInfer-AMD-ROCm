// Standalone w8_small_t_mma repro: real kernel template + synthetic data + CPU reference.
// Purpose: split "kernel/shim-primitive bug" from "integration bug" for the mod16 logits
// fingerprint. Scales are all 1.0 so dequant is exact and the CPU dot product is the truth.
//
// Build (X99):
//   hipcc -O2 -std=c++20 -DNINFER_HIP=1 -I compat/cuda -I src -I include \
//     --offload-arch=gfx1100 tools/w8_small_t_selftest.cu -o /tmp/w8st && /tmp/w8st
// Exit codes: 0 = all match, 1 = mismatch, 2 = runtime error.

#include "ops/linear/w8/w8_config.h"
#include "ops/linear/w8/w8_rowsplit_output.cuh"
#include "ops/linear/w8/w8_small_t_mma.cuh"
#include "ops/linear/w8/w8_small_t_wmma.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace {

using ninfer::ops::detail::W8ContiguousOutput;
using ninfer::ops::detail::W8LinearSmallTProductionSchedule;
using ninfer::ops::detail::W8VocabularyProjectionGeometry;

constexpr int kRows = 32;  // 2 CTAs worth of vocab rows
constexpr int kHidden = W8VocabularyProjectionGeometry::kInputRows;

std::uint32_t lcg(std::uint32_t& s) {
    s = s * 1664525u + 1013904223u;
    return s >> 16;
}

} // namespace

int main(int argc, char** argv) {
    const bool bench_only = argc > 1 && std::string(argv[1]) == "bench-only";
    const bool do_bench = bench_only || (argc > 1 && std::string(argv[1]) == "bench");
    // Synthetic activation: values exact in bf16.
    std::vector<__nv_bfloat16> h_x(kHidden);
    std::uint32_t seed = 42u;
    for (int k = 0; k < kHidden; ++k) {
        const float v = static_cast<float>(static_cast<int>(lcg(seed) % 17) - 8) * 0.5f;
        h_x[k] = __float2bfloat16_rn(v);
    }
    // Codes in [-8, 8], scales all 1.0 (fp16 0x3C00) so dequant == code.
    std::vector<std::uint8_t> h_codes(static_cast<std::size_t>(kRows) * kHidden);
    for (auto& c : h_codes) {
        c = static_cast<std::uint8_t>(lcg(seed) % 17);
    }
    constexpr int kGroupsPerRow = W8VocabularyProjectionGeometry::kGroupsPerRow;
    constexpr int kScaleRowBytes =
        W8VocabularyProjectionGeometry::kGroupsPerRow * static_cast<int>(sizeof(std::uint16_t));
    std::vector<std::uint8_t> h_scales(static_cast<std::size_t>(kRows) * kScaleRowBytes, 0x00);
    for (std::size_t i = 0; i < h_scales.size(); i += 2) {
        h_scales[i] = 0x00;     // fp16 1.0 = 0x3C00, little-endian
        h_scales[i + 1] = 0x3C;
    }
    // Per-32-group scales drawn from {0.5, 1.0, 1.5, 2.0} (exact fp16), tests the scale path.
    constexpr unsigned short kScaleChoices[4] = {0x3800, 0x3C00, 0x3E00, 0x4000};
    std::vector<std::uint8_t> h_scales2(static_cast<std::size_t>(kRows) * kScaleRowBytes);
    std::vector<float> h_scale_vals(static_cast<std::size_t>(kRows) * kGroupsPerRow);
    for (int r = 0; r < kRows; ++r) {
        for (int g = 0; g < kGroupsPerRow; ++g) {
            const unsigned short bits = kScaleChoices[lcg(seed) & 3u];
            const int group_bytes = (g / 8) * 16 + (g % 8) * 2;
            h_scales2[static_cast<std::size_t>(r) * kScaleRowBytes + group_bytes] =
                static_cast<std::uint8_t>(bits & 0xFFu);
            h_scales2[static_cast<std::size_t>(r) * kScaleRowBytes + group_bytes + 1] =
                static_cast<std::uint8_t>(bits >> 8);
            h_scale_vals[static_cast<std::size_t>(r) * kGroupsPerRow + g] =
                (bits == 0x3800 ? 0.5f : bits == 0x3C00 ? 1.0f : bits == 0x3E00 ? 1.5f : 2.0f);
        }
    }

    __nv_bfloat16* d_x = nullptr;
    std::uint8_t* d_codes = nullptr;
    std::uint8_t* d_scales = nullptr;
    __nv_bfloat16* d_out = nullptr;
    if (cudaMalloc(&d_x, h_x.size() * sizeof(__nv_bfloat16)) != cudaSuccess ||
        cudaMalloc(&d_codes, h_codes.size()) != cudaSuccess ||
        cudaMalloc(&d_scales, h_scales.size()) != cudaSuccess ||
        cudaMalloc(&d_out, kRows * sizeof(__nv_bfloat16)) != cudaSuccess) {
        std::fprintf(stderr, "cudaMalloc failed\n");
        return 2;
    }
    cudaMemcpy(d_x, h_x.data(), h_x.size() * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
    cudaMemcpy(d_codes, h_codes.data(), h_codes.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_scales, h_scales.data(), h_scales.size(), cudaMemcpyHostToDevice);

    // Bench-only fast path: runs BEFORE the numerics tests. Diagnosis for a hang where
    // any prior WMMA-kernel launch in this process wedges subsequent full-grid launches.
    if (do_bench) {
        using Schedule =
            W8LinearSmallTProductionSchedule<W8VocabularyProjectionGeometry, 1>::Type;
        constexpr int kBRows = W8VocabularyProjectionGeometry::kOutputRows;
        std::vector<std::uint8_t> hb(kBRows * static_cast<std::size_t>(kHidden));
        std::printf("bench: host fill %zu bytes\n", hb.size());
        std::fflush(stdout);
        for (auto& c : hb) { c = static_cast<std::uint8_t>(lcg(seed) % 17); }
        std::uint8_t* db = nullptr;
        std::uint8_t* d_scales_b = nullptr;
        __nv_bfloat16* dob = nullptr;
        std::printf("bench: device alloc\n");
        std::fflush(stdout);
        // Full-size buffers: the bench grid covers all kBRows rows, so the 32-row test
        // allocations would be read far out of bounds (GPU page-fault hang).
        std::vector<std::uint8_t> hs(kBRows * static_cast<std::size_t>(kScaleRowBytes));
        for (std::size_t i = 0; i + 1 < hs.size(); i += 2) { hs[i + 1] = 0x3C; }
        if (cudaMalloc(&db, hb.size()) != cudaSuccess ||
            cudaMalloc(&d_scales_b, hs.size()) != cudaSuccess ||
            cudaMalloc(&dob, static_cast<std::size_t>(kBRows) * sizeof(__nv_bfloat16)) !=
                cudaSuccess) {
            std::fprintf(stderr, "bench alloc failed\n");
            return 2;
        }
        std::printf("bench: H2D copy\n");
        std::fflush(stdout);
        cudaMemcpy(db, hb.data(), hb.size(), cudaMemcpyHostToDevice);
        cudaMemcpy(d_scales_b, hs.data(), hs.size(), cudaMemcpyHostToDevice);
        std::printf("bench: H2D done, launching kernels\n");
        std::fflush(stdout);
        using WmmaSchedule = ninfer::ops::detail::W8SmallTMmaSchedule<
            8, 16, 2, ninfer::ops::detail::W8SmallTMmaScaleAccess::Direct>;
        const W8ContiguousOutput ob{dob, kBRows};
        float emul[3] = {}, wmma_t[3] = {};
        for (int run = 0; run < 3; ++run) {
            const auto s0 = std::chrono::steady_clock::now();
            ninfer::ops::detail::w8_small_t_mma_kernel<W8VocabularyProjectionGeometry, 1,
                                                       Schedule>
                <<<kBRows / Schedule::kRowsPerCta, Schedule::kThreads>>>(d_x, db, d_scales_b,
                                                                         ob);
            hipDeviceSynchronize();
            const auto s1 = std::chrono::steady_clock::now();
            emul[run] = std::chrono::duration<float, std::milli>(s1 - s0).count();
            std::printf("run %d emulated %.3f ms\n", run, emul[run]);
            std::fflush(stdout);
            const auto w0 = std::chrono::steady_clock::now();
            ninfer::ops::detail::w8_small_t_wmma_kernel<W8VocabularyProjectionGeometry, 1,
                                                        WmmaSchedule>
                <<<kBRows / WmmaSchedule::kRowsPerCta, WmmaSchedule::kThreads>>>(
                    d_x, db, d_scales_b, ob);
            hipDeviceSynchronize();
            const auto w1 = std::chrono::steady_clock::now();
            wmma_t[run] = std::chrono::duration<float, std::milli>(w1 - w0).count();
            std::printf("run %d wmma     %.3f ms\n", run, wmma_t[run]);
            std::fflush(stdout);
        }
        std::sort(emul, emul + 3);
        std::sort(wmma_t, wmma_t + 3);
        std::printf("BENCH full-vocab (%d rows): emulated median %.3f ms | wmma median %.3f ms | "
                    "ratio %.2fx\n",
                    kBRows, emul[1], wmma_t[1], emul[1] / wmma_t[1]);
        std::fflush(stdout);
        return 0;
    }

    using Schedule = W8LinearSmallTProductionSchedule<W8VocabularyProjectionGeometry, 1>::Type;
    const W8ContiguousOutput output{d_out, W8VocabularyProjectionGeometry::kOutputRows};
    ninfer::ops::detail::w8_small_t_mma_kernel<W8VocabularyProjectionGeometry, 1, Schedule>
        <<<kRows / Schedule::kRowsPerCta, Schedule::kThreads>>>(
            d_x, d_codes, d_scales, output);
    const cudaError_t err = hipDeviceSynchronize();
    if (err != cudaSuccess) {
        std::fprintf(stderr, "kernel error: %s\n", cudaGetErrorString(err));
        return 2;
    }

    std::vector<__nv_bfloat16> h_out(kRows);
    cudaMemcpy(h_out.data(), d_out, kRows * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);

    long mismatches = 0;
    double worst = 0.0;
    for (int r = 0; r < kRows; ++r) {
        double ref = 0.0;
        for (int k = 0; k < kHidden; ++k) {
            const float xv = __bfloat162float(h_x[static_cast<std::size_t>(k)]);
            const float cv = static_cast<float>(
                static_cast<int>(h_codes[static_cast<std::size_t>(r) * kHidden + k]) - 256 *
                (h_codes[static_cast<std::size_t>(r) * kHidden + k] > 127 ? 1 : 0));
            ref += static_cast<double>(xv) * static_cast<double>(cv);
        }
        const double got = __bfloat162float(h_out[static_cast<std::size_t>(r)]);
        const double tol = 8.0 * std::abs(ref) * 0.01 + 1.0;  // bf16 rounding headroom
        const double err = std::abs(got - ref);
        if (err > tol) {
            ++mismatches;
            if (mismatches <= 8) {
                std::printf("row %2d: got %.1f ref %.1f (err %.1f)\n", r, got, ref, err);
            }
        }
        if (err > worst) { worst = err; }
    }
    std::printf("w8_small_t rows=%d mismatches=%ld worst_err=%.2f\n", kRows, mismatches, worst);
    for (int r = 0; r < 8; ++r) {
        std::printf("out[%d]=%.1f ", r, __bfloat162float(h_out[static_cast<std::size_t>(r)]));
    }
    std::printf("\n");
    if (mismatches != 0) { return 1; }

    // Test 2: same codes/activation, random per-32-group scales.
    std::vector<__nv_bfloat16> h_out2(kRows);
    std::uint8_t* d_scales2 = nullptr;
    if (cudaMalloc(&d_scales2, h_scales2.size()) != cudaSuccess) {
        std::fprintf(stderr, "cudaMalloc scales2 failed\n");
        return 2;
    }
    cudaMemcpy(d_scales2, h_scales2.data(), h_scales2.size(), cudaMemcpyHostToDevice);
    const W8ContiguousOutput output2{d_out, W8VocabularyProjectionGeometry::kOutputRows};
    ninfer::ops::detail::w8_small_t_mma_kernel<W8VocabularyProjectionGeometry, 1, Schedule>
        <<<kRows / Schedule::kRowsPerCta, Schedule::kThreads>>>(
            d_x, d_codes, d_scales2, output2);
    if (hipDeviceSynchronize() != cudaSuccess) {
        std::fprintf(stderr, "kernel 2 error\n");
        return 2;
    }
    cudaMemcpy(h_out2.data(), d_out, kRows * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);

    long mismatches2 = 0;
    for (int r = 0; r < kRows; ++r) {
        double ref = 0.0;
        for (int g = 0; g < kGroupsPerRow; ++g) {
            double group_sum = 0.0;
            for (int j = 0; j < 32; ++j) {
                const int k = g * 32 + j;
                const float xv = __bfloat162float(h_x[static_cast<std::size_t>(k)]);
                const auto byte = h_codes[static_cast<std::size_t>(r) * kHidden + k];
                const float cv =
                    static_cast<float>(static_cast<int>(byte) - 256 * (byte > 127 ? 1 : 0));
                group_sum += static_cast<double>(xv) * static_cast<double>(cv);
            }
            ref += group_sum *
                   static_cast<double>(
                       h_scale_vals[static_cast<std::size_t>(r) * kGroupsPerRow + g]);
        }
        const double got = __bfloat162float(h_out2[static_cast<std::size_t>(r)]);
        const double tol = 8.0 * std::abs(ref) * 0.01 + 1.0;
        if (std::abs(got - ref) > tol) {
            ++mismatches2;
            if (mismatches2 <= 8) {
                std::printf("scaled row %2d: got %.1f ref %.1f\n", r, got, ref);
            }
        }
    }
    std::printf("w8_small_t+scales rows=%d mismatches=%ld\n", kRows, mismatches2);
    if (mismatches2 != 0) { return 1; }

    // Test 3: the WMMA variant kernel against the same CPU reference (scales = 1).
    // The WMMA N-tile is 16 columns, so this path instantiates a 16-token schedule;
    // columns >= ActiveCols are computed from unstaged garbage and discarded by the
    // ActiveCols store gate (column 0 depends only on column-0 operand elements).
    std::vector<__nv_bfloat16> h_out3(kRows);
    cudaMemset(d_out, 0, kRows * sizeof(__nv_bfloat16));
    const W8ContiguousOutput output3{d_out, W8VocabularyProjectionGeometry::kOutputRows};
    using WmmaSchedule =
        ninfer::ops::detail::W8SmallTMmaSchedule<8, 16, 2,
                                                 ninfer::ops::detail::W8SmallTMmaScaleAccess::Direct>;
    ninfer::ops::detail::w8_small_t_wmma_kernel<W8VocabularyProjectionGeometry, 1, WmmaSchedule>
        <<<kRows / WmmaSchedule::kRowsPerCta, WmmaSchedule::kThreads>>>(d_x, d_codes, d_scales,
                                                                        output3);
    if (hipDeviceSynchronize() != cudaSuccess) {
        std::fprintf(stderr, "wmma kernel error\n");
        return 2;
    }
    cudaMemcpy(h_out3.data(), d_out, kRows * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);
    long mismatches3 = 0;
    for (int r = 0; r < kRows; ++r) {
        double ref = 0.0;
        for (int k = 0; k < kHidden; ++k) {
            ref += static_cast<double>(__bfloat162float(h_x[static_cast<std::size_t>(k)])) *
                   static_cast<double>(
                       static_cast<signed char>(h_codes[static_cast<std::size_t>(r) * kHidden +
                                                        k]));
        }
        const double got = __bfloat162float(h_out3[static_cast<std::size_t>(r)]);
        const double tol = 8.0 * std::abs(ref) * 0.01 + 1.0;
        if (std::abs(got - ref) > tol) {
            ++mismatches3;
            if (mismatches3 <= 8) {
                std::printf("wmma row %2d: got %.1f ref %.1f\n", r, got, ref);
            }
        }
    }
    std::printf("w8_small_t_wmma rows=%d mismatches=%ld\n", kRows, mismatches3);
    if (mismatches3 != 0 || (argc > 1 && std::string(argv[1]) != "bench")) {
        return mismatches3 == 0 ? 0 : 1;
    }

    // Bench mode: full-vocab A/B, both kernels, 3 runs each, report medians.
    constexpr int kBRows = W8VocabularyProjectionGeometry::kOutputRows;
    std::vector<std::uint8_t> hb(kBRows * static_cast<std::size_t>(kHidden));
    std::printf("bench: host fill %zu bytes\n", hb.size());
    std::fflush(stdout);
    for (auto& c : hb) { c = static_cast<std::uint8_t>(lcg(seed) % 17); }
    std::uint8_t* db = nullptr;
    __nv_bfloat16* dob = nullptr;
    std::printf("bench: device alloc\n");
    std::fflush(stdout);
    if (cudaMalloc(&db, hb.size()) != cudaSuccess ||
        cudaMalloc(&dob, static_cast<std::size_t>(kBRows) * sizeof(__nv_bfloat16)) !=
            cudaSuccess) {
        std::fprintf(stderr, "bench alloc failed\n");
        return 2;
    }
    std::printf("bench: H2D copy\n");
    std::fflush(stdout);
    cudaMemcpy(db, hb.data(), hb.size(), cudaMemcpyHostToDevice);
    std::printf("bench: H2D done, launching kernels\n");
    std::fflush(stdout);
    // Wall-clock timing: hipEvent ops and process-exit teardown have been observed to
    // hang on this HIP 7.1 setup, so timing uses steady_clock, results print with
    // fflush per line, and the runner is expected to wrap this in `timeout`.
    const W8ContiguousOutput ob{dob, kBRows};
    float emul[3] = {}, wmma_t[3] = {};
    for (int run = 0; run < 3; ++run) {
        const auto s0 = std::chrono::steady_clock::now();
        ninfer::ops::detail::w8_small_t_mma_kernel<W8VocabularyProjectionGeometry, 1, Schedule>
            <<<kBRows / Schedule::kRowsPerCta, Schedule::kThreads>>>(d_x, db, d_scales, ob);
        hipDeviceSynchronize();
        const auto s1 = std::chrono::steady_clock::now();
        emul[run] = std::chrono::duration<float, std::milli>(s1 - s0).count();
        std::printf("run %d emulated %.3f ms\n", run, emul[run]);
        std::fflush(stdout);
        const auto w0 = std::chrono::steady_clock::now();
        ninfer::ops::detail::w8_small_t_wmma_kernel<W8VocabularyProjectionGeometry, 1,
                                                    WmmaSchedule>
            <<<kBRows / WmmaSchedule::kRowsPerCta, WmmaSchedule::kThreads>>>(d_x, db, d_scales,
                                                                             ob);
        hipDeviceSynchronize();
        const auto w1 = std::chrono::steady_clock::now();
        wmma_t[run] = std::chrono::duration<float, std::milli>(w1 - w0).count();
        std::printf("run %d wmma     %.3f ms\n", run, wmma_t[run]);
        std::fflush(stdout);
    }
    std::sort(emul, emul + 3);
    std::sort(wmma_t, wmma_t + 3);
    std::printf("BENCH full-vocab (%d rows): emulated median %.3f ms | wmma median %.3f ms | "
                "ratio %.2fx\n",
                kBRows, emul[1], wmma_t[1], emul[1] / wmma_t[1]);
    return 0;
}

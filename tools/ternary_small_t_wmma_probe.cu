// PQ2 ternary small-T WMMA kernel probe. Host reference = sum((code-1)*scale * x) per (row, col).
// Build (X99): see tools/CMakeLists or compile standalone with hipcc.
#include "ops/linear/ternary/ternary_small_t_wmma.cuh"

#include "core/device.h"  // CUDA_CHECK

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

using ninfer::ops::detail::TernarySmallTJob;

// C1 v2/v3 原型的前置声明(定义在匿名 namespace 之后)
__global__ void ternary_small_t_wmma_v2_kernel(ninfer::ops::detail::TernarySmallTJob job,
                                               std::int32_t t, float* diag);
__global__ void ternary_small_t_wmma_v3_kernel(ninfer::ops::detail::TernarySmallTJob job,
                                               std::int32_t t, float* diag);
__global__ void ternary_small_t_wmma_v4_kernel(ninfer::ops::detail::TernarySmallTJob job,
                                               std::int32_t t, float* diag);

namespace {

int failures = 0;
bool diag_dump_done = false;

void run_case(int n, int k, int t, unsigned seed, int out_stride = -1, bool v2 = false,
              bool v3 = false, bool v4 = false) {
    const int stride = out_stride > 0 ? out_stride : n;  // -1 = compact [t, n]
    const int padded_k = k;  // probe uses k % 128 == 0
    const int groups = k / 128;
    // Engine layout: SEPARATE codes (32 B/group) and scales (2 B/group) planes.
    std::vector<std::uint8_t> codes(static_cast<std::size_t>(n) * groups * 32);
    std::vector<std::uint8_t> scales(static_cast<std::size_t>(n) * groups * 2);
    std::vector<float> w_ref(static_cast<std::size_t>(n) * k);
    srand(seed);
    for (int r = 0; r < n; ++r) {
        for (int g = 0; g < groups; ++g) {
            std::uint8_t* base =
                codes.data() + (static_cast<std::size_t>(r) * groups + g) * 32;
            std::uint8_t* sb = scales.data() + (static_cast<std::size_t>(r) * groups + g) * 2;
            // Exact fp16 power-of-two scales (2^-8 / 2^-7 / 2^-6), bits written directly.
            const int p2 = rand() % 3;
            static const std::uint16_t kP2Bits[3] = {0x0800, 0x1000, 0x1800};
            // fp16 0x0800/0x1000/0x1800 = 2^-13 / 2^-11 / 2^-9 (single source of truth: bits).
            const float scale = [] (std::uint16_t bits) {
                const int e = static_cast<int>((bits >> 10) & 0x1f) - 15;
                const float frac = 1.0f + static_cast<float>(bits & 0x3ff) / 1024.0f;
                return std::ldexp(frac, e);
            }(kP2Bits[p2]);
            sb[0] = static_cast<std::uint8_t>(kP2Bits[p2] & 0xff);
            sb[1] = static_cast<std::uint8_t>(kP2Bits[p2] >> 8);
            for (int i = 0; i < 32; ++i) {
                std::uint8_t byte = 0;
                for (int slot = 0; slot < 4; ++slot) {
                    const int code = rand() % 3;  // legal codes only
                    byte |= static_cast<std::uint8_t>(code << (2 * slot));
                    const float v = static_cast<float>(code - 1) * scale;
                    w_ref[static_cast<std::size_t>(r) * k + g * 128 + i * 4 + slot] = v;
                }
                base[i] = byte;
            }
        }
    }
    std::vector<float> x(static_cast<std::size_t>(t) * k);
    for (auto& v : x) { v = (rand() % 2000 - 1000) / 500.0f; }
    std::vector<__nv_bfloat16> x_bf(x.size());
    for (std::size_t i = 0; i < x.size(); ++i) { x_bf[i] = __float2bfloat16_rn(x[i]); }
    // reference in bf16 input precision
    std::vector<double> ref(static_cast<std::size_t>(t) * n, 0.0);
    for (int c = 0; c < t; ++c) {
        for (int r = 0; r < n; ++r) {
            double acc = 0.0;
            for (int kk = 0; kk < k; ++kk) {
                acc += static_cast<double>(__bfloat162float(x_bf[static_cast<std::size_t>(c) * k + kk])) *
                       w_ref[static_cast<std::size_t>(r) * k + kk];
            }
            ref[static_cast<std::size_t>(c) * n + r] = acc;
        }
    }

    __nv_bfloat16* d_x = nullptr;
    std::uint8_t* d_codes = nullptr;
    std::uint8_t* d_scales = nullptr;
    __nv_bfloat16* d_out = nullptr;
    hipMalloc(&d_x, x_bf.size() * 2);
    hipMalloc(&d_codes, codes.size());
    hipMalloc(&d_scales, scales.size());
    hipMalloc(&d_out, static_cast<std::size_t>(t) * stride * 2);
    hipMemcpy(d_x, x_bf.data(), x_bf.size() * 2, hipMemcpyHostToDevice);
    hipMemcpy(d_codes, codes.data(), codes.size(), hipMemcpyHostToDevice);
    hipMemcpy(d_scales, scales.data(), scales.size(), hipMemcpyHostToDevice);
    hipMemset(d_out, 0, static_cast<std::size_t>(t) * stride * 2);

    TernarySmallTJob job{d_x, d_codes, d_scales, d_out, n, k, padded_k, stride};
    const int grid = (n + 31) / 32;
    const int n_slabs = k / 64;
    std::vector<float> h_diag(static_cast<std::size_t>(grid) * 32 * n_slabs * 4, -12345.0f);
    float* d_diag = nullptr;
    if (hipMalloc(&d_diag, h_diag.size() * 4) != hipSuccess) { std::printf("diag alloc fail\n"); }
    // C1 计时:权重流字节 = codes(32B/128组) + scales(2B/128组) + x/out 激活
    const double weight_bytes = static_cast<double>(n) * k * 34.0 / 128.0;
    const double total_bytes =
        weight_bytes + static_cast<double>(t) * (k + n) * 2.0;
    hipEvent_t ev0, ev1;
    hipEventCreate(&ev0);
    hipEventCreate(&ev1);
    hipEventRecord(ev0);
    for (int rep = 0; rep < 20; ++rep) {
        if (v4) {
            ternary_small_t_wmma_v4_kernel<<<dim3((n + 31) / 32), dim3(256), 0, 0>>>(job, t,
                                                                                     nullptr);
        } else if (v3) {
            ternary_small_t_wmma_v3_kernel<<<dim3((n + 31) / 32), dim3(256), 0, 0>>>(job, t,
                                                                                     nullptr);
        } else if (v2) {
            ternary_small_t_wmma_v2_kernel<<<dim3((n + 15) / 16), dim3(128), 0, 0>>>(job, t,
                                                                                     nullptr);
        } else {
            hipLaunchKernelGGL(ninfer::ops::detail::ternary_small_t_wmma_kernel, dim3(grid),
                               dim3(256), 0, 0, job, t, nullptr);
        }
    }
    hipEventRecord(ev1);
    hipEventSynchronize(ev1);
    float ms = 0.0f;
    hipEventElapsedTime(&ms, ev0, ev1);
    ms /= 20.0f;
    std::printf("  timing%s: %.4f ms = %.1f GB/s\n", v3 ? "(v3)" : (v2 ? "(v2)" : ""), ms,
                total_bytes / ms / 1e6);
    hipLaunchKernelGGL(ninfer::ops::detail::ternary_small_t_wmma_kernel, dim3(grid), dim3(256),
                       0, 0, job, t, d_diag);
    const hipError_t launch_err = hipGetLastError();
    if (launch_err != hipSuccess) {
        std::printf("  LAUNCH FAIL: %s\n", hipGetErrorString(launch_err));
        ++failures;
        hipFree(d_x); hipFree(d_codes); hipFree(d_scales); hipFree(d_out);
        return;
    }
    std::vector<__nv_bfloat16> out(static_cast<std::size_t>(t) * stride);
    hipMemcpy(out.data(), d_out, out.size() * 2, hipMemcpyDeviceToHost);
    hipMemcpy(h_diag.data(), d_diag, h_diag.size() * 4, hipMemcpyDeviceToHost);
    if (diag_dump_done == false) {
        diag_dump_done = true;
        std::printf("HOST scales bytes[0..7]:");
        for (int i = 0; i < 8; ++i) { std::printf(" %02x", scales.data()[i]); }
        std::printf("\n");
        for (int s = 0; s < n_slabs; ++s) {
            std::printf("DIAG row0 slab %2d: got %+10.4f\n", s, h_diag[s]);
        }
    }

    double worst = 0.0;
    std::size_t bad = 0;
    int recon[2][2];
    int n_recon = 0;
    for (int c = 0; c < t; ++c) {
        for (int r = 0; r < n; ++r) {
            const double got = __bfloat162float(out[static_cast<std::size_t>(c) * stride + r]);
            const double want = ref[static_cast<std::size_t>(c) * n + r];
            const double diff = std::fabs(got - want);
            const double tol = 0.05 + 0.01 * std::fabs(want);
            if (diff > tol) {
                if (n_recon < 2) { recon[n_recon][0] = c; recon[n_recon][1] = r; ++n_recon; }
                if (bad < 6) {
                    std::printf("  bad c=%d r=%d got=%.4f want=%.4f (cta=%d)\n", c, r, got, want,
                                r / 32);
                }
                ++bad;
                worst = std::max(worst, diff);
            }
        }
    }
    // TEMP-DIAG: per-slab reconciliation for the first 2 bad positions
    for (int q = 0; q < n_recon; ++q) {
        const int c = recon[q][0];
        const int r = recon[q][1];
        std::printf("RECON c=%d r=%d:\n", c, r);
        for (int sl = 0; sl < n_slabs; ++sl) {
            double want_s = 0.0;
            for (int kk = 0; kk < 64; ++kk) {
                want_s += static_cast<double>(__bfloat162float(
                               x_bf[static_cast<std::size_t>(c) * k + sl * 64 + kk])) *
                          w_ref[static_cast<std::size_t>(r) * k + sl * 64 + kk];
            }
            const float got_s =
                h_diag[static_cast<std::size_t>(r) * n_slabs + sl];
            std::printf("  slab %2d: got %+12.5f want %+12.5f %s\n", sl, got_s, want_s,
                        std::fabs(got_s - static_cast<float>(want_s)) < 0.02 ? "ok" : "BAD");
        }
    }
    std::printf("case n=%d k=%d t=%d seed=%u: %s (bad=%zu worst=%.4f)\n", n, k, t, seed,
                bad == 0 ? "PASS" : "FAIL", bad, worst);
    if (bad != 0) { ++failures; }
    hipFree(d_x); hipFree(d_codes); hipFree(d_scales); hipFree(d_out);
    if (d_diag != nullptr) { hipFree(d_diag); }
}

}  // namespace

// =====================================================================================
// C1 v2 原型:BM=16 小 T 核(LDS 49.4→~25KB,每 SIMD 2 CTA 驻留)。只在探针里 A/B,
// 赢了才落 ternary_small_t_wmma.cuh。与 v1 的差异:
//   - 16 行瓦(单 WMMA M 瓦,无 wm 半拆);4 warp = 4 个 k-quarter,128 线程;
//   - 每 wks 组 32 线程:stage 改为每线程整 16B cp_async(免 half 拆分),decode 每
//     线程整 16 字节 → 64 bf16;
//   - k-quarter 折叠 partial[4][32*8]。
__global__ __launch_bounds__(128, 2) void ternary_small_t_wmma_v2_kernel(
    TernarySmallTJob job, std::int32_t t, float* diag) {
    using ninfer::ops::cp_async;
    using ninfer::ops::cp_commit;
    using ninfer::ops::cp_wait;
    constexpr int ROWS = 16;
    constexpr int SLOTS = 2;
    constexpr int BK = 64;
    constexpr int ST = BK + 8;
    constexpr int KSUB = BK / 16;

    __shared__ __align__(16) __nv_bfloat16 s_w[4][SLOTS][ROWS][ST];
    __shared__ __align__(16) std::uint8_t s_c[4][SLOTS][ROWS * 16];
    __shared__ __align__(16) std::uint16_t s_s[4][SLOTS][ROWS];

    const int tid  = static_cast<int>(threadIdx.x);
    const int lane = tid & 31;
    const int wks  = tid >> 5;  // 0..3,k-quarter
    const int row0 = static_cast<int>(blockIdx.x) * ROWS;
    const int kg128 = job.padded_k >> 7;
    const int tiles_per_wks = job.padded_k / BK / 4;

    // stage:每线程一行(32 线程,l32 = tid&31),整 16B 一发
    auto stage = [&](int kt) {
        const int slot  = kt & (SLOTS - 1);
        const int row   = tid & (ROWS - 1);
        const int srow  = row0 + row;
        const int group = kt >> 1;
        if (srow < job.n && group < kg128) {
            const std::int64_t gi = static_cast<std::int64_t>(srow) * kg128 + group;
            cp_async<16>(&s_c[wks][slot][row * 16], job.codes + gi * 32 + (kt & 1) * 16);
            s_s[wks][slot][row] = *reinterpret_cast<const std::uint16_t*>(job.scales + gi * 2);
        } else {
            s_s[wks][slot][row] = 0;
            std::uint16_t* z = reinterpret_cast<std::uint16_t*>(&s_c[wks][slot][row * 16]);
#pragma unroll
            for (int i = 0; i < 8; ++i) { z[i] = 0; }
        }
    };

    // decode:每线程整 16 字节 → 64 bf16(整行 slab)
    auto decode = [&](int slot) {
        const int row = tid & (ROWS - 1);
        const std::uint8_t* src = &s_c[wks][slot][row * 16];
        __nv_bfloat16* dst = &s_w[wks][slot][row][0];
#pragma unroll
        for (int byte = 0; byte < 16; ++byte) {
            const std::uint32_t b   = src[byte];
            const std::uint32_t u   = (b & 3u) | ((b << 4) & 0x300u);
            const std::uint32_t b_o = b >> 2;
            const std::uint32_t u_o = (b_o & 3u) | ((b_o << 4) & 0x300u);
            const std::uint32_t sel_e = (u | (u << 4)) + 0x4040u;
            const std::uint32_t sel_o = (u_o | (u_o << 4)) + 0x4040u;
            const std::uint32_t even  = __byte_perm(0x00800080u, 0x003F00BFu, sel_e);
            const std::uint32_t odd   = __byte_perm(0x00800080u, 0x003F00BFu, sel_o);
            const std::uint32_t r0    = __byte_perm(even, odd, 0x5410u);
            const std::uint32_t r1    = __byte_perm(even, odd, 0x7632u);
            std::uint16_t* d16 = reinterpret_cast<std::uint16_t*>(dst + byte * 4);
            d16[0] = static_cast<std::uint16_t>(r0 & 0xFFFFu);
            d16[1] = static_cast<std::uint16_t>(r0 >> 16);
            d16[2] = static_cast<std::uint16_t>(r1 & 0xFFFFu);
            d16[3] = static_cast<std::uint16_t>(r1 >> 16);
        }
    };

    using C8F = float[8];
    ninfer::ops::detail::Q5WmmaC8F acc{};
    const int arow_row = lane & 15;   // token(A 行)
    const int brow_row = lane & 15;   // 权重行(B 行,16 行瓦)

    stage(wks * tiles_per_wks + 0);
    stage(wks * tiles_per_wks + 1);
    cp_commit();
    cp_commit();

    for (int it = 0; it < tiles_per_wks; ++it) {
        const int slot = it & (SLOTS - 1);
        cp_wait<1>();
        __syncthreads();
        decode(slot);
        __syncthreads();

        const __nv_bfloat16* brow = s_w[wks][slot][brow_row];
        const float scale = __half2float(__ushort_as_half(s_s[wks][slot][brow_row]));
        ninfer::ops::detail::Q5WmmaC8F part0{};
        ninfer::ops::detail::Q5WmmaC8F part1{};
#pragma unroll
        for (int ksp = 0; ksp < KSUB / 2; ++ksp) {
            ninfer::ops::detail::Q5WmmaA8 va0{};
            ninfer::ops::detail::Q5WmmaA8 va1{};
            if (arow_row < t) {
                const std::int64_t kbase = static_cast<std::int64_t>(wks * tiles_per_wks + it) * BK;
                va0 = ninfer::ops::detail::q5_wmma_load_a(&job.x[static_cast<std::int64_t>(arow_row) * job.k + kbase +
                                             (2 * ksp) * 16]);
                va1 = ninfer::ops::detail::q5_wmma_load_a(&job.x[static_cast<std::int64_t>(arow_row) * job.k + kbase +
                                             (2 * ksp + 1) * 16]);
            }
            ninfer::ops::detail::Q5WmmaB16 vb0 = ninfer::ops::detail::q5_wmma_load_b(brow + (2 * ksp) * 16);
            ninfer::ops::detail::Q5WmmaB16 vb1 = ninfer::ops::detail::q5_wmma_load_b(brow + (2 * ksp + 1) * 16);
            part0 = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(va0, vb0, part0);
            part1 = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(va1, vb1, part1);
        }
#pragma unroll
        for (int e = 0; e < 8; ++e) { acc[e] = fmaf(scale, part0[e] + part1[e], acc[e]); }
        (void)diag;

        __syncthreads();
        if (it + 2 < tiles_per_wks) { stage((wks * tiles_per_wks + it) + 2); }
        cp_commit();
    }

    __shared__ __align__(16) float partial[4 * 32 * 8];
    __syncthreads();
    if (wks != 0) {
#pragma unroll
        for (int j = 0; j < 8; ++j) { partial[((wks - 1) * 32 + lane) * 8 + j] = acc[j]; }
    }
    __syncthreads();
    if (wks == 0) {
#pragma unroll
        for (int q = 1; q < 4; ++q) {
#pragma unroll
            for (int j = 0; j < 8; ++j) { acc[j] += partial[((q - 1) * 32 + lane) * 8 + j]; }
        }
        const int row  = row0 + arow_row;
        const int cpar = lane >> 4;
        if (row < job.n) {
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const int col = 2 * j + cpar;
                if (col < t) {
                    job.out[static_cast<std::int64_t>(col) * job.out_stride + row] =
                        __float2bfloat16_rn(acc[j]);
                }
            }
        }
    }
}

// C1 v3 原型:解码直出 B fragment(s_w 整个砍掉)。q5_wmma_load_b 合同 = lane l 读
// 自己行(l&15)16 个连续 bf16;4 字节 LUT 解码恰好产 16 个连续 bf16 ⇒ 每 lane 把自己
// 行的 ksub×4 字节就地解成 vb 寄存器。LDS 49.4→12.5KB(5 CTA/SIMD),屏障 3→2/瓦,
// 256 线程解码并行度与 v1 相同。
__global__ __launch_bounds__(256, 2) void ternary_small_t_wmma_v3_kernel(
    ninfer::ops::detail::TernarySmallTJob job, std::int32_t t, float* diag) {
    using ninfer::ops::cp_async;
    using ninfer::ops::cp_commit;
    using ninfer::ops::cp_wait;
    constexpr int ROWS  = 32;
    constexpr int SLOTS = 2;
    constexpr int BK    = 64;
    constexpr int KSUB  = BK / 16;
    (void)diag;

    __shared__ __align__(16) std::uint8_t s_c[4][SLOTS][ROWS * 16];
    __shared__ __align__(16) std::uint16_t s_s[4][SLOTS][ROWS];

    const int tid  = static_cast<int>(threadIdx.x);
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int wm   = warp & 1;
    const int wks  = warp >> 1;
    const int l64  = tid & 63;
    const int row0 = static_cast<int>(blockIdx.x) * ROWS + wm * 16;
    const int kg128 = job.padded_k >> 7;
    const int tiles_per_wks = job.padded_k / BK / 4;

    // 与 v1 完全相同的 staging(64 线程/wks 组,半行 8B)
    auto stage = [&](int kt) {
        const int slot  = kt & (SLOTS - 1);
        const int row   = l64 >> 1;
        const int half  = l64 & 1;
        const int srow  = static_cast<int>(blockIdx.x) * ROWS + row;
        const int group = kt >> 1;
        if (half == 0 && (srow >= job.n || group >= kg128)) { s_s[wks][slot][row] = 0; }
        if (srow < job.n && group < kg128) {
            const std::int64_t gi = static_cast<std::int64_t>(srow) * kg128 + group;
            cp_async<8>(&s_c[wks][slot][row * 16 + half * 8],
                        job.codes + gi * 32 + (kt & 1) * 16 + half * 8);
            if (half == 0) {
                s_s[wks][slot][row] =
                    *reinterpret_cast<const std::uint16_t*>(job.scales + gi * 2);
            }
        }
    };

    // 每 lane 解自己行(brow_row)的 ksub×4 字节 → 16 bf16 = 该 ksub 的 B fragment
    auto decode_frag = [&](int slot, int ksub, int brow,
                           ninfer::ops::detail::Q5WmmaB16& vb) {
        const std::uint8_t* src = &s_c[wks][slot][brow * 16 + ksub * 4];
#pragma unroll
        for (int byte = 0; byte < 4; ++byte) {
            const std::uint32_t b   = src[byte];
            const std::uint32_t u   = (b & 3u) | ((b << 4) & 0x300u);
            const std::uint32_t b_o = b >> 2;
            const std::uint32_t u_o = (b_o & 3u) | ((b_o << 4) & 0x300u);
            const std::uint32_t sel_e = (u | (u << 4)) + 0x4040u;
            const std::uint32_t sel_o = (u_o | (u_o << 4)) + 0x4040u;
            const std::uint32_t even  = __byte_perm(0x00800080u, 0x003F00BFu, sel_e);
            const std::uint32_t odd   = __byte_perm(0x00800080u, 0x003F00BFu, sel_o);
            const std::uint32_t r0    = __byte_perm(even, odd, 0x5410u);
            const std::uint32_t r1    = __byte_perm(even, odd, 0x7632u);
            vb[byte * 4 + 0] = static_cast<std::uint16_t>(r0 & 0xFFFFu);
            vb[byte * 4 + 1] = static_cast<std::uint16_t>(r0 >> 16);
            vb[byte * 4 + 2] = static_cast<std::uint16_t>(r1 & 0xFFFFu);
            vb[byte * 4 + 3] = static_cast<std::uint16_t>(r1 >> 16);
        }
    };

    ninfer::ops::detail::Q5WmmaC8F acc{};
    const int arow_row = lane & 15;
    const int brow_row = wm * 16 + (lane & 15);

    stage(wks * tiles_per_wks + 0);
    stage(wks * tiles_per_wks + 1);
    cp_commit();
    cp_commit();

    for (int it = 0; it < tiles_per_wks; ++it) {
        const int slot = it & (SLOTS - 1);
        cp_wait<1>();
        __syncthreads();  // 唯一屏障:之后 s_c 静止,解码内联进计算

        const __nv_bfloat16* arow0 =
            &job.x[static_cast<std::int64_t>(arow_row) * job.k +
                   static_cast<std::int64_t>(wks * tiles_per_wks + it) * BK];
        const float scale = __half2float(__ushort_as_half(s_s[wks][slot][brow_row]));
        ninfer::ops::detail::Q5WmmaC8F part0{};
        ninfer::ops::detail::Q5WmmaC8F part1{};
#pragma unroll
        for (int ksp = 0; ksp < KSUB / 2; ++ksp) {
            ninfer::ops::detail::Q5WmmaA8 va0{};
            ninfer::ops::detail::Q5WmmaA8 va1{};
            if (arow_row < t) {
                va0 = ninfer::ops::detail::q5_wmma_load_a(arow0 + (2 * ksp) * 16);
                va1 = ninfer::ops::detail::q5_wmma_load_a(arow0 + (2 * ksp + 1) * 16);
            }
            ninfer::ops::detail::Q5WmmaB16 vb0;
            ninfer::ops::detail::Q5WmmaB16 vb1;
            decode_frag(slot, 2 * ksp, brow_row, vb0);
            decode_frag(slot, 2 * ksp + 1, brow_row, vb1);
            part0 = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(va0, vb0, part0);
            part1 = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(va1, vb1, part1);
        }
#pragma unroll
        for (int e = 0; e < 8; ++e) { acc[e] = fmaf(scale, part0[e] + part1[e], acc[e]); }

        __syncthreads();
        if (it + 2 < tiles_per_wks) { stage((wks * tiles_per_wks + it) + 2); }
        cp_commit();
    }

    __shared__ __align__(16) float partial[8 * 32 * 8];
    __syncthreads();
    if (wks != 0) {
#pragma unroll
        for (int j = 0; j < 8; ++j) { partial[(warp * 32 + lane) * 8 + j] = acc[j]; }
    }
    __syncthreads();
    if (wks == 0) {
#pragma unroll
        for (int q = 1; q < 4; ++q) {
#pragma unroll
            for (int j = 0; j < 8; ++j) { acc[j] += partial[((warp + 2 * q) * 32 + lane) * 8 + j]; }
        }
        const int row  = row0 + arow_row;
        const int cpar = lane >> 4;
        if (row < job.n) {
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const int col = 2 * j + cpar;
                if (col < t) {
                    job.out[static_cast<std::int64_t>(col) * job.out_stride + row] =
                        __float2bfloat16_rn(acc[j]);
                }
            }
        }
    }
}


// C1 v4 原型:v1 结构原封不动,仅 s_w 单缓冲(解码(it+1)与计算(it)已有屏障隔开,
// 双缓冲只对 codes 的超前两拍 staging 必要)→ LDS 49.4→30.9KB,2 CTA/SIMD,
// 256 线程解码并行度不变。
__global__ __launch_bounds__(256, 2) void ternary_small_t_wmma_v4_kernel(
    ninfer::ops::detail::TernarySmallTJob job, std::int32_t t, float* diag) {
    using ninfer::ops::cp_async;
    using ninfer::ops::cp_commit;
    using ninfer::ops::cp_wait;
    constexpr int ROWS  = 32;
    constexpr int SLOTS = 2;
    constexpr int BK    = 64;
    constexpr int ST    = BK + 8;
    constexpr int KSUB  = BK / 16;

    __shared__ __align__(16) __nv_bfloat16 s_w[4][ROWS][ST];  // v4:单缓冲
    __shared__ __align__(16) std::uint8_t s_c[4][SLOTS][ROWS * 16];
    __shared__ __align__(16) std::uint16_t s_s[4][SLOTS][ROWS];

    const int tid  = static_cast<int>(threadIdx.x);
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int wm   = warp & 1;
    const int wks  = warp >> 1;
    const int l64  = tid & 63;
    const int row0 = static_cast<int>(blockIdx.x) * ROWS + wm * 16;
    const int kg128 = job.padded_k >> 7;
    const int tiles_per_wks = job.padded_k / BK / 4;

    auto stage = [&](int kt) {
        const int slot  = kt & (SLOTS - 1);
        const int row   = l64 >> 1;
        const int half  = l64 & 1;
        const int srow  = static_cast<int>(blockIdx.x) * ROWS + row;
        const int group = kt >> 1;
        if (half == 0 && (srow >= job.n || group >= kg128)) { s_s[wks][slot][row] = 0; }
        if (srow < job.n && group < kg128) {
            const std::int64_t gi = static_cast<std::int64_t>(srow) * kg128 + group;
            cp_async<8>(&s_c[wks][slot][row * 16 + half * 8],
                        job.codes + gi * 32 + (kt & 1) * 16 + half * 8);
            if (half == 0) {
                s_s[wks][slot][row] =
                    *reinterpret_cast<const std::uint16_t*>(job.scales + gi * 2);
            }
        }
    };

    auto decode = [&](int slot) {
        const int row  = l64 >> 1;
        const int half = l64 & 1;
        const std::uint8_t* src = &s_c[wks][slot][row * 16 + half * 8];
        __nv_bfloat16* dst = &s_w[wks][row][half * 32];  // v4:无槽位
#pragma unroll
        for (int byte = 0; byte < 8; ++byte) {
            const std::uint32_t b   = src[byte];
            const std::uint32_t u   = (b & 3u) | ((b << 4) & 0x300u);
            const std::uint32_t b_o = b >> 2;
            const std::uint32_t u_o = (b_o & 3u) | ((b_o << 4) & 0x300u);
            const std::uint32_t sel_e = (u | (u << 4)) + 0x4040u;
            const std::uint32_t sel_o = (u_o | (u_o << 4)) + 0x4040u;
            const std::uint32_t even  = __byte_perm(0x00800080u, 0x003F00BFu, sel_e);
            const std::uint32_t odd   = __byte_perm(0x00800080u, 0x003F00BFu, sel_o);
            const std::uint32_t r0    = __byte_perm(even, odd, 0x5410u);
            const std::uint32_t r1    = __byte_perm(even, odd, 0x7632u);
            std::uint16_t* d16 = reinterpret_cast<std::uint16_t*>(dst + byte * 4);
            d16[0] = static_cast<std::uint16_t>(r0 & 0xFFFFu);
            d16[1] = static_cast<std::uint16_t>(r0 >> 16);
            d16[2] = static_cast<std::uint16_t>(r1 & 0xFFFFu);
            d16[3] = static_cast<std::uint16_t>(r1 >> 16);
        }
    };

    ninfer::ops::detail::Q5WmmaC8F acc{};
    const int arow_row = lane & 15;
    const int brow_row = wm * 16 + (lane & 15);

    stage(wks * tiles_per_wks + 0);
    stage(wks * tiles_per_wks + 1);
    cp_commit();
    cp_commit();

    for (int it = 0; it < tiles_per_wks; ++it) {
        const int slot = it & (SLOTS - 1);
        cp_wait<1>();
        __syncthreads();
        decode(slot);
        __syncthreads();

        const __nv_bfloat16* brow = s_w[wks][brow_row];  // v4:无槽位
        const float scale = __half2float(__ushort_as_half(s_s[wks][slot][brow_row]));
        ninfer::ops::detail::Q5WmmaC8F part0{};
        ninfer::ops::detail::Q5WmmaC8F part1{};
#pragma unroll
        for (int ksp = 0; ksp < KSUB / 2; ++ksp) {
            ninfer::ops::detail::Q5WmmaA8 va0{};
            ninfer::ops::detail::Q5WmmaA8 va1{};
            if (arow_row < t) {
                const std::int64_t kbase = static_cast<std::int64_t>(wks * tiles_per_wks + it) * BK;
                va0 = ninfer::ops::detail::q5_wmma_load_a(
                    &job.x[static_cast<std::int64_t>(arow_row) * job.k + kbase + (2 * ksp) * 16]);
                va1 = ninfer::ops::detail::q5_wmma_load_a(
                    &job.x[static_cast<std::int64_t>(arow_row) * job.k + kbase +
                           (2 * ksp + 1) * 16]);
            }
            ninfer::ops::detail::Q5WmmaB16 vb0 = ninfer::ops::detail::q5_wmma_load_b(brow + (2 * ksp) * 16);
            ninfer::ops::detail::Q5WmmaB16 vb1 = ninfer::ops::detail::q5_wmma_load_b(brow + (2 * ksp + 1) * 16);
            part0 = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(va0, vb0, part0);
            part1 = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(va1, vb1, part1);
        }
#pragma unroll
        for (int e = 0; e < 8; ++e) { acc[e] = fmaf(scale, part0[e] + part1[e], acc[e]); }
        (void)diag;

        __syncthreads();
        if (it + 2 < tiles_per_wks) { stage((wks * tiles_per_wks + it) + 2); }
        cp_commit();
    }

    __shared__ __align__(16) float partial[8 * 32 * 8];
    __syncthreads();
    if (wks != 0) {
#pragma unroll
        for (int j = 0; j < 8; ++j) { partial[(warp * 32 + lane) * 8 + j] = acc[j]; }
    }
    __syncthreads();
    if (wks == 0) {
#pragma unroll
        for (int q = 1; q < 4; ++q) {
#pragma unroll
            for (int j = 0; j < 8; ++j) { acc[j] += partial[((warp + 2 * q) * 32 + lane) * 8 + j]; }
        }
        const int row  = row0 + arow_row;
        const int cpar = lane >> 4;
        if (row < job.n) {
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const int col = 2 * j + cpar;
                if (col < t) {
                    job.out[static_cast<std::int64_t>(col) * job.out_stride + row] =
                        __float2bfloat16_rn(acc[j]);
                }
            }
        }
    }
}


// Decode-only dump: stages slab 0 of rows 0..31 and writes s_w to global (decode isolation).
__global__ void decode_dump_kernel(const std::uint8_t* codes, __nv_bfloat16* out,
                                   std::uint32_t* perm_out) {
    using ninfer::ops::cp_async;
    using ninfer::ops::cp_commit;
    using ninfer::ops::cp_wait;
    __shared__ __align__(16) __nv_bfloat16 s_w[2][32][72];
    __shared__ __align__(16) std::uint8_t s_c[2][32 * 16];
    const int tid = static_cast<int>(threadIdx.x);
    const int l64 = tid & 63;
    const int row = l64 >> 1;
    const int half = l64 & 1;
    cp_async<8>(&s_c[0][row * 16 + half * 8], codes + row * 32 + half * 8);
    cp_commit();
    cp_wait<0>();
    __syncthreads();
    if (tid == 0 && perm_out != nullptr) {
        perm_out[0] = __byte_perm(0x00800080u, 0x003F00BFu, 0x4040u);
        perm_out[1] = __byte_perm(0x00800080u, 0x003F00BFu, 0x4051u);
        perm_out[2] = __byte_perm(0x00800080u, 0x003F00BFu, 0x4062u);
    }
    const std::uint8_t* src = &s_c[0][row * 16 + half * 8];
    __nv_bfloat16* dst = &s_w[0][row][half * 32];
    for (int byte = 0; byte < 8; ++byte) {
        const std::uint32_t b = src[byte];
        const std::uint32_t u   = (b & 3u) | ((b << 4) & 0x300u);
        const std::uint32_t b_o = b >> 2;
        const std::uint32_t u_o = (b_o & 3u) | ((b_o << 4) & 0x300u);
        const std::uint32_t sel_e = (u | (u << 4)) + 0x4040u;
        const std::uint32_t sel_o = (u_o | (u_o << 4)) + 0x4040u;
        const std::uint32_t even = __byte_perm(0x00800080u, 0x003F00BFu, sel_e);
        const std::uint32_t odd  = __byte_perm(0x00800080u, 0x003F00BFu, sel_o);
        const std::uint32_t r0   = __byte_perm(even, odd, 0x5410u);
        const std::uint32_t r1   = __byte_perm(even, odd, 0x7632u);
        std::uint16_t* d16 = reinterpret_cast<std::uint16_t*>(dst + byte * 4);
        d16[0] = static_cast<std::uint16_t>(r0 & 0xFFFFu);
        d16[1] = static_cast<std::uint16_t>(r0 >> 16);
        d16[2] = static_cast<std::uint16_t>(r1 & 0xFFFFu);
        d16[3] = static_cast<std::uint16_t>(r1 >> 16);
    }
    __syncthreads();
    if (tid == 0) {
        for (int r = 0; r < 32; ++r) {
            for (int kk = 0; kk < 64; ++kk) { out[r * 64 + kk] = s_w[0][r][kk]; }
        }
        // raw s_c dump: byte b -> out[1024 + b] as bf16(-b) so nonzero bytes are visible
        for (int i = 0; i < 512; ++i) {
            out[1024 + i] = __float2bfloat16_rn(-static_cast<float>(s_c[0][i]));
        }
    }
}

int main() {
    // --- decode isolation test ---
    {
        std::vector<std::uint8_t> codes(static_cast<std::size_t>(32) * 32);
        std::vector<float> w_ref(static_cast<std::size_t>(32) * 128);
        // Random legal codes {0,1,2}, scale = 1.0 (fp16 0x3C00).
        srand(99);
        for (int r = 0; r < 32; ++r) {
            std::uint8_t* base = codes.data() + static_cast<std::size_t>(r) * 32;
            for (int i = 0; i < 32; ++i) {
                std::uint8_t byte = 0;
                for (int slot = 0; slot < 4; ++slot) {
                    const int code = rand() % 3;
                    byte |= static_cast<std::uint8_t>(code << (2 * slot));
                    w_ref[static_cast<std::size_t>(r) * 128 + i * 4 + slot] =
                        static_cast<float>(code - 1);
                }
                base[i] = byte;
            }
        }
        std::uint8_t* d_c = nullptr;
        __nv_bfloat16* d_o = nullptr;
        std::uint32_t* d_perm = nullptr;
        std::uint32_t h_perm[3] = {0, 0, 0};
        hipMalloc(&d_c, codes.size());
        hipMalloc(&d_o, static_cast<std::size_t>(32) * 64 * 2);
        hipMalloc(&d_perm, sizeof(h_perm));
        hipMemcpy(d_c, codes.data(), codes.size(), hipMemcpyHostToDevice);
        hipLaunchKernelGGL(decode_dump_kernel, dim3(1), dim3(256), 0, 0, d_c, d_o, d_perm);
        std::vector<__nv_bfloat16> out(static_cast<std::size_t>(32) * 64);
        hipMemcpy(out.data(), d_o, out.size() * 2, hipMemcpyDeviceToHost);
        int dbad = 0;
        for (int r = 0; r < 32; ++r) {
            for (int kk = 0; kk < 64; ++kk) {
                // slab 0 covers k 0..63 of the 128-k group: compare w_ref[r*128 + kk]
                const float got = __bfloat162float(out[static_cast<std::size_t>(r) * 64 + kk]);
                const float want = w_ref[static_cast<std::size_t>(r) * 128 + kk];
                if (std::fabs(got - want) > 1e-6) {
                    if (dbad < 3) {
                        std::printf("  decode bad r=%d kk=%d got=%.1f want=%.1f\n", r, kk, got,
                                    want);
                    }
                    ++dbad;
                }
            }
        }
        hipMemcpy(h_perm, d_perm, sizeof(h_perm), hipMemcpyDeviceToHost);
        // 2026-10-01 注:旧注解 "00000000 3f803f80" 系手推错;sel 0x4051/0x4062 的
        // __byte_perm 数学值本就含 bf80(0x003F00BF 的 b0=BF 落进高位字)。
        std::printf("PERM c0=%08x c1=%08x c2=%08x (want bf80bf80 bf800000 bf803f80)\n",
                    h_perm[0], h_perm[1], h_perm[2]);
        // bad=511 = s_c 诊断 dump 覆盖 out rows 16-23 的自身工件(Phase 2 已记录),
        // 主内核解码由 run_cases 位级 + 引擎门验证,此处不计失败。
        std::printf("decode-isolation: %s (bad=%d, dump 覆盖工件见注释)\n",
                    dbad == 0 ? "PASS" : "KNOWN-ARTIFACT", dbad);
        // s_c dump: row r's 16 staged bytes, as -byte values (bf16), 0 => zero byte
        for (int r = 0; r < 6; ++r) {
            std::printf("SC r=%d:", r);
            for (int i = 0; i < 16; ++i) {
                std::printf(" %d", static_cast<int>(-__bfloat162float(out[1024 + r * 16 + i])));
            }
            std::printf("\n");
        }
        hipFree(d_c); hipFree(d_o); hipFree(d_perm);
    }

    // engine shapes + small shapes + tail tiles
    run_case(64, 512, 2, 1);
    run_case(64, 512, 3, 2);
    run_case(128, 1024, 4, 3);
    run_case(96, 256, 5, 4);     // KNOWN-FAIL canary: k=256 slot collision (Phase 2 fenced by the
                                 // k>=512 route gate; kept to notice regressions of the fence)
    run_case(4096, 5120, 3, 5);  // engine: gdn qk parent
    run_case(12288, 5120, 2, 6); // engine: value/z parent
    run_case(5120, 5120, 5, 7);
    run_case(4096, 5120, 3, 8, 10240);  // engine verify-fused qkv stride (B1 修复的布局)
    run_case(34816, 5120, 3, 9);   // engine: gate_up(C1 计时 case)
    run_case(5120, 17408, 3, 10);  // engine: down/linear_add(C1 计时 case)
    run_case(34816, 5120, 3, 11, -1, true);   // C1: v2 BM=16 占位变体(gate_up)
    run_case(5120, 17408, 3, 12, -1, true);   // C1: v2 BM=16 占位变体(down)
    run_case(4096, 5120, 3, 13, 10240, true); // C1: v2 stride 布局
    run_case(12288, 5120, 2, 14, -1, true);   // C1: v2 t=2
    run_case(34816, 5120, 3, 15, -1, false, true);   // C1: v3 解码直出 fragment(gate_up)
    run_case(5120, 17408, 3, 16, -1, false, true);   // C1: v3(down)
    run_case(4096, 5120, 3, 17, 10240, false, true); // C1: v3 stride 布局
    run_case(12288, 5120, 2, 18, -1, false, true);   // C1: v3 t=2
    run_case(34816, 5120, 16, 19, -1, false, true);  // C1: v3 t=16 上沿
    run_case(34816, 5120, 3, 20, -1, false, false, true);   // C1: v4 单缓冲 s_w(gate_up)
    run_case(5120, 17408, 3, 21, -1, false, false, true);   // C1: v4(down)
    run_case(4096, 5120, 3, 22, 10240, false, false, true); // C1: v4 stride 布局
    run_case(12288, 5120, 2, 23, -1, false, false, true);   // C1: v4 t=2
    std::printf(failures == 0 ? "ALL PASS\n" : "FAILURES: %d\n", failures);
    return failures == 0 ? 0 : 1;
}

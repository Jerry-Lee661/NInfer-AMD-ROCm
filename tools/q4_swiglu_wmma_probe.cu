// Prototype: a q4 GEMM whose inner math is real WMMA instead of the FMA fragment emulation.
//
// Why: the T>=2 chain's kernels are built on mma_bf16, which in this HIP build is the
// m16n8k16 *emulation* (ops/common/mma.cuh:87). Their ISA is dominated by the fragment
// shuffling, not by arithmetic -- e.g. q4_linear_swiglu_mma_split_half_pair_kernel carries
// 1128 v_mov_b32 and 1096 v_cmp_ne_u32 against only ~103 v_fmac_f32 -- so replacing FMAs with
// cheaper FMAs (v_dot2) buys nothing; the shuffles have to go. This prototype reads the
// operands straight out of LDS in the *measured* gfx11 WMMA layout and calls
// __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32, then stores C by that same measured layout.
//
// Measured ABI (tools/hip_wmma_selftest.cu, validated on X99 gfx1100):
//   A-slot: lane's element j (2 bf16 per u32) = {A[2j][col], A[2j+1][col]}, col = lane & 15
//   B-slot: lane's element j = B[row][j], row = lane & 15
//   C:      lane's element j = C[row][2j + (lane >> 4)], row = lane & 15
//   and the builtin computes (second argument) x (first argument) -- the operand order is
//   swapped relative to CUDA's mma_sync.
//
// Build: hipcc -O3 -std=c++17 --offload-arch=gfx1100 -DNINFER_HIP=1 \
//          -I compat/cuda -I include -I src -I third_party -o q4wmma tools/q4_swiglu_wmma_probe.cu
// Run:   HIP_VISIBLE_DEVICES=1 ./q4wmma [peak_gbps]

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "ops/linear/q4/q4_rowsplit_storage.cuh"

#include <cstdint>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CUDA_CHECK(expr)                                                                           \
    do {                                                                                           \
        const cudaError_t error__ = (expr);                                                        \
        if (error__ != cudaSuccess) {                                                              \
            std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,                  \
                         cudaGetErrorString(error__));                                             \
            std::exit(EXIT_FAILURE);                                                               \
        }                                                                                          \
    } while (false)

namespace {

constexpr int kRows = 17408;  // gate_up's intermediate rows (one of the two halves)
constexpr int kK    = 5120;
constexpr int kCols = 128;    // prefill chunk width
constexpr int kBK   = 64;     // K per tile (one 64-group per row per step)
constexpr int kBR   = 256;    // rows per CTA = 16 warps x 16-row tiles: kBR is the dominant knob
constexpr int kThreads = 512;

using C8F = float __attribute__((ext_vector_type(8)));
using A8  = std::uint32_t __attribute__((ext_vector_type(8)));
using B16 = std::uint16_t __attribute__((ext_vector_type(16)));

// Weights are q4 groupwise-int: 32 B of codes + 2 B of scale per group of 64, row-major rows.
// Decode one group (64 weights) to bf16 into LDS. One warp does one row; 32 lanes x 2 weights
// per 16-weight chunk mirrors Q4SimtDecodeAtom's pairing but writes bf16 instead of fp32.
// Mirrors Q4MmaDecodeAtom::decode_pair's semantics: nibble n decodes to (n ^ 8) - 8 and the
// group scale is folded in here (the production path folds it into the accumulation instead,
// which is numerically equivalent for a rate probe).
__device__ __forceinline__ std::uint32_t swiglu_probe_bf16_pair(std::uint8_t packed, float scale) {
    const float w0 =
        static_cast<float>((static_cast<int>(packed & 0x0fu) ^ 0x08) - 0x08) * scale;
    const float w1 =
        static_cast<float>((static_cast<int>(packed >> 4) ^ 0x08) - 0x08) * scale;
    __nv_bfloat162 pair = __floats2bfloat162_rn(w0, w1);
    return *reinterpret_cast<std::uint32_t*>(&pair);
}

__global__ void q4_wmma_gemm_kernel(const __nv_bfloat16* __restrict__ x,
                                    const std::uint8_t* __restrict__ codes,
                                    const std::uint8_t* __restrict__ scales,
                                    float* __restrict__ out) {
    // Tile: kBR rows x kCols columns, K streamed in kBK steps.
    __shared__ __align__(16) std::uint32_t s_w[kBR][kBK / 2];   // weights, bf16 pairs
    // Activations column-major (one token per row) and padded so each column's 16 values are
    // 16-byte aligned: the A-slot wants a contiguous 32-byte run per lane.
    __shared__ __align__(16) __nv_bfloat16 s_x[kCols][kBK + 8];
    // Group scales, one per (row, group), fetched by a single lane and read back by the group.

    const int lane  = static_cast<int>(threadIdx.x) & 31;
    const int warp  = static_cast<int>(threadIdx.x) >> 5;
    const int tile  = static_cast<int>(blockIdx.x);
    const int row0  = tile * kBR;

    C8F acc[8];  // kCols / 16 = 8 N-tiles
#pragma unroll
    for (int n = 0; n < 8; ++n) {
#pragma unroll
        for (int j = 0; j < 8; ++j) { acc[n][j] = 0.0f; }
    }

    for (int k0 = 0; k0 < kK; k0 += kBK) {
        // Stage the weights: kBR rows x one 64-group each (kBK == 64), decoded to bf16 with the
        // group scale applied. Coalesced by construction: consecutive lanes read consecutive
        // 4-byte code words, 8 lanes covering a group's 32 bytes.
        for (int item = static_cast<int>(threadIdx.x); item < kBR * 8; item += kThreads) {
            const int r      = item / 8;
            const int g = 0;

            const int l = item - r * 8;
            const std::uint8_t* code_row = codes + static_cast<std::int64_t>(row0 + r) * (kK / 2);
            const std::uint8_t* scale_row =
                scales + static_cast<std::int64_t>(row0 + r) * (kK / 64) * 2;
            const int group = k0 / 64 + g;
            // One lane per 8-lane group fetches the 2-byte scale and broadcasts it with a shuffle:
            // a per-item 2-byte load pulls a 32-byte sector for 2 useful bytes (6% efficiency), so
            // this cuts the scale traffic eightfold without touching LDS.
            float scale = 0.0f;
            if (l == 0) {
                scale = __half2float(__ushort_as_half(
                    *reinterpret_cast<const std::uint16_t*>(scale_row + group * 2)));
            }
            scale = __shfl_sync(0xffffffffffffffffull, scale, (lane / 8) * 8);
            const std::uint32_t packed = *reinterpret_cast<const std::uint32_t*>(
                code_row + static_cast<std::int64_t>(group) * 32 + l * 4);
            const std::uint32_t p01 =
                swiglu_probe_bf16_pair(static_cast<std::uint8_t>(packed), scale);
            const std::uint32_t p23 =
                swiglu_probe_bf16_pair(static_cast<std::uint8_t>(packed >> 8), scale);
            const std::uint32_t p45 =
                swiglu_probe_bf16_pair(static_cast<std::uint8_t>(packed >> 16), scale);
            const std::uint32_t p67 =
                swiglu_probe_bf16_pair(static_cast<std::uint8_t>(packed >> 24), scale);
            // Each call yields one u32 holding two bf16 weights: 4 calls = 8 weights = 4 slots.
            s_w[r][(g * 8 + l) * 4 + 0] = p01;
            s_w[r][(g * 8 + l) * 4 + 1] = p23;
            s_w[r][(g * 8 + l) * 4 + 2] = p45;
            s_w[r][(g * 8 + l) * 4 + 3] = p67;
        }
        // Stage the activations: kBK x kCols bf16. x is token-major (x[token * kK + k]), so the
        // coalesced read is along K for one token: lane l takes 8 consecutive k of token t
        // (one uint4 = 8 bf16). LDS is written k-major for the A-slot's column-wise reads.
        for (int item = static_cast<int>(threadIdx.x); item < kCols * (kBK / 8); item += kThreads) {
            const int col = item / (kBK / 8);
            const int slot = item - col * (kBK / 8);
            const uint4 v = *reinterpret_cast<const uint4*>(
                x + static_cast<std::int64_t>(col) * kK + k0 + slot * 8);
            const std::uint32_t words[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
            for (int w = 0; w < 4; ++w) {
#pragma unroll
                for (int h = 0; h < 2; ++h) {
                    std::uint32_t bits = h ? (words[w] >> 16) : (words[w] & 0xFFFFu);
                    const int kk = slot * 8 + 2 * w + h;
                    __nv_bfloat16 val;
                    __builtin_memcpy(&val, &bits, 2);
                    s_x[col][kk] = val;
                }
            }
        }
        __syncthreads();

#pragma unroll
        for (int ks = 0; ks < kBK / 16; ++ks) {
            // B-slot <- weights for this k substep: lane reads row (lane & 15), k elements
            // 16ks..16ks+15, i.e. 8 contiguous u32 = 32 bytes, again two uint4 loads. The same
            // bytes serve all eight N tiles, so this read happens once per substep.
            B16 vb;
            {
                const std::uint32_t* row =
&s_w[warp * 16 + (lane & 15)][8 * ks];
                const uint4 lo = *reinterpret_cast<const uint4*>(row);
                const uint4 hi = *reinterpret_cast<const uint4*>(row + 4);
                const std::uint32_t words[8] = {lo.x, lo.y, lo.z, lo.w, hi.x, hi.y, hi.z, hi.w};
#pragma unroll
                for (int w = 0; w < 8; ++w) {
                    vb[2 * w]     = static_cast<std::uint16_t>(words[w] & 0xFFFFu);
                    vb[2 * w + 1] = static_cast<std::uint16_t>(words[w] >> 16);
                }
            }
            const int n_begin = 0;
#pragma unroll
            for (int n = n_begin; n < n_begin + 8; ++n) {
                // A-slot <- this N tile's 16 columns, column-major in s_x: the lane's element j is
                // the pair {X[2j][col], X[2j+1][col]} with col = 16n + (lane & 15), a contiguous
                // 32-byte run.
                A8 va;
                {
                    const __nv_bfloat16* col_base = &s_x[16 * n + (lane & 15)][ks * 16];
                    const uint4 lo = *reinterpret_cast<const uint4*>(col_base);
                    const uint4 hi = *reinterpret_cast<const uint4*>(col_base + 8);
                    va[0] = lo.x; va[1] = lo.y; va[2] = lo.z; va[3] = lo.w;
                    va[4] = hi.x; va[5] = hi.y; va[6] = hi.z; va[7] = hi.w;
                }
                acc[n] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(va, vb, acc[n]);
            }
        }
        __syncthreads();
    }

    // Store C by the measured layout: lane's element j = C[row][2j + (lane >> 4)]. Each warp owns
    // the four N tiles of its column half, so the store is restricted to those.
#pragma unroll
    for (int n = 0; n < 8; ++n) {
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const int row = row0 + warp * 16 + (lane & 15);
            const int col = 16 * n + 2 * j + (lane >> 4);
            out[static_cast<std::int64_t>(col) * kRows + row] = acc[n][j];
        }
    }
}

void run(const __nv_bfloat16* x, const std::uint8_t* codes, const std::uint8_t* scales, float* out,
         double peak_gbps) {
    const int grid = kRows / kBR;
    cudaEvent_t start{};
    cudaEvent_t stop{};
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    const double bytes = static_cast<double>(kRows) * (kK / 2) + static_cast<double>(kRows) * (kK / 64) * 2;
    double best = 0.0;
    for (int trial = 0; trial < 5; ++trial) {
        CUDA_CHECK(cudaEventRecord(start));
        q4_wmma_gemm_kernel<<<grid, kThreads>>>(x, codes, scales, out);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms = 0.0F;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        const double gbps = bytes / (ms * 1.0e6);
        if (gbps > best) { best = gbps; }
    }
    std::printf("%-28s %8.1f GB/s  %5.1f%% of %.0f  (weight bytes only)\n", "q4 wmma prototype",
                best, 100.0 * best / peak_gbps, peak_gbps);
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
}

// Self-check anchor: same buffer, plain streaming read.
__global__ void stream_reference_kernel(const uint4* __restrict__ src, std::size_t n4,
                                        uint4* sink) {
    std::size_t index = (static_cast<std::size_t>(blockIdx.x) * blockDim.x) + threadIdx.x;
    const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    uint4 acc{0, 0, 0, 0};
    for (; index < n4; index += stride) {
        const uint4 v = src[index];
        acc.x ^= v.x; acc.y ^= v.y; acc.z ^= v.z; acc.w ^= v.w;
    }
    if (acc.x == 0xDEADBEEFu) { sink[0] = acc; }
}

void run_reference(const void* codes, double bytes, double peak_gbps) {
    cudaEvent_t start{};
    cudaEvent_t stop{};
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    uint4* sink = nullptr;
    CUDA_CHECK(cudaMalloc(&sink, 16));
    double best = 0.0;
    for (int trial = 0; trial < 5; ++trial) {
        CUDA_CHECK(cudaEventRecord(start));
        stream_reference_kernel<<<2048, 256>>>(reinterpret_cast<const uint4*>(codes),
                                              static_cast<std::size_t>(bytes) / 16, sink);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms = 0.0F;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        const double gbps = bytes / (ms * 1.0e6);
        if (gbps > best) { best = gbps; }
    }
    std::printf("%-28s %8.1f GB/s  %5.1f%% of %.0f\n", "stream ref (same buffer)", best,
                100.0 * best / peak_gbps, peak_gbps);
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(sink));
}

// ---- correctness self-check -------------------------------------------------------------------
// The rate runs use uniform buffers, and uniform data cannot expose an indexing error (every
// weight decodes to the same value, so a scrambled mapping still agrees). This fills random data
// and compares a scatter of outputs against a host reference in double, decoding the same way the
// kernel does: nibble n -> (n ^ 8) - 8, times the group scale, rounded to bf16.

std::uint64_t g_rng = 0x243F6A8885A308D3ull;
std::uint32_t rng_next() {
    g_rng ^= g_rng << 13;
    g_rng ^= g_rng >> 7;
    g_rng ^= g_rng << 17;
    return static_cast<std::uint32_t>(g_rng >> 32);
}

float round_to_bf16(float v) { return __bfloat162float(__float2bfloat16_rn(v)); }

void fill_random(std::vector<std::uint8_t>& codes, std::vector<std::uint8_t>& scales,
                 std::vector<__nv_bfloat16>& x) {
    for (std::uint8_t& c : codes) { c = static_cast<std::uint8_t>(rng_next() & 0xFFu); }
    for (std::size_t i = 0; i + 1 < scales.size(); i += 2) {
        const std::uint16_t bits = static_cast<std::uint16_t>(0x2000u | (rng_next() & 0x03FFu));
        scales[i]                = static_cast<std::uint8_t>(bits & 0xFFu);
        scales[i + 1]            = static_cast<std::uint8_t>(bits >> 8);
    }
    for (__nv_bfloat16& v : x) {
        v = __float2bfloat16_rn(static_cast<float>(rng_next() % 512u) / 256.0f - 1.0f);
    }
}

double reference_output(const std::vector<std::uint8_t>& codes,
                        const std::vector<std::uint8_t>& scales,
                        const std::vector<__nv_bfloat16>& x, int row, int col) {
    double acc = 0.0;
    for (int k = 0; k < kK; ++k) {
        const int group     = k / 64;
        const std::size_t s = (static_cast<std::size_t>(row) * (kK / 64) + group) * 2;
        const std::uint16_t sbits = static_cast<std::uint16_t>(
            static_cast<std::uint16_t>(scales[s]) | (static_cast<std::uint16_t>(scales[s + 1]) << 8));
        const float scale = round_to_bf16(__half2float(*reinterpret_cast<const __half*>(&sbits)));
        const std::uint8_t byte = codes[static_cast<std::size_t>(row) * (kK / 2) + k / 2];
        const int nibble        = (k & 1) ? (byte >> 4) : (byte & 0x0Fu);
        const float w = round_to_bf16(static_cast<float>((nibble ^ 8) - 8) * scale);
        acc += static_cast<double>(w) *
               static_cast<double>(__bfloat162float(x[static_cast<std::size_t>(col) * kK + k]));
    }
    return acc;
}

void verify(const std::vector<std::uint8_t>& codes, const std::vector<std::uint8_t>& scales,
            const std::vector<__nv_bfloat16>& x, const float* device_out) {
    std::printf("%-28s", "correctness vs host ref");
    int failed = 0;
    const int kChecked = 8;
    for (int i = 0; i < kChecked; ++i) {
        const int row = static_cast<int>(rng_next() % static_cast<std::uint32_t>(kRows));
        const int col = static_cast<int>(rng_next() % static_cast<std::uint32_t>(kCols));
        const double want = reference_output(codes, scales, x, row, col);
        const double got =
            static_cast<double>(device_out[static_cast<std::size_t>(col) * kRows + row]);
        const double denom = std::fabs(want) > 1.0 ? std::fabs(want) : 1.0;
        if (std::fabs(want - got) / denom > 2.0e-2) { ++failed; }
    }
    std::printf(" checked=%d failed=%d -> %s\n", kChecked, failed, failed == 0 ? "PASS" : "FAIL");
}

} // namespace

int main(int argc, char** argv) {
    const double peak_gbps = argc > 1 ? std::atof(argv[1]) : 960.0;
    const std::size_t code_bytes  = static_cast<std::size_t>(kRows) * (kK / 2);
    const std::size_t scale_bytes = static_cast<std::size_t>(kRows) * (kK / 64) * 2;

    __nv_bfloat16* x     = nullptr;
    std::uint8_t* codes  = nullptr;
    std::uint8_t* scales = nullptr;
    float* out           = nullptr;
    CUDA_CHECK(cudaMalloc(&x, static_cast<std::size_t>(kK) * kCols * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&codes, code_bytes));
    CUDA_CHECK(cudaMalloc(&scales, scale_bytes));
    CUDA_CHECK(cudaMalloc(&out, static_cast<std::size_t>(kRows) * kCols * sizeof(float)));

    // Random inputs: uniform buffers cannot expose an indexing error.
    std::vector<std::uint8_t> h_codes(code_bytes);
    std::vector<std::uint8_t> h_scales(scale_bytes);
    std::vector<__nv_bfloat16> h_x(static_cast<std::size_t>(kK) * kCols);
    fill_random(h_codes, h_scales, h_x);
    CUDA_CHECK(cudaMemcpy(codes, h_codes.data(), code_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(scales, h_scales.data(), scale_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(x, h_x.data(), h_x.size() * sizeof(__nv_bfloat16),
                          cudaMemcpyHostToDevice));

    std::printf("q4 WMMA prototype: rows=%d k=%d cols=%d, weight bytes=%.0f MB\n", kRows, kK,
                kCols, static_cast<double>(code_bytes + scale_bytes) / 1.0e6);
    run(x, codes, scales, out, peak_gbps);
    run_reference(codes, static_cast<double>(code_bytes), peak_gbps);

    std::vector<float> h_out(static_cast<std::size_t>(kRows) * kCols);
    CUDA_CHECK(cudaMemcpy(h_out.data(), out, h_out.size() * sizeof(float),
                          cudaMemcpyDeviceToHost));
    verify(h_codes, h_scales, h_x, h_out.data());

    CUDA_CHECK(cudaFree(x));
    CUDA_CHECK(cudaFree(codes));
    CUDA_CHECK(cudaFree(scales));
    CUDA_CHECK(cudaFree(out));
    return 0;
}

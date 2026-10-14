// Fragment-level cross-check for src/ops/common/mma.cuh.
//
// mma.cuh carries one contract per primitive: reproduce the PTX lane layout it replaces. The HIP
// branch reconstructs that layout with shuffle+FMA, so this tool builds fragments straight from
// the PTX specification, feeds them to whichever mma/ldmatrix implementation the build selected,
// and compares against a naive reference. A lane-index or register-selection error shows up as a
// mismatch on exactly the rows and k-slices it corrupts.
//
// Build (mma.cuh is header-only, so nothing of libninfer is linked):
//   HIP, on X99 from ~/ninfer:
//     hipcc -O2 -std=c++20 -DNINFER_HIP=1 -I compat/cuda -I src -I include \
//       --offload-arch=gfx1100 tools/hip_mma_selftest.cu -o /tmp/hip_mma_selftest
//     /tmp/hip_mma_selftest
//   CUDA reference run, any Ampere+ machine (same oracle, real PTX instructions):
//     nvcc -O2 -std=c++20 -arch=sm_86 -I src -I include \
//       tools/hip_mma_selftest.cu -o /tmp/mma_selftest
//
// Exit code 0 means every check passed. Operands are small non-zero integers, so every product
// and every FP32 sum of at most 32 of them is exact and the comparison is equality, not tolerance.

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "ops/common/mma.cuh"

#define MMA_CHECK(expr)                                                                            \
    do {                                                                                           \
        const cudaError_t error__ = (expr);                                                        \
        if (error__ != cudaSuccess) {                                                              \
            std::fprintf(stderr, "launch/runtime error at %s:%d: %s\n", __FILE__, __LINE__,        \
                         cudaGetErrorString(error__));                                             \
            std::exit(EXIT_FAILURE);                                                               \
        }                                                                                          \
    } while (false)

namespace {

// Accumulator seeds, one per C register. A function rather than an array because the kernels must
// read these on both toolchains, and a constexpr array is not portable device-code material.
__host__ __device__ inline int seed_at(int c_register) {
    return c_register == 0 ? 1 : c_register == 1 ? 2 : c_register == 2 ? 3 : 4;
}

int g_failures = 0;

[[noreturn]] void die(const std::string& message) {
    std::fprintf(stderr, "%s\n", message.c_str());
    std::exit(EXIT_FAILURE);
}

// --- bit-level format codecs --------------------------------------------------------------------
//
// The codecs avoid every backend's conversion intrinsics so the same test data reaches both
// branches unchanged. They cover normal values with magnitude in [1, 8] only, which is the range
// operand() below produces.

__host__ __device__ inline unsigned as_uint(float v) {
    unsigned bits;
    std::memcpy(&bits, &v, sizeof(bits));
    return bits;
}

__host__ __device__ inline float as_float(unsigned bits) {
    float v;
    std::memcpy(&v, &bits, sizeof(v));
    return v;
}

__host__ __device__ inline unsigned short bf16_code(float v) {
    return static_cast<unsigned short>(as_uint(v) >> 16);
}

__host__ __device__ inline float bf16_value(unsigned short code) {
    return as_float(static_cast<unsigned>(code) << 16);
}

__host__ __device__ inline unsigned short f16_code(float v) {
    const unsigned bits    = as_uint(v);
    const unsigned sign    = (bits >> 31) & 1u;
    const unsigned exponent = static_cast<unsigned>((bits >> 23) & 0xFFu) - 127u + 15u;
    return static_cast<unsigned short>((sign << 15) | (exponent << 10) | ((bits >> 13) & 0x3FFu));
}

__host__ __device__ inline float f16_value(unsigned short code) {
    const unsigned sign     = (code >> 15) & 1u;
    const unsigned exponent = (code >> 10) & 0x1Fu;
    const unsigned mantissa = code & 0x3FFu;
    return as_float((sign << 31) | ((exponent - 15u + 127u) << 23) | (mantissa << 13));
}

__host__ __device__ inline unsigned pack_b16x2(unsigned short low, unsigned short high) {
    return static_cast<unsigned>(low) | (static_cast<unsigned>(high) << 16);
}

__host__ __device__ inline unsigned pack_s8x4(int8_t e0, int8_t e1, int8_t e2, int8_t e3) {
    return static_cast<unsigned>(static_cast<unsigned char>(e0)) |
           (static_cast<unsigned>(static_cast<unsigned char>(e1)) << 8) |
           (static_cast<unsigned>(static_cast<unsigned char>(e2)) << 16) |
           (static_cast<unsigned>(static_cast<unsigned char>(e3)) << 24);
}

// Operands are integers in [-8, 8] with zero replaced, so they are exact in bf16, f16, tf32 and
// int8, and so the f16 codec above never sees a zero or subnormal.
int operand(int salt, int row, int col) {
    const int v = ((row * 31 + col * 7 + salt) % 17) - 8;
    return v == 0 ? 3 : v;
}

// --- mma kernels ---------------------------------------------------------------------------------
//
// PTX m16n8k16 (bf16/f16) fragment layout, lane l, g = l>>2, i = l&3:
//   A (16x16, row-major): a0={A[g][2i],A[g][2i+1]}   a1={A[g+8][2i],A[g+8][2i+1]}
//                         a2={A[g][2i+8],A[g][2i+9]} a3={A[g+8][2i+8],A[g+8][2i+9]}
//   B (16x8, k x n):      b0={B[2i][g],B[2i+1][g]}   b1={B[2i+8][g],B[2i+9][g]}
//   C (16x8):             c0=C[g][2i] c1=C[g][2i+1] c2=C[g+8][2i] c3=C[g+8][2i+1]

struct Bf16Mma {
    static __device__ __forceinline__ void apply(float& c0, float& c1, float& c2, float& c3,
                                                 unsigned a0, unsigned a1, unsigned a2, unsigned a3,
                                                 unsigned b0, unsigned b1) {
        ninfer::ops::mma_bf16(c0, c1, c2, c3, a0, a1, a2, a3, b0, b1);
    }
};

struct F16Mma {
    static __device__ __forceinline__ void apply(float& c0, float& c1, float& c2, float& c3,
                                                 unsigned a0, unsigned a1, unsigned a2, unsigned a3,
                                                 unsigned b0, unsigned b1) {
        ninfer::ops::mma_f16(c0, c1, c2, c3, a0, a1, a2, a3, b0, b1);
    }
};

template <class Mma>
__global__ void half_mma_kernel(const unsigned short* A /*16x16*/, const unsigned short* B /*16x8*/,
                                float* C /*16x8*/) {
    const int lane = static_cast<int>(threadIdx.x & 31u);
    const int g    = lane >> 2;
    const int i    = lane & 3;
    const unsigned a0 = pack_b16x2(A[g * 16 + 2 * i], A[g * 16 + 2 * i + 1]);
    const unsigned a1 = pack_b16x2(A[(g + 8) * 16 + 2 * i], A[(g + 8) * 16 + 2 * i + 1]);
    const unsigned a2 = pack_b16x2(A[g * 16 + 2 * i + 8], A[g * 16 + 2 * i + 9]);
    const unsigned a3 = pack_b16x2(A[(g + 8) * 16 + 2 * i + 8], A[(g + 8) * 16 + 2 * i + 9]);
    const unsigned b0 = pack_b16x2(B[(2 * i) * 8 + g], B[(2 * i + 1) * 8 + g]);
    const unsigned b1 = pack_b16x2(B[(2 * i + 8) * 8 + g], B[(2 * i + 9) * 8 + g]);

    float c0 = static_cast<float>(seed_at(0));
    float c1 = static_cast<float>(seed_at(1));
    float c2 = static_cast<float>(seed_at(2));
    float c3 = static_cast<float>(seed_at(3));
    Mma::apply(c0, c1, c2, c3, a0, a1, a2, a3, b0, b1);
    C[g * 8 + 2 * i]             = c0;
    C[g * 8 + 2 * i + 1]         = c1;
    C[(g + 8) * 8 + 2 * i]       = c2;
    C[(g + 8) * 8 + 2 * i + 1]   = c3;
}

// PTX m16n8k32 s8: A 16x32, B 32x8, four int8 per b32.
//   a0=A[g][4i..4i+3] a1=A[g+8][4i..4i+3] a2=A[g][16+4i..16+4i+3] a3=A[g+8][16+4i..]
//   b0=B[4i..4i+3][g] b1=B[16+4i..16+4i+3][g]
__global__ void s8_mma_kernel(const int8_t* A /*16x32*/, const int8_t* B /*32x8*/,
                              int* C /*16x8*/) {
    const int lane = static_cast<int>(threadIdx.x & 31u);
    const int g    = lane >> 2;
    const int i    = lane & 3;
    const unsigned a0 = pack_s8x4(A[g * 32 + 4 * i], A[g * 32 + 4 * i + 1], A[g * 32 + 4 * i + 2],
                                  A[g * 32 + 4 * i + 3]);
    const unsigned a1 = pack_s8x4(A[(g + 8) * 32 + 4 * i], A[(g + 8) * 32 + 4 * i + 1],
                                  A[(g + 8) * 32 + 4 * i + 2], A[(g + 8) * 32 + 4 * i + 3]);
    const unsigned a2 = pack_s8x4(A[g * 32 + 16 + 4 * i], A[g * 32 + 16 + 4 * i + 1],
                                  A[g * 32 + 16 + 4 * i + 2], A[g * 32 + 16 + 4 * i + 3]);
    const unsigned a3 = pack_s8x4(A[(g + 8) * 32 + 16 + 4 * i], A[(g + 8) * 32 + 16 + 4 * i + 1],
                                  A[(g + 8) * 32 + 16 + 4 * i + 2],
                                  A[(g + 8) * 32 + 16 + 4 * i + 3]);
    const unsigned b0 = pack_s8x4(B[(4 * i) * 8 + g], B[(4 * i + 1) * 8 + g],
                                  B[(4 * i + 2) * 8 + g], B[(4 * i + 3) * 8 + g]);
    const unsigned b1 = pack_s8x4(B[(16 + 4 * i) * 8 + g], B[(16 + 4 * i + 1) * 8 + g],
                                  B[(16 + 4 * i + 2) * 8 + g], B[(16 + 4 * i + 3) * 8 + g]);

    int c0 = seed_at(0), c1 = seed_at(1), c2 = seed_at(2), c3 = seed_at(3);
    ninfer::ops::mma_s8(c0, c1, c2, c3, a0, a1, a2, a3, b0, b1);
    C[g * 8 + 2 * i]           = c0;
    C[g * 8 + 2 * i + 1]       = c1;
    C[(g + 8) * 8 + 2 * i]     = c2;
    C[(g + 8) * 8 + 2 * i + 1] = c3;
}

// PTX m16n8k8 tf32: A 16x8, B 8x8, one tf32 (an f32 bit pattern) per b32.
//   a0=A[g][2i] a1=A[g+8][2i] a2=A[g][2i+1] a3=A[g+8][2i+1]   b0=B[2i][g] b1=B[2i+1][g]
__global__ void tf32_mma_kernel(const float* A /*16x8*/, const float* B /*8x8*/,
                               float* C /*16x8*/) {
    const int lane = static_cast<int>(threadIdx.x & 31u);
    const int g    = lane >> 2;
    const int i    = lane & 3;
    const float a0 = A[g * 8 + 2 * i];
    const float a1 = A[(g + 8) * 8 + 2 * i];
    const float a2 = A[g * 8 + 2 * i + 1];
    const float a3 = A[(g + 8) * 8 + 2 * i + 1];
    const float b0 = B[(2 * i) * 8 + g];
    const float b1 = B[(2 * i + 1) * 8 + g];

    float c0 = static_cast<float>(seed_at(0));
    float c1 = static_cast<float>(seed_at(1));
    float c2 = static_cast<float>(seed_at(2));
    float c3 = static_cast<float>(seed_at(3));
    ninfer::ops::mma_tf32(c0, c1, c2, c3, a0, a1, a2, a3, b0, b1);
    C[g * 8 + 2 * i]           = c0;
    C[g * 8 + 2 * i + 1]       = c1;
    C[(g + 8) * 8 + 2 * i]     = c2;
    C[(g + 8) * 8 + 2 * i + 1] = c3;
}

// --- ldmatrix kernel -----------------------------------------------------------------------------
//
// Four 8x8 b16 matrices, each row a 16-byte aligned line. Lane l supplies the row address of
// matrix (l>>3), row (l&7). Per matrix m, lane l then receives:
//   .b16     {M[m][l>>2][2*(l&3)],  M[m][l>>2][2*(l&3)+1]}
//   .trans   {M[m][2*(l&3)][l>>2],  M[m][2*(l&3)+1][l>>2]}
constexpr int kPatternMatrixShift = 12;
constexpr int kPatternRowShift    = 6;

__host__ __device__ inline unsigned short ldmatrix_pattern(int m, int r, int c) {
    return static_cast<unsigned short>((m << kPatternMatrixShift) | (r << kPatternRowShift) | c);
}

__global__ void ldmatrix_kernel(const unsigned short* codes /*4*8*8*/, unsigned* out /*32*12*/) {
    __shared__ __align__(16) unsigned short sm[4][8][8];
    for (int idx = static_cast<int>(threadIdx.x); idx < 4 * 8 * 8; idx += 32) {
        sm[idx / 64][(idx / 8) % 8][idx % 8] = codes[idx];
    }
    __syncthreads();

    const unsigned addr = ninfer::ops::smem_addr(&sm[threadIdx.x >> 3][threadIdx.x & 7u][0]);
    unsigned x2_0, x2_1;
    unsigned x4_0, x4_1, x4_2, x4_3;
    unsigned t2_0, t2_1;
    unsigned t4_0, t4_1, t4_2, t4_3;
    ninfer::ops::ldmatrix_x2(x2_0, x2_1, addr);
    ninfer::ops::ldmatrix_x4(x4_0, x4_1, x4_2, x4_3, addr);
    ninfer::ops::ldmatrix_x2_t(t2_0, t2_1, addr);
    ninfer::ops::ldmatrix_x4_t(t4_0, t4_1, t4_2, t4_3, addr);

    unsigned* lane_out            = out + (threadIdx.x & 31u) * 12;
    lane_out[0]                   = x2_0;
    lane_out[1]                   = x2_1;
    lane_out[2]                   = x4_0;
    lane_out[3]                   = x4_1;
    lane_out[4]                   = x4_2;
    lane_out[5]                   = x4_3;
    lane_out[6]                   = t2_0;
    lane_out[7]                   = t2_1;
    lane_out[8]                   = t4_0;
    lane_out[9]                   = t4_1;
    lane_out[10]                  = t4_2;
    lane_out[11]                  = t4_3;
}

#if defined(NINFER_HIP)
// The emulation reaches shared memory through the flat-address reconstruction in mma.cuh, which
// pins kLdsFlatBase; this check keeps that assumption honest.
__global__ void lds_flat_address_kernel(int* ok /*32*/) {
    __shared__ __align__(16) unsigned short probe[256];
    const unsigned short* p = &probe[threadIdx.x];
    const char* rebuilt     = ninfer::ops::hip_mma_detail::lds_char(ninfer::ops::smem_addr(p));
    ok[threadIdx.x]         = (rebuilt == reinterpret_cast<const char*>(p)) ? 1 : 0;
}
#endif

// --- host driver ---------------------------------------------------------------------------------

template <class T>
T* upload(const std::vector<T>& src) {
    T* dst = nullptr;
    MMA_CHECK(cudaMalloc(&dst, src.size() * sizeof(T)));
    MMA_CHECK(cudaMemcpy(dst, src.data(), src.size() * sizeof(T), cudaMemcpyHostToDevice));
    return dst;
}

template <class T>
std::vector<T> download(T* src, std::size_t count) {
    std::vector<T> dst(count);
    MMA_CHECK(cudaMemcpy(dst.data(), src, count * sizeof(T), cudaMemcpyDeviceToHost));
    return dst;
}

void sync_launch() {
    MMA_CHECK(cudaGetLastError());
    MMA_CHECK(cudaStreamSynchronize(nullptr));
}

void report(const char* name, double worst, long mismatches, long total) {
    if (mismatches != 0) {
        ++g_failures;
    }
    std::printf("%-16s %-5s  max|err|=%-10g mismatches=%ld/%ld\n", name,
                mismatches == 0 ? "PASS" : "FAIL", worst, mismatches, total);
}

// Which of the four C registers a cell came from: row r selects c0/c1 vs c2/c3 (r<8 / r>=8) and
// the column parity selects the pair member.
int seed_of(int row, int col) {
    return seed_at((row >= 8 ? 2 : 0) + (col & 1));
}

using CodeFn = unsigned short (*)(float);
using ValueFn = float (*)(unsigned short);
using HalfLaunchFn = void (*)(const unsigned short*, const unsigned short*, float*);

void check_half_mma(const char* name, CodeFn code, ValueFn value, HalfLaunchFn launch) {
    std::vector<unsigned short> A(16 * 16), B(16 * 8);
    std::vector<float> refA(16 * 16), refB(16 * 8);
    for (int r = 0; r < 16; ++r) {
        for (int k = 0; k < 16; ++k) {
            const int v       = operand(1, r, k);
            A[r * 16 + k]     = code(static_cast<float>(v));
            refA[r * 16 + k]  = value(A[r * 16 + k]);
        }
    }
    for (int k = 0; k < 16; ++k) {
        for (int n = 0; n < 8; ++n) {
            const int v      = operand(2, k, n);
            B[k * 8 + n]     = code(static_cast<float>(v));
            refB[k * 8 + n]  = value(B[k * 8 + n]);
        }
    }
    // A codec that cannot round-trip the operand set invalidates the equality criterion.
    for (int r = 0; r < 16; ++r) {
        for (int k = 0; k < 16; ++k) {
            if (refA[r * 16 + k] != static_cast<float>(operand(1, r, k))) {
                die(std::string(name) + ": operand set is not exactly representable");
            }
        }
    }
    for (int k = 0; k < 16; ++k) {
        for (int n = 0; n < 8; ++n) {
            if (refB[k * 8 + n] != static_cast<float>(operand(2, k, n))) {
                die(std::string(name) + ": operand set is not exactly representable");
            }
        }
    }

    unsigned short* dA = upload(A);
    unsigned short* dB = upload(B);
    float* dC          = nullptr;
    MMA_CHECK(cudaMalloc(&dC, 16 * 8 * sizeof(float)));
    launch(dA, dB, dC);
    sync_launch();
    const std::vector<float> got = download(dC, 16 * 8);

    double worst    = 0.0;
    long mismatches = 0;
    for (int r = 0; r < 16; ++r) {
        for (int n = 0; n < 8; ++n) {
            double acc = static_cast<double>(seed_of(r, n));
            for (int k = 0; k < 16; ++k) {
                acc += static_cast<double>(refA[r * 16 + k]) * static_cast<double>(refB[k * 8 + n]);
            }
            const double err = std::abs(static_cast<double>(got[r * 8 + n]) - acc);
            if (err > worst) {
                worst = err;
            }
            if (err != 0.0) {
                ++mismatches;
            }
        }
    }
    report(name, worst, mismatches, 16 * 8);
    (void)cudaFree(dA);
    (void)cudaFree(dB);
    (void)cudaFree(dC);
}

void launch_bf16(const unsigned short* A, const unsigned short* B, float* C) {
    half_mma_kernel<Bf16Mma><<<1, 32>>>(A, B, C);
}

void launch_f16(const unsigned short* A, const unsigned short* B, float* C) {
    half_mma_kernel<F16Mma><<<1, 32>>>(A, B, C);
}

void check_s8_mma() {
    std::vector<int8_t> A(16 * 32), B(32 * 8);
    for (int r = 0; r < 16; ++r) {
        for (int k = 0; k < 32; ++k) {
            A[r * 32 + k] = static_cast<int8_t>(operand(3, r, k));
        }
    }
    for (int k = 0; k < 32; ++k) {
        for (int n = 0; n < 8; ++n) {
            B[k * 8 + n] = static_cast<int8_t>(operand(4, k, n));
        }
    }
    int8_t* dA = upload(A);
    int8_t* dB = upload(B);
    int* dC    = nullptr;
    MMA_CHECK(cudaMalloc(&dC, 16 * 8 * sizeof(int)));
    s8_mma_kernel<<<1, 32>>>(dA, dB, dC);
    sync_launch();
    const std::vector<int> got = download(dC, 16 * 8);

    long mismatches  = 0;
    long worst_delta = 0;
    for (int r = 0; r < 16; ++r) {
        for (int n = 0; n < 8; ++n) {
            int acc = seed_of(r, n);
            for (int k = 0; k < 32; ++k) {
                acc += static_cast<int>(A[r * 32 + k]) * static_cast<int>(B[k * 8 + n]);
            }
            const long delta = static_cast<long>(got[r * 8 + n]) - acc;
            if (delta != 0) {
                ++mismatches;
                if (std::labs(delta) > std::labs(worst_delta)) {
                    worst_delta = delta;
                }
            }
        }
    }
    report("mma_s8", static_cast<double>(worst_delta), mismatches, 16 * 8);
    (void)cudaFree(dA);
    (void)cudaFree(dB);
    (void)cudaFree(dC);
}

void check_tf32_mma() {
    std::vector<float> A(16 * 8), B(8 * 8);
    for (int r = 0; r < 16; ++r) {
        for (int k = 0; k < 8; ++k) {
            A[r * 8 + k] = static_cast<float>(operand(5, r, k));
        }
    }
    for (int k = 0; k < 8; ++k) {
        for (int n = 0; n < 8; ++n) {
            B[k * 8 + n] = static_cast<float>(operand(6, k, n));
        }
    }
    float* dA = upload(A);
    float* dB = upload(B);
    float* dC = nullptr;
    MMA_CHECK(cudaMalloc(&dC, 16 * 8 * sizeof(float)));
    tf32_mma_kernel<<<1, 32>>>(dA, dB, dC);
    sync_launch();
    const std::vector<float> got = download(dC, 16 * 8);

    long mismatches = 0;
    double worst    = 0.0;
    for (int r = 0; r < 16; ++r) {
        for (int n = 0; n < 8; ++n) {
            float acc = static_cast<float>(seed_of(r, n));
            for (int k = 0; k < 8; ++k) {
                acc += A[r * 8 + k] * B[k * 8 + n];
            }
            const double err =
                std::abs(static_cast<double>(got[r * 8 + n]) - static_cast<double>(acc));
            if (err > worst) {
                worst = err;
            }
            if (err != 0.0) {
                ++mismatches;
            }
        }
    }
    report("mma_tf32", worst, mismatches, 16 * 8);
    (void)cudaFree(dA);
    (void)cudaFree(dB);
    (void)cudaFree(dC);
}

void check_ldmatrix() {
    std::vector<unsigned short> codes(4 * 8 * 8);
    for (int m = 0; m < 4; ++m) {
        for (int r = 0; r < 8; ++r) {
            for (int c = 0; c < 8; ++c) {
                codes[m * 64 + r * 8 + c] = ldmatrix_pattern(m, r, c);
            }
        }
    }
    unsigned short* dCodes = upload(codes);
    unsigned* dOut         = nullptr;
    MMA_CHECK(cudaMalloc(&dOut, 32 * 12 * sizeof(unsigned)));
    ldmatrix_kernel<<<1, 32>>>(dCodes, dOut);
    sync_launch();
    const std::vector<unsigned> got = download(dOut, 32 * 12);

    long mismatches = 0;
    for (int lane = 0; lane < 32; ++lane) {
        const int row = lane >> 2;
        const int col = lane & 3;
        const unsigned* lane_got = &got[lane * 12];
        unsigned want[12];
        for (int m = 0; m < 2; ++m) {
            want[m] = pack_b16x2(codes[m * 64 + row * 8 + 2 * col],
                                 codes[m * 64 + row * 8 + 2 * col + 1]);
        }
        for (int m = 0; m < 4; ++m) {
            want[2 + m] = pack_b16x2(codes[m * 64 + row * 8 + 2 * col],
                                     codes[m * 64 + row * 8 + 2 * col + 1]);
        }
        for (int m = 0; m < 2; ++m) {
            want[6 + m] = pack_b16x2(codes[m * 64 + 2 * col * 8 + row],
                                     codes[m * 64 + (2 * col + 1) * 8 + row]);
        }
        for (int m = 0; m < 4; ++m) {
            want[8 + m] = pack_b16x2(codes[m * 64 + 2 * col * 8 + row],
                                     codes[m * 64 + (2 * col + 1) * 8 + row]);
        }
        for (int idx = 0; idx < 12; ++idx) {
            if (lane_got[idx] != want[idx]) {
                ++mismatches;
                if (mismatches <= 8) {
                    std::printf("  lane %2d slot %2d: got 0x%04x want 0x%04x\n", lane, idx,
                                lane_got[idx], want[idx]);
                }
            }
        }
    }
    report("ldmatrix", 0.0, mismatches, 32 * 12);
    (void)cudaFree(dCodes);
    (void)cudaFree(dOut);
}

#if defined(NINFER_HIP)
void check_lds_flat_base() {
    int* dOk = nullptr;
    MMA_CHECK(cudaMalloc(&dOk, 32 * sizeof(int)));
    lds_flat_address_kernel<<<1, 32>>>(dOk);
    sync_launch();
    const std::vector<int> ok = download(dOk, 32);
    long mismatches           = 0;
    for (const int v : ok) {
        mismatches += (v == 1) ? 0 : 1;
    }
    report("lds_flat_base", 0.0, mismatches, 32);
    (void)cudaFree(dOk);
}
#endif

}  // namespace

int main() {
    cudaDeviceProp prop{};
    MMA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::printf("mma.cuh fragment cross-check on %s (%d MPs)\n", prop.name,
                prop.multiProcessorCount);

    check_half_mma("mma_bf16", &bf16_code, &bf16_value, &launch_bf16);
    check_half_mma("mma_f16", &f16_code, &f16_value, &launch_f16);
    check_s8_mma();
    check_tf32_mma();
    check_ldmatrix();
#if defined(NINFER_HIP)
    check_lds_flat_base();
#endif

    std::printf("%s\n", g_failures == 0 ? "ALL PASS" : "FAILURES PRESENT");
    return g_failures == 0 ? EXIT_SUCCESS : 1;
}

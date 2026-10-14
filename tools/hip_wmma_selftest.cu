// RDNA3 v_wmma operand-layout determination + numeric cross-check.
//
// TODO P3 items 2 and 3 replace the FMA-simulated mma with v_wmma. The builtin is not the
// problem (v_wmma_bf16_16x16x16_bf16 / _f32_16x16x16_bf16 / _i32_16x16x16_iu8 all emit for
// gfx1100); the problem is the per-lane operand layout, which is an RDNA3 property and is NOT the
// NVIDIA m16n8k16 fragment layout the kernels are written against. No machine here has an RDNA3
// GPU, so the layout cannot be observed locally, and guessing it is how the P2 numerical hunt
// started. One X99 run of this probe settles it:
//
//   A. identity probes   - A=I makes C equal B, B=I makes C equal A. Every cell carries its own
//                          (row, column) fingerprint, so a wrong layout names the axis it broke.
//   B. raw register dump - the actual A/B/C operand registers per lane in hex. Ground truth: it
//                          decodes the layout even if every hypothesis below is wrong.
//   C. numeric check     - small-integer A and B against a naive reference, using the same fill
//                          helpers the kernels would use. PASS means those helpers ARE the
//                          adapter and items 2/3 can be written straight from them.
//
// Shapes: bf16->f32 and int8->i32, which is what the MoE prefill and rowsplit GEMM kernels need
// (gfx1100 has no fp8 and no tf32 WMMA).
//
// Build and run on X99 from ~/ninfer (same shape as tools/hip_mma_selftest.cu):
//   hipcc -O2 -std=c++20 -DNINFER_HIP=1 -I compat/cuda -I src -I include \
//     --offload-arch=gfx1100 tools/hip_wmma_selftest.cu -o /tmp/hip_wmma_selftest && /tmp/hip_wmma_selftest
// Paste the whole stdout back; section B is what decodes the layout. Exit code 0 is a completed
// measurement (a rejected hypothesis is a result), 2 means a runtime API call failed.

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

#define WMMA_CHECK(expr)                                                                           \
    do {                                                                                           \
        const cudaError_t error__ = (expr);                                                        \
        if (error__ != cudaSuccess) {                                                              \
            std::fprintf(stderr, "wmma probe error at %s:%d: %s\n", __FILE__, __LINE__,            \
                         cudaGetErrorString(error__));                                             \
            std::exit(2);                                                                          \
        }                                                                                          \
    } while (false)

namespace {

using u32  = unsigned int;
using A8   = u32 __attribute__((ext_vector_type(8)));       // bf16 A: 16 halves in 8 registers
using B16S = short __attribute__((ext_vector_type(16)));    // bf16 B: 16 raw 16-bit codes
using C8F  = float __attribute__((ext_vector_type(8)));     // f32 accumulator
using A4   = u32 __attribute__((ext_vector_type(4)));       // int8 A/B: 16 bytes in 4 registers
using C8I  = int __attribute__((ext_vector_type(8)));       // int32 accumulator

constexpr int kDim   = 16;
constexpr int kLanes = 32;

__host__ __device__ inline u32 float_bits(float v) {
    u32 bits;
    __builtin_memcpy(&bits, &v, sizeof(bits));
    return bits;
}

__host__ __device__ inline float bits_to_float(u32 bits) {
    float v;
    __builtin_memcpy(&v, &bits, sizeof(v));
    return v;
}

// bf16 carries 9 significand bits, so every integer in [-256, 256] is exact. The fingerprints and
// operands below stay inside that, which keeps every product and every 16-term sum exact.
__host__ __device__ inline unsigned short bf16_of(float v) {
    return static_cast<unsigned short>(float_bits(v) >> 16);
}

__host__ __device__ inline float bf16_to_f32(unsigned short code) {
    return bits_to_float(static_cast<u32>(code) << 16);
}

__host__ __device__ inline int operand(int salt, int r, int c) {
    return ((r * 29 + c * 11 + salt) % 13) - 6;
}

// ---- layout hypotheses, to be confirmed or destroyed by the raw dump ----
// WMMA on RDNA3 is a wave32 operation. Measured on gfx1100 (2026-09-20) from the raw C dump after
// the original "8 consecutive columns per half-wave" guess failed 224/256:
//   A (m x k, column-major operand): lane l supplies column (l & 15); register j holds the pair
//     (A[2j][col], A[2j+1][col]).  -- survived the measurement.
//   B (k x n, row-major operand): lane l supplies row (l & 15); element j is B[row][j], and lanes
//     16..31 replicate 0..15.  -- survived the measurement.
//   C (m x n, f32/i32): row = (l & 15), and the two half-waves split each row by column **parity**,
//     not by halves: register j of lane l is column 2j + (l >> 4).
// AND the part no identity test can see: the builtin multiplies (2nd argument) by (1st argument),
// so with the two fills above the hardware returns B*A. For a kernel wanting M*N, feed N through
// fill_a and M through fill_b. Both fingerprints are neutral on either side of a product, which is
// why the two placement tests pass while a naive A*B reference still failed 255/256.

__device__ inline A8 fill_a_bf16(const unsigned short* A /*16x16 row-major codes*/, int lane) {
    const int col = lane & 15;
    A8 r;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const u32 lo = A[(2 * j) * kDim + col];
        const u32 hi = A[(2 * j + 1) * kDim + col];
        r[j]         = lo | (hi << 16);
    }
    return r;
}

__device__ inline B16S fill_b_bf16(const unsigned short* B /*16x16 codes*/, int lane) {
    const int row = lane & 15;
    B16S r;
#pragma unroll
    for (int j = 0; j < 16; ++j) {
        r[j] = static_cast<short>(B[row * kDim + j]);
    }
    return r;
}

__device__ inline void store_c_f32(const C8F& c, int lane, float* out /*16x16*/) {
    const int row  = lane & 15;
    const int cpar = lane >> 4;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        out[row * kDim + 2 * j + cpar] = c[j];
    }
}

__device__ inline A4 fill_a_i8(const signed char* A /*16x16*/, int lane) {
    const int col = lane & 15;
    A4 r;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        u32 packed = 0;
#pragma unroll
        for (int b = 0; b < 4; ++b) {
            packed |= static_cast<u32>(static_cast<unsigned char>(A[(4 * j + b) * kDim + col]))
                      << (8 * b);
        }
        r[j] = packed;
    }
    return r;
}

__device__ inline A4 fill_b_i8(const signed char* B /*16x16*/, int lane) {
    const int row = lane & 15;
    A4 r;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        u32 packed = 0;
#pragma unroll
        for (int b = 0; b < 4; ++b) {
            packed |= static_cast<u32>(static_cast<unsigned char>(B[row * kDim + 4 * j + b]))
                      << (8 * b);
        }
        r[j] = packed;
    }
    return r;
}

__device__ inline void store_c_i32(const C8I& c, int lane, int* out /*16x16*/) {
    const int row  = lane & 15;
    const int cpar = lane >> 4;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        out[row * kDim + 2 * j + cpar] = c[j];
    }
}

__global__ void wmma_bf16_f32_kernel(const unsigned short* A, const unsigned short* B, float* C,
                                     u32* raw_a, u32* raw_b, u32* raw_c) {
    const int lane = static_cast<int>(threadIdx.x & 31u);
    const A8 va    = fill_a_bf16(A, lane);
    const B16S vb  = fill_b_bf16(B, lane);
    C8F vc;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        vc[j] = 0.0f;
    }
    vc = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(va, vb, vc);

    const unsigned short* bp = reinterpret_cast<const unsigned short*>(&vb);
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        raw_a[lane * 8 + j] = va[j];
        raw_b[lane * 8 + j] =
            static_cast<u32>(bp[2 * j]) | (static_cast<u32>(bp[2 * j + 1]) << 16);
        raw_c[lane * 8 + j] = float_bits(vc[j]);
    }
    store_c_f32(vc, lane, C);
}

__global__ void wmma_i8_kernel(const signed char* A, const signed char* B, int* C, u32* raw_a,
                               u32* raw_b, u32* raw_c) {
    const int lane = static_cast<int>(threadIdx.x & 31u);
    const A4 va    = fill_a_i8(A, lane);
    const A4 vb    = fill_b_i8(B, lane);
    C8I vc;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        vc[j] = 0;
    }
    vc = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(1, va, 1, vb, vc, 0);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        raw_a[lane * 4 + j]     = va[j];
        raw_b[lane * 4 + j]     = vb[j];
        raw_c[lane * 8 + j]     = static_cast<u32>(vc[j]);
        raw_c[lane * 8 + 4 + j] = static_cast<u32>(vc[j + 4]);
    }
    store_c_i32(vc, lane, C);
}

std::vector<float> naive_bf16(const std::vector<unsigned short>& A,
                              const std::vector<unsigned short>& B) {
    std::vector<float> C(kDim * kDim, 0.0f);
    for (int r = 0; r < kDim; ++r) {
        for (int n = 0; n < kDim; ++n) {
            float acc = 0.0f;
            for (int k = 0; k < kDim; ++k) {
                acc += bf16_to_f32(A[r * kDim + k]) * bf16_to_f32(B[k * kDim + n]);
            }
            C[r * kDim + n] = acc;
        }
    }
    return C;
}

std::vector<int> naive_i8(const std::vector<signed char>& A, const std::vector<signed char>& B) {
    std::vector<int> C(kDim * kDim, 0);
    for (int r = 0; r < kDim; ++r) {
        for (int n = 0; n < kDim; ++n) {
            int acc = 0;
            for (int k = 0; k < kDim; ++k) {
                acc += static_cast<int>(A[r * kDim + k]) * static_cast<int>(B[k * kDim + n]);
            }
            C[r * kDim + n] = acc;
        }
    }
    return C;
}

void dump_raw(const char* title, const std::vector<u32>& raw, int regs_per_lane) {
    std::printf("%s\n", title);
    for (int lane = 0; lane < kLanes; ++lane) {
        std::printf("  lane %2d:", lane);
        for (int j = 0; j < regs_per_lane; ++j) {
            std::printf(" %08x", raw[lane * regs_per_lane + j]);
        }
        std::printf("\n");
    }
}

void print_matrix(const char* title, const std::vector<float>& m) {
    std::printf("%s\n", title);
    for (int r = 0; r < kDim; ++r) {
        std::printf("  ");
        for (int c = 0; c < kDim; ++c) {
            std::printf("%6.0f", m[r * kDim + c]);
        }
        std::printf("\n");
    }
}

// One bf16 run: fill A/B from host data, launch, return C plus the raw operand registers.
struct Bf16Run {
    std::vector<float> c;
    std::vector<u32> raw_a, raw_b, raw_c;
};

Bf16Run run_bf16(const std::vector<unsigned short>& A, const std::vector<unsigned short>& B) {
    unsigned short* dA  = nullptr;
    unsigned short* dB  = nullptr;
    float* dC           = nullptr;
    u32 *rA = nullptr, *rB = nullptr, *rC = nullptr;
    WMMA_CHECK(cudaMalloc(&dA, A.size() * sizeof(unsigned short)));
    WMMA_CHECK(cudaMalloc(&dB, B.size() * sizeof(unsigned short)));
    WMMA_CHECK(cudaMalloc(&dC, kDim * kDim * sizeof(float)));
    WMMA_CHECK(cudaMalloc(&rA, kLanes * 8 * sizeof(u32)));
    WMMA_CHECK(cudaMalloc(&rB, kLanes * 8 * sizeof(u32)));
    WMMA_CHECK(cudaMalloc(&rC, kLanes * 8 * sizeof(u32)));
    WMMA_CHECK(cudaMemcpy(dA, A.data(), A.size() * sizeof(unsigned short),
                          cudaMemcpyHostToDevice));
    WMMA_CHECK(cudaMemcpy(dB, B.data(), B.size() * sizeof(unsigned short),
                          cudaMemcpyHostToDevice));
    wmma_bf16_f32_kernel<<<1, kLanes>>>(dA, dB, dC, rA, rB, rC);
    WMMA_CHECK(cudaGetLastError());
    WMMA_CHECK(cudaStreamSynchronize(nullptr));

    Bf16Run out;
    out.c.resize(kDim * kDim);
    out.raw_a.resize(kLanes * 8);
    out.raw_b.resize(kLanes * 8);
    out.raw_c.resize(kLanes * 8);
    WMMA_CHECK(cudaMemcpy(out.c.data(), dC, out.c.size() * sizeof(float), cudaMemcpyDeviceToHost));
    WMMA_CHECK(cudaMemcpy(out.raw_a.data(), rA, out.raw_a.size() * sizeof(u32),
                          cudaMemcpyDeviceToHost));
    WMMA_CHECK(cudaMemcpy(out.raw_b.data(), rB, out.raw_b.size() * sizeof(u32),
                          cudaMemcpyDeviceToHost));
    WMMA_CHECK(cudaMemcpy(out.raw_c.data(), rC, out.raw_c.size() * sizeof(u32),
                          cudaMemcpyDeviceToHost));
    (void)cudaFree(dA);
    (void)cudaFree(dB);
    (void)cudaFree(dC);
    (void)cudaFree(rA);
    (void)cudaFree(rB);
    (void)cudaFree(rC);
    return out;
}

// C should equal `want` cell by cell; fingerprints are exact integers.
long compare(const std::vector<float>& got, const std::vector<float>& want) {
    long mism = 0;
    for (std::size_t i = 0; i < want.size(); ++i) {
        if (std::fabs(got[i] - want[i]) > 1e-3f) {
            ++mism;
        }
    }
    return mism;
}

std::vector<unsigned short> fingerprint_bf16(int salt) {
    std::vector<unsigned short> M(kDim * kDim);
    for (int r = 0; r < kDim; ++r) {
        for (int c = 0; c < kDim; ++c) {
            M[r * kDim + c] = bf16_of(static_cast<float>(16 * r + c + salt));
        }
    }
    return M;
}

}  // namespace

int main() {
    cudaDeviceProp prop{};
    WMMA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::printf("=== RDNA3 v_wmma layout probe on %s (%d MPs) ===\n", prop.name,
                prop.multiProcessorCount);
    std::printf("hypotheses: A col (col=lane&15) pairs (A[2j][col],A[2j+1][col]) | "
                "B row (row=lane&15) | C row=lane&15, col=2j+(lane>>4)\n");
    std::printf("            (C is the gfx1100 measurement that replaced the "
                "col base=(lane>>4)*8 guess; a FAIL means that decode is wrong too)\n\n");

    std::vector<unsigned short> I(kDim * kDim);
    for (int r = 0; r < kDim; ++r) {
        for (int c = 0; c < kDim; ++c) {
            I[r * kDim + c] = bf16_of(r == c ? 1.0f : 0.0f);
        }
    }
    const std::vector<unsigned short> Fp = fingerprint_bf16(0);
    std::vector<float> Fpw(Fp.size());
    for (std::size_t i = 0; i < Fp.size(); ++i) {
        Fpw[i] = bf16_to_f32(Fp[i]);
    }

    // A. identity probes
    {
        const Bf16Run a_is_i = run_bf16(I, Fp);
        std::printf("[bf16] A=I, C must equal B (fingerprint 16*r+c): mismatches %ld/256 %s\n",
                    compare(a_is_i.c, Fpw), compare(a_is_i.c, Fpw) == 0 ? "OK" : "SEE DUMPS");
        const Bf16Run b_is_i = run_bf16(Fp, I);
        std::printf("[bf16] B=I, C must equal A (fingerprint 16*r+c): mismatches %ld/256 %s\n",
                    compare(b_is_i.c, Fpw), compare(b_is_i.c, Fpw) == 0 ? "OK" : "SEE DUMPS");

        dump_raw("\n[bf16] raw A operand regs, per lane (8 x u32, low half = first element)",
                 a_is_i.raw_a, 8);
        dump_raw("[bf16] raw B operand regs, per lane", a_is_i.raw_b, 8);
        dump_raw("[bf16] raw C regs after A=I run (should read back B's fingerprint)", a_is_i.raw_c,
                 8);
        print_matrix("[bf16] C reconstructed under the hypotheses (A=I run)", a_is_i.c);
    }

    // C. numeric cross-check on data
    {
        std::vector<unsigned short> A(kDim * kDim), B(kDim * kDim);
        for (int r = 0; r < kDim; ++r) {
            for (int c = 0; c < kDim; ++c) {
                A[r * kDim + c] = bf16_of(static_cast<float>(operand(1, r, c)));
                B[r * kDim + c] = bf16_of(static_cast<float>(operand(7, r, c)));
            }
        }
        const Bf16Run run             = run_bf16(A, B);
        // Measured on gfx1100: the builtin forms (second argument) x (first argument), not the CUDA
        // wmma::mma_sync(d, a, b) order. A and B here are the probe's own labels, so the reference
        // is B*A. A kernel wanting M*N must therefore pass N through fill_a and M through fill_b.
        const std::vector<float> ref = naive_bf16(B, A);
        const long mism             = compare(run.c, ref);
        double worst                = 0.0;
        for (std::size_t i = 0; i < ref.size(); ++i) {
            worst = std::fmax(worst, std::fabs(static_cast<double>(run.c[i]) - ref[i]));
        }
        std::printf("\n[bf16->f32] data cross-check vs naive FP32: mismatches %ld/256 max|err|=%g\n",
                    mism, worst);
        // An identity matrix is its own transpose, so both fingerprint tests above are blind to an
        // axis-role error: they pin placement, not which operand axis feeds the reduction. Asymmetric
        // data is not blind, so name the candidate that actually matches instead of leaving "FAIL".
        if (mism != 0) {
            std::vector<unsigned short> At(kDim * kDim), Bt(kDim * kDim);
            for (int r = 0; r < kDim; ++r) {
                for (int c = 0; c < kDim; ++c) {
                    At[c * kDim + r] = A[r * kDim + c];
                    Bt[c * kDim + r] = B[r * kDim + c];
                }
            }
            struct { const char* name; std::vector<float> ref; } cands[] = {
                {"A*B        ", naive_bf16(A, B)},
                {"A^T*B      ", naive_bf16(At, B)},
                {"A*B^T      ", naive_bf16(A, Bt)},
                {"A^T*B^T    ", naive_bf16(At, Bt)},
                {"B*A        ", naive_bf16(B, A)},
                {"(A*B)^T    ", naive_bf16(Bt, At)},
                {"(A^T*B)^T  ", naive_bf16(Bt, A)},
                {"(A*B^T)^T  ", naive_bf16(B, At)},
            };
            for (const auto& cand : cands) {
                std::printf("            vs %s : mismatches %ld\n", cand.name,
                            compare(run.c, cand.ref));
            }
        }
        std::printf("            -> %s\n",
                    mism == 0
                        ? "PASS: fill_a_bf16/fill_b_bf16/store_c_f32 are the kernel adapter"
                        : "FAIL: layout hypothesis wrong; see the candidate list above");
    }

    // int8 -> i32
    {
        std::vector<signed char> A(kDim * kDim), B(kDim * kDim);
        for (int r = 0; r < kDim; ++r) {
            for (int c = 0; c < kDim; ++c) {
                A[r * kDim + c] = static_cast<signed char>(operand(3, r, c));
                B[r * kDim + c] = static_cast<signed char>(operand(9, r, c));
            }
        }
        signed char *dA = nullptr, *dB = nullptr;
        int* dC                     = nullptr;
        u32 *rA = nullptr, *rB = nullptr, *rC = nullptr;
        WMMA_CHECK(cudaMalloc(&dA, A.size()));
        WMMA_CHECK(cudaMalloc(&dB, B.size()));
        WMMA_CHECK(cudaMalloc(&dC, kDim * kDim * sizeof(int)));
        WMMA_CHECK(cudaMalloc(&rA, kLanes * 4 * sizeof(u32)));
        WMMA_CHECK(cudaMalloc(&rB, kLanes * 4 * sizeof(u32)));
        WMMA_CHECK(cudaMalloc(&rC, kLanes * 8 * sizeof(u32)));
        WMMA_CHECK(cudaMemcpy(dA, A.data(), A.size(), cudaMemcpyHostToDevice));
        WMMA_CHECK(cudaMemcpy(dB, B.data(), B.size(), cudaMemcpyHostToDevice));
        wmma_i8_kernel<<<1, kLanes>>>(dA, dB, dC, rA, rB, rC);
        WMMA_CHECK(cudaGetLastError());
        WMMA_CHECK(cudaStreamSynchronize(nullptr));
        std::vector<int> got(kDim * kDim);
        std::vector<u32> rawA(kLanes * 4), rawB(kLanes * 4), rawC(kLanes * 8);
        WMMA_CHECK(cudaMemcpy(got.data(), dC, got.size() * sizeof(int), cudaMemcpyDeviceToHost));
        WMMA_CHECK(cudaMemcpy(rawA.data(), rA, rawA.size() * sizeof(u32), cudaMemcpyDeviceToHost));
        WMMA_CHECK(cudaMemcpy(rawB.data(), rB, rawB.size() * sizeof(u32), cudaMemcpyDeviceToHost));
        WMMA_CHECK(cudaMemcpy(rawC.data(), rC, rawC.size() * sizeof(u32), cudaMemcpyDeviceToHost));
        const std::vector<int> ref = naive_i8(B, A);  // same operand-order reversal as bf16
        long mism = 0;
        for (std::size_t i = 0; i < ref.size(); ++i) {
            mism += (got[i] == ref[i]) ? 0 : 1;
        }
        std::printf("\n[int8->i32] data cross-check vs naive int: mismatches %ld/256 -> %s\n", mism,
                    mism == 0 ? "PASS" : "FAIL (raw operand dump below decodes the packing)");
        dump_raw("[int8] raw A operand regs (4 x u32 = 16 signed bytes, low byte first)", rawA, 4);
        dump_raw("[int8] raw B operand regs", rawB, 4);
        dump_raw("[int8] raw C operand regs (8 x i32)", rawC, 8);
        (void)cudaFree(dA);
        (void)cudaFree(dB);
        (void)cudaFree(dC);
        (void)cudaFree(rA);
        (void)cudaFree(rB);
        (void)cudaFree(rC);
    }

    std::printf("\n=== done: paste all of the above back ===\n");
    return 0;
}

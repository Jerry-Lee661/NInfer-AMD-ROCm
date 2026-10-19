// T5 片 1(2026-10-16): MLP 持久链核的发射端(唯一包含链核的设备 TU)。
//
// 段体来源(逐字复制迁移, 只改"谁在什么网格上跑", 不改任何算术; 原件的任何修改必须镜像到
// 本文件, 否则 md5 门会以输出漂移报警):
//   s1 norm    : ops/kernel/rmsnorm.cuh rmsnorm_cta_bf16x2_kernel<Offset,512,8>
//                (链内按逻辑 512 线程双份映射到物理 256 线程: 每物理线程兼算逻辑 t 与 t+256,
//                 归约树逐级同序 block_reduce_sum<512> => 位等价)
//   s2/s5 rotq : ops/linear/ternary/ternary_rotation_kernels.cuh ternary_rotate_quant_q8_v3_kernel
//   s3/s6 gemv : ops/linear/ternary/ternary_rowsplit_gemv.cuh ternary_pq2_gemv_dp4a_kernel
//                (含 M2 刀二 residual epilogue; s6 与残差原地别名)
//   s4 silu    : ops/kernel/silu_and_mul.cuh silu_mul_pair(经 ops/common/math.cuh silu())
//   屏障       : atomicInc 自复位(末位到齐回 0, 到齐者自旋等 0; 无清零节点, 图重放安全)
//   蝶形       : ops/kv_cache/hadamard_d256.cuh(公共头, 非手抄)
//
// 数值契约 = 与原 6 节点路径逐字节一致。仲裁探针(scripts/probe_mlp_chain.cu)三闸全绿:
// REF↔CHAIN 8/8 缓冲、屏障自复位、驻留改造体↔引擎原版体 4/4。
// 驻留画像(LDS 主约束): ex 8192B + lds_t(bf16 位存)4224B + norm 68B = 12484B/CTA,
// 4 CTA/CU 实测共驻留上限 => gfx1100 实发网格 384(运行时按物理 CU 数夹取, G=480 实测
// 驱动级 launch failure, 上限就是 384)。

#include "ops/linear/ternary/ternary_mlp_chain.h"

#include "core/device.h"
#include "ops/common/math.cuh"
#include "ops/common/warp.cuh"
#include "ops/kv_cache/hadamard_d256.cuh"
#include "ops/linear/ternary/ternary_rotation.h"
#include "ops/linear/ternary/ternary_row_view.h"
#include "ops/linear/ternary/ternary_s8_scratch.h"
// R1 刀(2026-10-17): [recurrent 批更新 + 统一尾链]。recurrent 段体直接复用引擎这组 device
// 函数(load_state_tile / run_recurrent_sequence<true,OutputEffects> / store_state_tile +
// BatchUpdateAccess), 保证 T=1 体逐字同源。
#include "ops/linear_attention/gated_delta_net/recurrent.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <string>

namespace ninfer::ops::detail {
namespace {

constexpr int kChainHidden    = 5120;
constexpr int kChainGuRows    = 34816;
constexpr int kChainAct       = 17408;
constexpr int kChainDnRows    = 5120;
constexpr int kChainGuGroups  = kChainHidden / 128; // 40
constexpr int kChainDnGroups  = kChainAct / 128;    // 136
constexpr int kChainNormBlock = 512;                // 引擎 norm 的逻辑块(= 半宽 256 x 2)
constexpr int kChainThreads   = 256;
constexpr int kChainGridMax   = 384; // gfx1100(96 CU x 4 CTA/CU)实测拍定上限; 实发网格见 chain_grid()
constexpr int kCodeBytesPerGroup  = 32;
constexpr int kScaleBytesPerGroup = 2;
constexpr int kGroupK             = 128;

// T5 片 2a(前置链 o_proj): attn = sigmoid_mul + rotq(6144) + o_proj GEMV+res;
// gdn = gated_rmsnorm + rotq(6144) + o_proj GEMV+res。两型层 o_proj 形状相同(k=6144, n=5120)。
constexpr int kOutProjRows     = 5120;
constexpr int kOutProjK        = 6144;
constexpr int kOutProjGroups   = kOutProjK / 128; // 48
constexpr int kGrRows          = 48;              // gated_rmsnorm 行数 = gdn_v_heads
constexpr int kGrD             = 128;             // gated_rmsnorm 行宽 = gdn_v_dim
constexpr int kOutProjNBlk     = kOutProjK / 1024; // 6 (rotq v3 的 signs 行数)
constexpr int kOutProjRotqGrid = (kOutProjNBlk + 1) / 2; // 3

// T5 片 3(输入投影链): 每层 X 侧输入投影, 三支链路(k 同为 5120)——
//   attn   : [input_norm + rotq(k=5120) + 四段 qkv GEMV] 3 节点 → 1(ternary_input_chain_attn);
//   gdn 刀一: [input_norm + 闸门 GEMV] 2 节点 → 1(ternary_norm_gating_chain);
//   gdn 刀二: [rotq(k=5120) + 三段 qk|value|z GEMV] 2 节点 → 1(ternary_segn_chain_split)。
constexpr int kInProjK        = kChainHidden;            // 5120
constexpr int kInProjGroups   = kInProjK / 128;          // 40
constexpr int kInProjNBlk     = kInProjK / 1024;         // 5 (rotq v3 的 signs 行数)
constexpr int kInProjRotqGrid = (kInProjNBlk + 1) / 2;   // 3
constexpr int kGatingN        = 48;                      // 闸门行数 = gdn_v_heads
constexpr int kGatingRows     = 2 * kGatingN;            // 96 (a 行 48 + b 行 48)
constexpr int kGatingMinGrid  = kGatingRows;             // gdn 链网格门槛 = 一 CTA 一行
// 输入投影输出面(qwen3_6 27B 几何; 链核只在这些形状上准入, 不匹配即回落原路径)。
constexpr int kAttnQRows      = 6144;                    // q / gate 行数
constexpr int kAttnKvRows     = 1024;                    // k / v 行数

// 门: NINFER_INPUT_CHAIN=0 回退(默认开; 按值解析——只有 "0" 关)。
// 翻转依据(2026-10-16 轮尾门全绿): 四臂 md5 逐字节(fb04ff85) + 节点 522→394(−128, 与
// 三链预测逐数字吻合) + 同窗墙钟 57.85→60.27(+4.2%); d2 两臂 md5/验收/节点全同(链 T=1
// 专用, verify T=3 不触碰=设计自证); PPL 两臂 1.756730/5.793462 与 M3-0 基线逐位同;
// serve 同窗 32.35→33.90(+4.8%, 10/10 请求无丢流)。证据 = bench/evidence/t5-slice3-*。
bool ternary_input_chain_enabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("NINFER_INPUT_CHAIN");
        return value == nullptr || std::string(value) != "0";
    }();
    return enabled;
}

// 门: NINFER_SEGN_CONV_CHAIN=0 回退(默认开; 翻转依据 = md5 门两臂逐字节同 + nospec +1.1%
// + d2/PPL 中性, 见 bench/evidence/s2a-conv-chain-2026-10-17.md)。
bool ternary_segn_conv_chain_enabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("NINFER_SEGN_CONV_CHAIN");
        return value == nullptr || std::string(value) != "0";
    }();
    return enabled;
}

// 门: NINFER_OUTPROJ_CHAIN=0 回退(默认开; 翻转依据 = 同片 1 三件套纪律, 见本轮证据)。
bool ternary_out_proj_chain_enabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("NINFER_OUTPROJ_CHAIN");
        return value == nullptr || std::string(value) != "0";
    }();
    return enabled;
}

// 门: NINFER_MLP_CHAIN=0 回退(默认开; 读一次, 图捕获语义安全。翻转依据 = 三件套全绿:
// md5 逐字节 + 节点 970→650 + 同窗 +9.8%, 见 bench/evidence/t5-slice1-wiring-2026-10-16.md)。
bool ternary_mlp_chain_enabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("NINFER_MLP_CHAIN");
        return value == nullptr || std::string(value) != "0";
    }();
    return enabled;
}

// 门: NINFER_TAIL_CHAIN=0 回退(默认开; 按值解析——只有 "0" 关)。
// 片 2b: [o_proj 前置段 + MLP 段] 统一尾链(2→1)。另要求前置链与 MLP 链两个门都开
// (统一核复用两者的段体与准入)。翻转依据(2026-10-16 轮尾门全绿): 六臂 md5 逐字节
// (fb04ff85) + 冒烟 f479ad68; 节点 394→330(−64, 与预测逐数字吻合); 同窗墙钟
// 58.67→59.04(+0.63%); d2 两臂 md5/验收/节点全同(链 T=1 专用, verify T=3 不触碰=设计
// 自证); PPL 两臂 1.756730/5.793462 与 M3-0 基线逐位同; serve 同窗 31.96→32.11
// (服务端 34.13→34.31, +0.5%, 10/10 无丢流)。证据 = bench/evidence/t5-slice2b-*。
bool ternary_tail_chain_enabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("NINFER_TAIL_CHAIN");
        return value == nullptr || std::string(value) != "0";
    }();
    return enabled;
}

// 门: NINFER_REC_TAIL_CHAIN=0 回退(默认关; 翻转前置 = md5 门 + nospec A/B 绿, 见本轮证据)。
// R1 刀(2026-10-17 剩余节点组): [GDN recurrent 批更新 + 统一尾链] 2 节点并 1, 接 T=1 单列
// 快照支的 batch_update 节点(update-in-place, width=1/batch=1)。
bool ternary_rec_tail_chain_enabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("NINFER_REC_TAIL_CHAIN");
        return value != nullptr && std::string(value) != "0";
    }();
    return enabled;
}

// 网格 = min(384, 4 x 物理 CU 数)。驻留上限 = 4 CTA/CU(由 lds_t bf16 位存后的 LDS
// 12484B/CTA 决定) x CU 数; 384 = gfx1100(96 CU)实测拍定值。其它卡按同一比例收窄,
// 保证共驻留(超限 = 挂死)。物理 CU 数查询失败(非预期)则拒绝链核, 回落原路径(fail-safe)。
unsigned chain_grid() {
    static const unsigned grid = []() -> unsigned {
        int cu  = 0;
        int dev = 0;
        if (hipGetDevice(&dev) != hipSuccess ||
            hipDeviceGetAttribute(&cu, hipDeviceAttributePhysicalMultiProcessorCount, dev) !=
                hipSuccess ||
            cu <= 0 || cu > 256) {
            return 0;
        }
        const unsigned g = static_cast<unsigned>(cu) * 4u;
        return g < kChainGridMax ? g : kChainGridMax;
    }();
    return grid;
}

// 链核的屏障计数槽: MLP 链 5 道段间屏障用 [0..4], 前置链(o_proj)2 道用 [5..6],
// S2a 三合链(segn+conv)2 道用 [7..8], 统一尾链内部 1 道用 [9], R1 刀(recurrent→前置段)
// 1 道用 [10]。设备全局由装载器零初始化; 每次链跑末位到齐自复位回 0, 因此不需要任何
// 清零节点, 图回放天然安全。约束(同 QHOIST / T4 arm 契约): 单会话静态 —— 同一设备上
// 并发回放两张含链核的图(多引擎/多流)会共享这组槽, 属未支持用法。
__device__ unsigned int g_ternary_mlp_chain_counters[11];

__device__ __forceinline__ void chain_barrier(unsigned int* counter) {
    __syncthreads();
    __threadfence();
    if (threadIdx.x == 0) {
        atomicInc(counter, static_cast<unsigned>(gridDim.x) - 1u); // 到齐自复位(末位回 0)
        while (atomicAdd(counter, 0u) != 0u) {}
    }
    __syncthreads();
    __threadfence();
}

// rmsnorm.cuh rmsnorm_epilogue<RmsEpilogue::Offset> 逐字(unit-offset 端标):
// weight += 1; value = x * inv * weight。
__device__ __forceinline__ float chain_norm_epilogue_offset(float x, float inv, float weight) {
    weight += 1.0f;
    return x * inv * weight;
}

// s1 链内版: 引擎 rmsnorm_cta_bf16x2_kernel<Offset,512,8> 的逻辑 512 线程映射到物理
// 256 线程(每线程兼算逻辑 t 与 t+256), 归约树逐级同序 => 位等价。
__device__ void chain_norm_stage(const __nv_bfloat162* x, const __nv_bfloat162* weight,
                                 __nv_bfloat162* out, std::int32_t d, float eps) {
    constexpr int PairsPerThread = 5; // pairs = 2560 / 512
    const int t                   = static_cast<int>(threadIdx.x);
    const int lane                = t & 31;
    const int warp                = t >> 5;

    __nv_bfloat162 va[PairsPerThread];
    __nv_bfloat162 vb[PairsPerThread];
    float sumA = 0.0f;
    float sumB = 0.0f;
#pragma unroll
    for (int k = 0; k < PairsPerThread; ++k) {
        {
            const int pair  = t + k * kChainNormBlock;
            va[k]           = x[pair];
            const float2 xf = __bfloat1622float2(va[k]);
            sumA += xf.x * xf.x + xf.y * xf.y;
        }
        {
            const int pair  = (t + 256) + k * kChainNormBlock;
            vb[k]           = x[pair];
            const float2 xf = __bfloat1622float2(vb[k]);
            sumB += xf.x * xf.x + xf.y * xf.y;
        }
    }
    sumA = warp_reduce_sum(sumA);
    sumB = warp_reduce_sum(sumB);
    __shared__ float sums[16];
    if (lane == 0) {
        sums[warp]     = sumA;
        sums[warp + 8] = sumB;
    }
    __syncthreads();
    float s = threadIdx.x < 16 ? sums[lane] : 0.0f;
    if (warp == 0) { s = warp_reduce_sum<16>(s); }
    __shared__ float inv_shared;
    if (threadIdx.x == 0) { inv_shared = rsqrtf(s / static_cast<float>(d) + eps); }
    __syncthreads();
    const float inv = inv_shared;
#pragma unroll
    for (int k = 0; k < PairsPerThread; ++k) {
        {
            const int pair  = t + k * kChainNormBlock;
            const float2 xf = __bfloat1622float2(va[k]);
            const float2 wf = __bfloat1622float2(weight[pair]);
            out[pair]       = __floats2bfloat162_rn(chain_norm_epilogue_offset(xf.x, inv, wf.x),
                                                    chain_norm_epilogue_offset(xf.y, inv, wf.y));
        }
        {
            const int pair  = (t + 256) + k * kChainNormBlock;
            const float2 xf = __bfloat1622float2(vb[k]);
            const float2 wf = __bfloat1622float2(weight[pair]);
            out[pair]       = __floats2bfloat162_rn(chain_norm_epilogue_offset(xf.x, inv, wf.x),
                                                    chain_norm_epilogue_offset(xf.y, inv, wf.y));
        }
    }
}

// s2/s5: rotquant v3 —— ternary_rotate_quant_q8_v3_kernel 逐字(含 P 置换分支; 准入要求
// perm_rep<=1, 分支不被走)。
__device__ void chain_rotq_v3_stage(const __nv_bfloat16* __restrict__ x,
                                    std::int8_t* __restrict__ xq, std::uint16_t* __restrict__ ds8,
                                    const float* __restrict__ signs, int n_blk, int k, int tokens,
                                    int perm_hd, int perm_nk, int perm_rep) {
    constexpr int kBlockSize = 1024;
    const int lane           = threadIdx.x & 31;
    const int warp_q         = (threadIdx.x >> 5) & 3;
    const int blk_lcl        = (threadIdx.x >> 5) >> 2;
    const int blocks         = k >> 10;
    const int block          = static_cast<int>(blockIdx.x) * 2 + blk_lcl;
    const int token          = static_cast<int>(blockIdx.y);
    const bool active        = block < blocks && token < tokens;

    const float* const signs_row = signs + (block % n_blk) * kBlockSize;
    const bool permuted          = perm_rep > 1;
    const __nv_bfloat16* x_token = x + static_cast<std::int64_t>(token) * k;
    std::int8_t* const xq_token  = xq + static_cast<std::int64_t>(token) * k;
    std::uint16_t* const ds_token = ds8 + static_cast<std::int64_t>(token) * (k >> 5);
    const int qbase              = warp_q * 256;

    __shared__ float ex[8][256];      // stage-2 交接:[warp][切片内维]
    // lds_t 存的是"已强制 bf16 舍入"的值(见下), 按 bf16 位存为 uint16 数值恒等;
    // smem 16640B -> 12484B 是 4 CTA/CU 共驻留(LDS 主约束)的关键, 原版核不受影响。
    __shared__ std::uint16_t lds_t[8][8][33]; // absmax 转置:[warp][组内 r][lane](pad 33)

    float values[8];
    if (active) {
#pragma unroll
        for (int r = 0; r < 8; ++r) {
            const int column = (block << 10) + qbase + r * 32 + lane;
            int source       = column;
            if (permuted) {
                const int hd = column % perm_hd;
                const int q  = column / perm_hd;
                source = hd + perm_hd * (q / perm_rep) + perm_hd * perm_nk * (q % perm_rep);
            }
            values[r] = __fmul_rn(__bfloat162float(x_token[source]),
                                  signs_row[qbase + r * 32 + lane]);
        }
        normalized_hadamard_d256_inplace(values, lane);
#pragma unroll
        for (int r = 0; r < 8; ++r) { ex[threadIdx.x >> 5][lane + 32 * r] = values[r]; }
    }
    __syncthreads();
    // stage bit8: 切片对 (0,2)/(1,3); stage bit9: (0,1)/(2,3)。
    // 读伙伴槽与写回自己槽之间必须整块同步(T3 实锤: 无屏障时 v3a-vs-v3b 差 1397 码);
    // 屏障放在 active 门之外(分歧屏障 = UB)。
    if (active) {
        const float* own     = ex[threadIdx.x >> 5];
        const float* partner = ex[(threadIdx.x >> 5) ^ 2];
        const int pb         = (warp_q & 2) ? 1 : 0;
#pragma unroll
        for (int r = 0; r < 8; ++r) {
            values[r] = pb ? __fsub_rn(partner[lane + 32 * r], own[lane + 32 * r])
                           : __fadd_rn(own[lane + 32 * r], partner[lane + 32 * r]);
        }
    }
    __syncthreads();
    if (active) {
        float* u = ex[threadIdx.x >> 5];
#pragma unroll
        for (int r = 0; r < 8; ++r) { u[lane + 32 * r] = values[r]; }
    }
    __syncthreads();
    if (active) {
        const float* own     = ex[threadIdx.x >> 5];
        const float* partner = ex[(threadIdx.x >> 5) ^ 1];
        const int pb         = (warp_q & 1) ? 1 : 0;
#pragma unroll
        for (int r = 0; r < 8; ++r) {
            values[r] = pb ? __fsub_rn(partner[lane + 32 * r], own[lane + 32 * r])
                           : __fadd_rn(own[lane + 32 * r], partner[lane + 32 * r]);
        }
        // D1024 总归一 = 2^-5(D256 自带 2^-4, 跨 warp 2 个 span 阶段再 ×2^-1)。
#pragma unroll
        for (int r = 0; r < 8; ++r) { values[r] = __fmul_rn(values[r], 0x1p-1f); }
        // 两核路径舍入点: 全部 Hadamard 阶段结束才落 bf16(先舍再跨 warp = 320 scale 全错)。
#pragma unroll
        for (int r = 0; r < 8; ++r) { values[r] = __bfloat162float(__float2bfloat16_rn(values[r])); }
        std::uint16_t* lw = &lds_t[threadIdx.x >> 5][0][0];
#pragma unroll
        for (int r = 0; r < 8; ++r) {
            lw[r * 33 + lane] = static_cast<std::uint16_t>(__float_as_uint(values[r]) >> 16);
        }
        __syncwarp();
        if (lane < 8) {
            const int g = lane;
            float amax  = 0.0f;
#pragma unroll
            for (int i = 0; i < 32; ++i) {
                amax = fmaxf(amax, fabsf(__uint_as_float(
                                       static_cast<std::uint32_t>(lw[g * 33 + i]) << 16)));
            }
            const float d   = amax / 127.0f;
            const float inv = d > 0.0f ? 1.0f / d : 0.0f;
            ds_token[(block << 5) + warp_q * 8 + g] = __half_as_ushort(__float2half(d));
            // unroll 4: 与全展开逐元素等价(每元素独立), VGPR 116->37 是共驻留的余量;
            // 引擎原版体的逐字节等价由探针第三闸(改造体 vs 原版体 4/4)实证。
#pragma unroll 4
            for (int i = 0; i < 32; ++i) {
                const float v =
                    __uint_as_float(static_cast<std::uint32_t>(lw[g * 33 + i]) << 16);
                const int qv = std::lrint(v * inv);
                xq_token[(block << 10) + (warp_q * 8 + g) * 32 + i] =
                    static_cast<std::int8_t>(std::max(-127, std::min(127, qv)));
            }
        }
    }
}

// ---- ternary_rowsplit_gemv.cuh 的 dp4a 帮助件(逐字; 该头是单 TU 头, 不能重复包含) ----
__device__ __forceinline__ int chain_pq2_dp4a_symbols4(unsigned int b) {
    const unsigned int y = (b & 0x33u) | ((b & 0xCCu) << 6);
    const unsigned int z = (y & 0x0303u) | ((y & 0x3030u) << 12);
    return static_cast<int>(__builtin_amdgcn_perm(0x020100FFu, 0x020100FFu, z));
}
__device__ __forceinline__ int chain_pq2_dp4a_dot4(int a, int b, int c) {
    return __builtin_amdgcn_sudot4(true, a, true, b, c, false);
}
__device__ __forceinline__ float chain_gemv_scale(const std::uint8_t* scale_ptr) {
    const std::uint16_t bits = *reinterpret_cast<const std::uint16_t*>(scale_ptr);
    return __half2float(__ushort_as_half(bits));
}

// s3/s6: GEMV 单行体 —— ternary_pq2_gemv_dp4a_kernel 逐字(含 residual epilogue; residual
// 不带 __restrict__ 且读在写前, 原地别名安全)。
__device__ __forceinline__ void chain_gemv_row(int lane, int row,
                                               const std::int8_t* __restrict__ xq,
                                               const std::uint16_t* __restrict__ ds8,
                                               const std::uint8_t* __restrict__ codes,
                                               const std::uint8_t* __restrict__ scales,
                                               __nv_bfloat16* __restrict__ out, int groups_per_row,
                                               const __nv_bfloat16* residual) {
    const std::uint8_t* code_row =
        codes + static_cast<std::int64_t>(row) * groups_per_row * kCodeBytesPerGroup;
    const std::uint8_t* scale_row =
        scales + static_cast<std::int64_t>(row) * groups_per_row * kScaleBytesPerGroup;
    float residual_value = 0.0f;
    if (residual != nullptr && lane == 0) { residual_value = __bfloat162float(residual[row]); }
    float accumulator = 0.0f;
#pragma unroll 8
    for (int group = 0; group < groups_per_row; ++group) {
        const std::uint8_t raw = code_row[group * kCodeBytesPerGroup + lane];
        const int w4           = chain_pq2_dp4a_symbols4(raw);
        const int x4           = *reinterpret_cast<const int*>(
            xq + static_cast<std::int64_t>(group) * kGroupK + lane * 4);
        const int dot = chain_pq2_dp4a_dot4(w4, x4, 0);
        const float wsc =
            chain_gemv_scale(scale_row + group * kScaleBytesPerGroup) *
            __half2float(__ushort_as_half(ds8[group * 4 + (lane >> 3)]));
        accumulator = fmaf(wsc, static_cast<float>(dot), accumulator);
    }
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        accumulator += __shfl_xor_sync(0xffffffffffffffffULL, accumulator, offset);
    }
    if (lane == 0) {
        const __nv_bfloat16 projected = __float2bfloat16_rn(accumulator);
        out[row] = residual == nullptr
                       ? projected
                       : __float2bfloat16_rn(residual_value + __bfloat162float(projected));
    }
}

// s4: silu 单对体(silu_and_mul.cuh silu_mul_pair 逐字, 用 ops/common/math.cuh 的 silu())。
__device__ __forceinline__ __nv_bfloat162 chain_silu_mul_pair(__nv_bfloat162 g,
                                                              __nv_bfloat162 u) {
    const float r0 = silu(__low2float(g)) * __low2float(u);
    const float r1 = silu(__high2float(g)) * __high2float(u);
    return __floats2bfloat162_rn(r0, r1);
}

// ---- T5 片 2a 前置链的两个段体 ----

// sigmoid_gate_mul.cuh sigmoid_gate_mul_pair / bf16x2 体逐字(原地: x 既是输入也是输出)。
__device__ __forceinline__ __nv_bfloat162 chain_sigmoid_gate_mul_pair(__nv_bfloat162 gate,
                                                                      __nv_bfloat162 x) {
    const float r0 = __low2float(x) * sigmoid(__low2float(gate));
    const float r1 = __high2float(x) * sigmoid(__high2float(gate));
    return __floats2bfloat162_rn(r0, r1);
}

// 前置链 s1(attn 支): sigmoid_mul 原地。4 对/线程, 与引擎 bf16x2 核同映射(n2=3072 对)。
__device__ void chain_sigmoid_gate_mul_stage(const __nv_bfloat16* __restrict__ gate,
                                             __nv_bfloat16* __restrict__ x) {
    constexpr int kPairsPerThread = 4;
    const std::int64_t tid = blockIdx.x * static_cast<std::int64_t>(kChainThreads) + threadIdx.x;
    const std::int64_t stride =
        static_cast<std::int64_t>(gridDim.x) * kChainThreads * kPairsPerThread;
    const std::int64_t n2 = kOutProjK / 2;
    const auto* gate2     = reinterpret_cast<const __nv_bfloat162*>(gate);
    auto* x2              = reinterpret_cast<__nv_bfloat162*>(x);
    for (std::int64_t j = tid * kPairsPerThread; j < n2; j += stride) {
        const __nv_bfloat162 g0 = gate2[j];
        const __nv_bfloat162 x0 = x2[j];
        if (j + 3 < n2) {
            const __nv_bfloat162 g1  = gate2[j + 1];
            const __nv_bfloat162 x1  = x2[j + 1];
            const __nv_bfloat162 g2  = gate2[j + 2];
            const __nv_bfloat162 x2v = x2[j + 2];
            const __nv_bfloat162 g3  = gate2[j + 3];
            const __nv_bfloat162 x3  = x2[j + 3];
            x2[j]                    = chain_sigmoid_gate_mul_pair(g0, x0);
            x2[j + 1]                = chain_sigmoid_gate_mul_pair(g1, x1);
            x2[j + 2]                = chain_sigmoid_gate_mul_pair(g2, x2v);
            x2[j + 3]                = chain_sigmoid_gate_mul_pair(g3, x3);
        } else {
            x2[j] = chain_sigmoid_gate_mul_pair(g0, x0);
            if (j + 1 < n2) { x2[j + 1] = chain_sigmoid_gate_mul_pair(gate2[j + 1], x2[j + 1]); }
            if (j + 2 < n2) { x2[j + 2] = chain_sigmoid_gate_mul_pair(gate2[j + 2], x2[j + 2]); }
        }
    }
}

// 前置链 s1(gdn 支): gated_rmsnorm —— rmsnorm_warp_bf16x2_kernel<Gated, 512> 的行体逐字
// (一 warp 一行, d=128, kMaxPairsPerLane=4, Gated epilogue: value *= silu(z); unit_offset=false
// 即权重不加 1)。链内 warp 全局号直接作行号(rows=48 < 3072 空闲 warp 不参与)。
__device__ void chain_gated_rmsnorm_stage(const __nv_bfloat16* __restrict__ x,
                                          const __nv_bfloat16* __restrict__ weight,
                                          const __nv_bfloat16* __restrict__ z,
                                          __nv_bfloat16* __restrict__ out, float eps) {
    constexpr int kMaxPairsPerLane = 4;
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp = static_cast<int>(blockIdx.x) * 8 + (static_cast<int>(threadIdx.x) >> 5);
    if (warp >= kGrRows) { return; }
    const std::int64_t row      = warp;
    const int pairs             = kGrD / 2;
    const std::int64_t row_base = row * static_cast<std::int64_t>(pairs);
    const auto* x2              = reinterpret_cast<const __nv_bfloat162*>(x);
    const auto* w2              = reinterpret_cast<const __nv_bfloat162*>(weight);
    const auto* z2              = reinterpret_cast<const __nv_bfloat162*>(z);
    auto* out2                  = reinterpret_cast<__nv_bfloat162*>(out);
    __nv_bfloat162 values[kMaxPairsPerLane];
    float sum = 0.0f;
#pragma unroll
    for (int k = 0; k < kMaxPairsPerLane; ++k) {
        const int pair = lane + k * 32;
        if (pair < pairs) {
            values[k]       = x2[row_base + pair];
            const float2 xf = __bfloat1622float2(values[k]);
            sum += xf.x * xf.x + xf.y * xf.y;
        }
    }
    sum       = warp_reduce_sum(sum);
    float inv = lane == 0 ? rsqrtf(sum / static_cast<float>(kGrD) + eps) : 0.0f;
    inv       = __shfl_sync(0xffffffffffffffffULL, inv, 0);
#pragma unroll
    for (int k = 0; k < kMaxPairsPerLane; ++k) {
        const int pair = lane + k * 32;
        if (pair < pairs) {
            const float2 xf = __bfloat1622float2(values[k]);
            const float2 wf = __bfloat1622float2(w2[pair]);
            const float2 zf = __bfloat1622float2(z2[row_base + pair]);
            const float r0  = (xf.x * inv * wf.x) * silu(zf.x);
            const float r1  = (xf.y * inv * wf.y) * silu(zf.y);
            out2[row_base + pair] = __floats2bfloat162_rn(r0, r1);
        }
    }
}

// 前置段体(s1..s3; 片 2b 抽出为 device 函数, 供独立核与统一尾链核逐字复用):
// 激活闸门 -> 屏障[5] -> rotq v3(k=6144) -> 屏障[6] -> o_proj GEMV+残差。
// gdn_mode=0: gate/a; gdn_mode=1: gr_x/gr_w/gr_z -> act(独立输出缓冲)。
__device__ void chain_run_out_proj_stages(
    int gdn_mode, const __nv_bfloat16* __restrict__ gate, __nv_bfloat16* __restrict__ act,
    const __nv_bfloat16* __restrict__ gr_x, const __nv_bfloat16* __restrict__ gr_w,
    const __nv_bfloat16* __restrict__ gr_z, float eps,
    const std::uint8_t* __restrict__ codes, const std::uint8_t* __restrict__ scales,
    const float* __restrict__ signs, int perm_hd, int perm_nk, int perm_rep,
    __nv_bfloat16* __restrict__ x_inout, std::int8_t* __restrict__ xq,
    std::uint16_t* __restrict__ ds8) {
    if (gdn_mode) {
        chain_gated_rmsnorm_stage(gr_x, gr_w, gr_z, act, eps);
    } else {
        chain_sigmoid_gate_mul_stage(gate, act);
    }
    chain_barrier(&g_ternary_mlp_chain_counters[5]);

    if (blockIdx.x < kOutProjRotqGrid) {
        chain_rotq_v3_stage(act, xq, ds8, signs, kOutProjNBlk, kOutProjK, 1, perm_hd, perm_nk,
                            perm_rep);
    }
    chain_barrier(&g_ternary_mlp_chain_counters[6]);

    {
        const int lane        = static_cast<int>(threadIdx.x) & 31;
        const int warp0       = static_cast<int>(blockIdx.x) * 8 + (static_cast<int>(threadIdx.x) >> 5);
        const int total_warps = static_cast<int>(gridDim.x) * 8;
        for (int row = warp0; row < kOutProjRows; row += total_warps) {
            chain_gemv_row(lane, row, xq, ds8, codes, scales, x_inout, kOutProjGroups, x_inout);
        }
    }
}

__global__ __launch_bounds__(kChainThreads) void ternary_out_proj_chain_kernel(
    int gdn_mode, const __nv_bfloat16* __restrict__ gate, __nv_bfloat16* __restrict__ act,
    const __nv_bfloat16* __restrict__ gr_x, const __nv_bfloat16* __restrict__ gr_w,
    const __nv_bfloat16* __restrict__ gr_z, float eps,
    const std::uint8_t* __restrict__ codes, const std::uint8_t* __restrict__ scales,
    const float* __restrict__ signs, int perm_hd, int perm_nk, int perm_rep,
    __nv_bfloat16* __restrict__ x_inout, std::int8_t* __restrict__ xq,
    std::uint16_t* __restrict__ ds8) {
    chain_run_out_proj_stages(gdn_mode, gate, act, gr_x, gr_w, gr_z, eps, codes, scales, signs,
                              perm_hd, perm_nk, perm_rep, x_inout, xq, ds8);
}

// 前置声明: MLP 段体定义在文件后段(片 1 区域), 统一尾链核先引用。
__device__ void chain_run_mlp_stages(
    const __nv_bfloat16* __restrict__ x_in, const __nv_bfloat16* __restrict__ norm_w, float eps,
    const std::uint8_t* __restrict__ gu_codes, const std::uint8_t* __restrict__ gu_scales,
    const std::uint8_t* __restrict__ dn_codes, const std::uint8_t* __restrict__ dn_scales,
    const float* __restrict__ gu_signs, const float* __restrict__ dn_signs, int gu_perm_hd,
    int gu_perm_nk, int gu_perm_rep, int dn_perm_hd, int dn_perm_nk, int dn_perm_rep,
    __nv_bfloat16* __restrict__ x_inout, __nv_bfloat16* __restrict__ h,
    __nv_bfloat16* __restrict__ g, __nv_bfloat16* __restrict__ act, std::int8_t* __restrict__ xq1,
    std::uint16_t* __restrict__ ds81, std::int8_t* __restrict__ xq2,
    std::uint16_t* __restrict__ ds82);

// ---- T5 片 2b: 统一尾链核 —— [o_proj 前置段 + MLP 段] 一次发射 ----
// 站点序判据: mixer 站的 o_proj 链与 mlp_tail 站的 MLP 链在节点流里严格相邻
// (run_layers: mixer -> profile.mark -> mlp_tail, 中间零节点), 故可并成一个持久核。
// 形状: 前置段写 x_inout(残差 += o_proj), MLP 段的 s1 norm 再读同一 x_inout => 参数里
// x_inout 一件两用。屏障计数槽: 前置段用 [5]/[6], MLP 段用 [0..4](互不冲突, 各自核内一次)。
// LDS 预算: 两段的 __shared__ 都声明在共用 stage 函数里(chain_rotq_v3_stage 的 lds_t、
// chain_norm_stage 的 sums/inv、s4 silu 与 gemv 栈上), 同一函数的多处调用共享一份分配,
// 故统一核的 LDS 与 MLP 链核同量级(12484B/CTA, 4 CTA/CU 驻留不变); VGPR 风险由冒烟门暴露。
__global__ __launch_bounds__(kChainThreads) void ternary_tail_chain_kernel(
    // 前置段(片 2a 参数集)
    int gdn_mode, const __nv_bfloat16* __restrict__ gate, __nv_bfloat16* __restrict__ op_act,
    const __nv_bfloat16* __restrict__ gr_x, const __nv_bfloat16* __restrict__ gr_w,
    const __nv_bfloat16* __restrict__ gr_z, float op_eps,
    const std::uint8_t* __restrict__ op_codes, const std::uint8_t* __restrict__ op_scales,
    const float* __restrict__ op_signs, int op_perm_hd, int op_perm_nk, int op_perm_rep,
    std::int8_t* __restrict__ op_xq, std::uint16_t* __restrict__ op_ds8,
    // MLP 段(片 1 参数集; x_inout 与前置段的残差 x 同为一件)
    const __nv_bfloat16* __restrict__ x_in, const __nv_bfloat16* __restrict__ norm_w,
    float mlp_eps, const std::uint8_t* __restrict__ gu_codes,
    const std::uint8_t* __restrict__ gu_scales, const std::uint8_t* __restrict__ dn_codes,
    const std::uint8_t* __restrict__ dn_scales, const float* __restrict__ gu_signs,
    const float* __restrict__ dn_signs, int gu_perm_hd, int gu_perm_nk, int gu_perm_rep,
    int dn_perm_hd, int dn_perm_nk, int dn_perm_rep, __nv_bfloat16* __restrict__ x_inout,
    __nv_bfloat16* __restrict__ h, __nv_bfloat16* __restrict__ g,
    __nv_bfloat16* __restrict__ mlp_act, std::int8_t* __restrict__ xq1,
    std::uint16_t* __restrict__ ds81, std::int8_t* __restrict__ xq2,
    std::uint16_t* __restrict__ ds82) {
    chain_run_out_proj_stages(gdn_mode, gate, op_act, gr_x, gr_w, gr_z, op_eps, op_codes,
                              op_scales, op_signs, op_perm_hd, op_perm_nk, op_perm_rep, x_inout,
                              op_xq, op_ds8);
    // 前置段末段(o_proj GEMV)全网格写 x_inout, MLP 段 s1 由 CTA0 整读 x_inout(与位序无关的
    // 依赖): 段间必须补齐屏障(计数槽 [9], 两段各自的 [5]/[6] 与 [0..4] 各用一次)。
    chain_barrier(&g_ternary_mlp_chain_counters[9]);
    chain_run_mlp_stages(x_in, norm_w, mlp_eps, gu_codes, gu_scales, dn_codes, dn_scales, gu_signs,
                         dn_signs, gu_perm_hd, gu_perm_nk, gu_perm_rep, dn_perm_hd, dn_perm_nk,
                         dn_perm_rep, x_inout, h, g, mlp_act, xq1, ds81, xq2, ds82);
}

// ---- R1 刀(2026-10-17 剩余节点组): [GDN recurrent 批更新 + 统一尾链] 2 节点并 1 ----
// 接线面 = text_context_impl GDN Verify 支的 batch_update 节点(快照支 T=1/width=1/batch=1)。
// 段序 = [recurrent(T=1) → 屏障[10] → 前置段(gated_rmsnorm 读 o=recurrent 输出) → 屏障[9]
// → MLP 段]。recurrent 段 = recurrent_batch_update_kernel 的 T=1 体逐字复用(同一组 device
// 函数与 BatchUpdateAccess); 唯一改写 = 坐标构造: 引擎几何一个 workgroup =
// (threadIdx.x=lane, threadIdx.y=warp∈[0,4)) + (blockIdx.x=vhead, blockIdx.z=state_tile),
// 链核是 256 平铺线程 ⇒ 平铺重映射(wg = blockIdx.x*2 + tid>>7, lane = tid&31,
// warp = (tid>>5)&3, vhead = wg%H_v, tile = wg/H_v)。逐 WG 计算自洽(不跨 WG 共享任何
// 状态) ⇒ 重映射不改变任何一条算术。数值契约 = 原两节点路径逐字节(md5 门判)。
__global__ __launch_bounds__(kChainThreads) void ternary_tail_chain_recurrent_kernel(
    // recurrent 段(q/k/v = {128,H,1,1} 视图; o = recurrent 输出 = 前置段 gr_x)
    const __nv_bfloat16* __restrict__ rec_q, const __nv_bfloat16* __restrict__ rec_k,
    const __nv_bfloat16* __restrict__ rec_v, const float* __restrict__ rec_g,
    const float* __restrict__ rec_beta, const float* __restrict__ rec_states_read,
    float* __restrict__ rec_states_write, const std::int32_t* __restrict__ rec_src_slots,
    const std::int32_t* __restrict__ rec_dst_slots, __nv_bfloat16* __restrict__ rec_out,
    gated_delta_net::head_map rec_heads, std::int64_t rec_slot_stride, float rec_scale,
    int rec_workgroups,
    // 前置段(gdn 支固定: gate=nullptr; op_act=on; gr_x=o; gr_w/gr_z 常驻)
    __nv_bfloat16* __restrict__ op_act, const __nv_bfloat16* __restrict__ gr_w,
    const __nv_bfloat16* __restrict__ gr_z, float op_eps,
    const std::uint8_t* __restrict__ op_codes, const std::uint8_t* __restrict__ op_scales,
    const float* __restrict__ op_signs, int op_perm_hd, int op_perm_nk, int op_perm_rep,
    std::int8_t* __restrict__ op_xq, std::uint16_t* __restrict__ op_ds8,
    // MLP 段(片 1 参数集; x_inout 与前置段的残差 x 同为一件)
    const __nv_bfloat16* __restrict__ x_in, const __nv_bfloat16* __restrict__ norm_w,
    float mlp_eps, const std::uint8_t* __restrict__ gu_codes,
    const std::uint8_t* __restrict__ gu_scales, const std::uint8_t* __restrict__ dn_codes,
    const std::uint8_t* __restrict__ dn_scales, const float* __restrict__ gu_signs,
    const float* __restrict__ dn_signs, int gu_perm_hd, int gu_perm_nk, int gu_perm_rep,
    int dn_perm_hd, int dn_perm_nk, int dn_perm_rep, __nv_bfloat16* __restrict__ x_inout,
    __nv_bfloat16* __restrict__ h, __nv_bfloat16* __restrict__ g,
    __nv_bfloat16* __restrict__ mlp_act, std::int8_t* __restrict__ xq1,
    std::uint16_t* __restrict__ ds81, std::int8_t* __restrict__ xq2,
    std::uint16_t* __restrict__ ds82) {
    {
        const int flat = static_cast<int>(threadIdx.x);
        const int lane = flat & 31;
        const int warp = (flat >> 5) & 3;
        const int wg   = static_cast<int>(blockIdx.x) * 2 + (flat >> 7);
        if (wg < rec_workgroups) {
            const auto hv = static_cast<std::uint32_t>(rec_heads.H_v);
            const std::uint32_t value_head = static_cast<std::uint32_t>(wg) % hv;
            const std::uint32_t state_tile = static_cast<std::uint32_t>(wg) / hv;
            const gated_delta_net::BatchUpdateAccess access{
                rec_q,        rec_k,        rec_v,        rec_g,      rec_beta,
                rec_states_read, rec_states_write, rec_src_slots, rec_dst_slots, rec_out,
                rec_heads,    rec_slot_stride, rec_scale};
            const gated_delta_net::RecurrentCoordinates coord{
                lane,
                warp,
                0,
                0,
                static_cast<std::int32_t>(state_tile),
                value_head,
                static_cast<std::uint32_t>(rec_heads.qk_head(static_cast<int>(value_head))),
                static_cast<std::uint32_t>(state_tile * gated_delta_net::kBlockDv +
                                           warp * gated_delta_net::kDvPerWarp),
                static_cast<std::uint32_t>(lane * gated_delta_net::kQkPerLane)};
            __align__(16) float rec_state[gated_delta_net::kDvPerWarp][gated_delta_net::kQkPerLane];
            gated_delta_net::load_state_tile(rec_state, access.state_read_base(coord), coord);
            gated_delta_net::run_recurrent_sequence<true, gated_delta_net::OutputEffects>(
                rec_state, access, coord, 1);
            gated_delta_net::store_state_tile(rec_state, access.state_write_base(coord), coord);
        }
    }
    // recurrent 段全网格写 o, 前置段 gated_rmsnorm 全网格读 o: 段间屏障(计数槽 [10])。
    chain_barrier(&g_ternary_mlp_chain_counters[10]);
    chain_run_out_proj_stages(1, nullptr, op_act, rec_out, gr_w, gr_z, op_eps, op_codes, op_scales,
                              op_signs, op_perm_hd, op_perm_nk, op_perm_rep, x_inout, op_xq, op_ds8);
    chain_barrier(&g_ternary_mlp_chain_counters[9]);
    chain_run_mlp_stages(x_in, norm_w, mlp_eps, gu_codes, gu_scales, dn_codes, dn_scales, gu_signs,
                         dn_signs, gu_perm_hd, gu_perm_nk, gu_perm_rep, dn_perm_hd, dn_perm_nk,
                         dn_perm_rep, x_inout, h, g, mlp_act, xq1, ds81, xq2, ds82);
}

// ---- T5 片 3 输入投影链的段体 ----

// 段表(逐字复刻 ternary_rowsplit_gemv.cuh 的 TernarySegTable; 该 .cuh 是单 TU 头, 不能包含)。
struct ChainSegTable {
    const std::uint8_t* codes[4];
    const std::uint8_t* scales[4];
    __nv_bfloat16* out[4];
    std::int32_t row_begin[4];
    std::int32_t nsegs;
};

// segn 单行体: ternary_pq2_gemv_dp4a_segn_t1_kernel 的行体逐字(段查表 + 无残差 epilogue)。
__device__ __forceinline__ void chain_segn_gemv_row(int lane, int row,
                                                    const std::int8_t* __restrict__ xq,
                                                    const std::uint16_t* __restrict__ ds8,
                                                    const ChainSegTable& table,
                                                    int groups_per_row) {
    std::int32_t seg = 0;
#pragma unroll
    for (int i = 1; i < 4; ++i) {
        if (row >= table.row_begin[i]) { seg = i; }
    }
    const std::int32_t local = row - table.row_begin[seg];
    const std::uint8_t* code_row =
        table.codes[seg] + static_cast<std::int64_t>(local) * groups_per_row * kCodeBytesPerGroup;
    const std::uint8_t* scale_row = table.scales[seg] + static_cast<std::int64_t>(local) *
                                                           groups_per_row * kScaleBytesPerGroup;
    float accumulator = 0.0f;
#pragma unroll 8
    for (int group = 0; group < groups_per_row; ++group) {
        const std::uint8_t raw = code_row[group * kCodeBytesPerGroup + lane];
        const int w4           = chain_pq2_dp4a_symbols4(raw);
        const int x4           = *reinterpret_cast<const int*>(
            xq + static_cast<std::int64_t>(group) * kGroupK + lane * 4);
        const int dot = chain_pq2_dp4a_dot4(w4, x4, 0);
        const float wsc =
            chain_gemv_scale(scale_row + group * kScaleBytesPerGroup) *
            __half2float(__ushort_as_half(ds8[group * 4 + (lane >> 3)]));
        accumulator = fmaf(wsc, static_cast<float>(dot), accumulator);
    }
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        accumulator += __shfl_xor_sync(0xffffffffffffffffULL, accumulator, offset);
    }
    if (lane == 0) { table.out[seg][local] = __float2bfloat16_rn(accumulator); }
}

// 闸门 GEMV 单行体: bf16_gdn_gating_proj_gemv_kernel 逐字(一 CTA 一行, 256 线程跨步归约)。
// 行 -> CTA 映射与引擎 <<<2*kN, kThreads>>> 相同(cta == global_row); 跨步次序、block_reduce_sum
// 的归约树、softplus/sigmoid 端标全逐字, 归约次序逐位一致。
__device__ void chain_gating_row_stage(int cta, const __nv_bfloat16* __restrict__ x,
                                      const __nv_bfloat16* __restrict__ a_w,
                                      const __nv_bfloat16* __restrict__ b_w,
                                      const float* __restrict__ a_log,
                                      const float* __restrict__ dt_bias, float* __restrict__ g,
                                      float* __restrict__ beta) {
    if (cta >= kGatingRows) { return; }
    const bool is_b    = cta >= kGatingN;
    const int row      = is_b ? cta - kGatingN : cta;
    const auto* weight = is_b ? b_w : a_w;
    __shared__ float chain_gate_sums[kChainThreads / 32];
    float acc = 0.0f;
    constexpr int kPairs        = kInProjK / 2;
    const std::int64_t row_base = static_cast<std::int64_t>(row) * kInProjK;
    const auto* x2              = reinterpret_cast<const __nv_bfloat162*>(x);
    const auto* w2              = reinterpret_cast<const __nv_bfloat162*>(weight + row_base);
    for (int p = static_cast<int>(threadIdx.x); p < kPairs; p += static_cast<int>(kChainThreads)) {
        const float2 wf = __bfloat1622float2(w2[p]);
        const float2 xf = __bfloat1622float2(x2[p]);
        acc             = fmaf(wf.x, xf.x, acc);
        acc             = fmaf(wf.y, xf.y, acc);
    }
    acc = block_reduce_sum<kChainThreads>(acc, chain_gate_sums);
    if (threadIdx.x == 0) {
        if (is_b) {
            beta[row] = sigmoid(acc);
        } else {
            const float sp = softplus(acc + dt_bias[row]);
            g[row]         = -expf(a_log[row]) * sp;
        }
    }
}

// 输入投影链核(attn 型): [input_norm] -> 屏障 -> rotq v3(k=5120) -> 屏障 -> segn GEMV。
// 4 段: q/k 来自 query_key 父, gate/v 来自 gate_value 父。
__global__ __launch_bounds__(kChainThreads) void ternary_input_chain_kernel(
    const __nv_bfloat16* __restrict__ x_raw, const __nv_bfloat16* __restrict__ norm_w, float eps,
    __nv_bfloat16* __restrict__ h, const float* __restrict__ signs, int n_blk, int perm_hd,
    int perm_nk, int perm_rep, ChainSegTable table, int rows_total,
    std::int8_t* __restrict__ xq, std::uint16_t* __restrict__ ds8) {
    if (blockIdx.x == 0) {
        chain_norm_stage(reinterpret_cast<const __nv_bfloat162*>(x_raw),
                         reinterpret_cast<const __nv_bfloat162*>(norm_w),
                         reinterpret_cast<__nv_bfloat162*>(h), kInProjK, eps);
    }
    chain_barrier(&g_ternary_mlp_chain_counters[7]);

    if (blockIdx.x < kInProjRotqGrid) {
        chain_rotq_v3_stage(h, xq, ds8, signs, n_blk, kInProjK, 1, perm_hd, perm_nk, perm_rep);
    }
    chain_barrier(&g_ternary_mlp_chain_counters[8]);

    {
        const int lane        = static_cast<int>(threadIdx.x) & 31;
        const int warp0       = static_cast<int>(blockIdx.x) * 8 + (static_cast<int>(threadIdx.x) >> 5);
        const int total_warps = static_cast<int>(gridDim.x) * 8;
        for (int row = warp0; row < rows_total; row += total_warps) {
            chain_segn_gemv_row(lane, row, xq, ds8, table, kInProjGroups);
        }
    }
}

// ---- T5 片 3b 刀一: [input_norm + 闸门 GEMV] 2 节点并 1 持久核 ----
// 接线面 = gdn_mix 的两支(Verify/prefill 同构), 经 Variant::gdn_norm_gating_chain 尝试;
// 未准入回落 ops::gdn_norm_gating_proj(Composed: rmsnorm 节点 + gdn_gating_proj_gemv 节点)。
// 数值契约 = 两节点原路径逐字节(md5 门判)。闸门体 = chain_gating_row_stage(逐字),
// norm 体 = chain_norm_stage(片 3 attn 链已过 md5 门)。
__global__ __launch_bounds__(kChainThreads) void ternary_norm_gating_chain_kernel(
    const __nv_bfloat16* __restrict__ x_raw, const __nv_bfloat16* __restrict__ norm_w, float eps,
    __nv_bfloat16* __restrict__ h, const __nv_bfloat16* __restrict__ gate_a_w,
    const __nv_bfloat16* __restrict__ gate_b_w, const float* __restrict__ a_log,
    const float* __restrict__ dt_bias, float* __restrict__ g_out, float* __restrict__ beta_out) {
    if (blockIdx.x == 0) {
        chain_norm_stage(reinterpret_cast<const __nv_bfloat162*>(x_raw),
                         reinterpret_cast<const __nv_bfloat162*>(norm_w),
                         reinterpret_cast<__nv_bfloat162*>(h), kInProjK, eps);
    }
    chain_barrier(&g_ternary_mlp_chain_counters[7]);
    chain_gating_row_stage(static_cast<int>(blockIdx.x), h, gate_a_w, gate_b_w, a_log, dt_bias,
                           g_out, beta_out);
}

// ---- T5 片 3b 刀二: [rotq v3 + segn GEMV] 2 节点并 1 持久核 ----
// 接线面 = launch_ternary_split 的入口(必须早于量化上提: hoist_begin 自己会发射旋转+量化
// 节点, 此后链核无从回收)。scratch 必须自带量化缓冲且旋转须为延迟态(激活未旋转, 由本核
// 就地旋转+量化)。段构造与 launch_ternary_split 逐字同构(qk 直切; value|z 同父两行区)。
__global__ __launch_bounds__(kChainThreads) void ternary_segn_chain_kernel(
    const __nv_bfloat16* __restrict__ activation, const float* __restrict__ signs, int n_blk,
    int perm_hd, int perm_nk, int perm_rep, ChainSegTable table, int rows_total,
    std::int8_t* __restrict__ xq, std::uint16_t* __restrict__ ds8) {
    if (blockIdx.x < kInProjRotqGrid) {
        chain_rotq_v3_stage(activation, xq, ds8, signs, n_blk, kInProjK, 1, perm_hd, perm_nk,
                            perm_rep);
    }
    chain_barrier(&g_ternary_mlp_chain_counters[7]);

    const int lane        = static_cast<int>(threadIdx.x) & 31;
    const int warp0       = static_cast<int>(blockIdx.x) * 8 + (static_cast<int>(threadIdx.x) >> 5);
    const int total_warps = static_cast<int>(gridDim.x) * 8;
    for (int row = warp0; row < rows_total; row += total_warps) {
        chain_segn_gemv_row(lane, row, xq, ds8, table, kInProjGroups);
    }
}

// ---- S2a(2026-10-17 剩余节点组): [rotq v3 + segn GEMV + gdn 投影卷积] 3 节点并 1 持久核 ----
// 接线面 = gdn_input_proj_conv_snapshot 的 ternary 分支(快照路径, T=1 单列)。前提与刀二
// 逐条一致(rot_deferred 激活 / 自带量化缓冲); 额外仅 27B 快照形态: 10240=2048+2048+6144。
// conv 段 = gdn_projected_conv_kernel<10240,2048,2048,6144,0,SnapshotHistoryPublish> 的单列
// (T=1, batch=1, token=0)体逐字同构: 同 fmaf 序(rn)、同 silu、同 publish(bf16(s1)/bf16(s2)/
// bf16(p))。数值契约 = 两节点原路径逐字节(md5 门判)。
__global__ __launch_bounds__(kChainThreads) void ternary_segn_conv_chain_kernel(
    const __nv_bfloat16* __restrict__ activation, const float* __restrict__ signs, int n_blk,
    int perm_hd, int perm_nk, int perm_rep, ChainSegTable table, int rows_total,
    std::int8_t* __restrict__ xq, std::uint16_t* __restrict__ ds8,
    const __nv_bfloat16* __restrict__ projected, const __nv_bfloat16* __restrict__ conv_weight,
    const __nv_bfloat16* __restrict__ conv_state_read, const std::int32_t* __restrict__ valid_columns,
    const std::int32_t* __restrict__ initial_state_slots,
    __nv_bfloat16* __restrict__ conv_state_write, const std::int32_t* __restrict__ snapshot_base_slots,
    __nv_bfloat16* __restrict__ query, __nv_bfloat16* __restrict__ key,
    __nv_bfloat16* __restrict__ value, int channels, int query_rows, int key_rows) {
    if (blockIdx.x < kInProjRotqGrid) {
        chain_rotq_v3_stage(activation, xq, ds8, signs, n_blk, kInProjK, 1, perm_hd, perm_nk,
                            perm_rep);
    }
    chain_barrier(&g_ternary_mlp_chain_counters[7]);

    {
        const int lane        = static_cast<int>(threadIdx.x) & 31;
        const int warp0       = static_cast<int>(blockIdx.x) * 8 + (static_cast<int>(threadIdx.x) >> 5);
        const int total_warps = static_cast<int>(gridDim.x) * 8;
        for (int row = warp0; row < rows_total; row += total_warps) {
            chain_segn_gemv_row(lane, row, xq, ds8, table, kInProjGroups);
        }
    }
    chain_barrier(&g_ternary_mlp_chain_counters[8]);

    // conv 段: 一线程一行, 全网格跨步(10240 行 <= 384*256 线程, 单轮覆盖)。
    const std::int32_t valid_raw = valid_columns == nullptr ? 1 : valid_columns[0];
    const std::int32_t valid     = valid_raw < 0 ? 0 : (valid_raw > 1 ? 1 : valid_raw);
    const std::int64_t slot_stride = static_cast<std::int64_t>(channels) * 3;
    const std::int64_t initial_base =
        static_cast<std::int64_t>(initial_state_slots[0]) * slot_stride;
    const std::int64_t snapshot_base =
        static_cast<std::int64_t>(snapshot_base_slots[0]) * slot_stride;
    const std::int64_t conv_stride = static_cast<std::int64_t>(gridDim.x) * kChainThreads;
    for (std::int64_t row = static_cast<std::int64_t>(blockIdx.x) * kChainThreads + threadIdx.x;
         row < channels; row += conv_stride) {
        if (valid <= 0) {
            if (row < query_rows) {
                query[row] = __float2bfloat16_rn(0.0F);
            } else if (row < query_rows + key_rows) {
                key[row - query_rows] = __float2bfloat16_rn(0.0F);
            } else {
                value[row - query_rows - key_rows] = __float2bfloat16_rn(0.0F);
            }
            continue;
        }
        const float s0 = __bfloat162float(conv_state_read[initial_base + row]);
        const float s1 = __bfloat162float(conv_state_read[initial_base + channels + row]);
        const float s2 = __bfloat162float(conv_state_read[initial_base + 2LL * channels + row]);
        const float w0 = __bfloat162float(conv_weight[row]);
        const float w1 = __bfloat162float(conv_weight[channels + row]);
        const float w2 = __bfloat162float(conv_weight[2LL * channels + row]);
        const float w3 = __bfloat162float(conv_weight[3LL * channels + row]);
        const float p  = __bfloat162float(projected[row]);
        float conv     = fmaf(w0, s0, 0.0F);
        conv           = fmaf(w1, s1, conv);
        conv           = fmaf(w2, s2, conv);
        conv           = fmaf(w3, p, conv);
        const __nv_bfloat16 output = __float2bfloat16_rn(silu(conv));
        if (row < query_rows) {
            query[row] = output;
        } else if (row < query_rows + key_rows) {
            key[row - query_rows] = output;
        } else {
            value[row - query_rows - key_rows] = output;
        }
        conv_state_write[snapshot_base + row]                  = __float2bfloat16_rn(s1);
        conv_state_write[snapshot_base + channels + row]       = __float2bfloat16_rn(s2);
        conv_state_write[snapshot_base + 2LL * channels + row] = __float2bfloat16_rn(p);
    }
}

// MLP 段体(s1..s6; 片 2b 抽出为 device 函数, 供独立核与统一尾链核逐字复用)。
// 段体来源逐字对应: s1 norm(chain_norm_stage) / s2,s5 rotq(chain_rotq_v3_stage) /
// s3,s6 gemv(chain_gemv_row, s6 带残差原地) / s4 silu(chain_silu_mul_pair)。
__device__ void chain_run_mlp_stages(
    const __nv_bfloat16* __restrict__ x_in, const __nv_bfloat16* __restrict__ norm_w, float eps,
    const std::uint8_t* __restrict__ gu_codes, const std::uint8_t* __restrict__ gu_scales,
    const std::uint8_t* __restrict__ dn_codes, const std::uint8_t* __restrict__ dn_scales,
    const float* __restrict__ gu_signs, const float* __restrict__ dn_signs, int gu_perm_hd,
    int gu_perm_nk, int gu_perm_rep, int dn_perm_hd, int dn_perm_nk, int dn_perm_rep,
    __nv_bfloat16* __restrict__ x_inout, __nv_bfloat16* __restrict__ h,
    __nv_bfloat16* __restrict__ g, __nv_bfloat16* __restrict__ act, std::int8_t* __restrict__ xq1,
    std::uint16_t* __restrict__ ds81, std::int8_t* __restrict__ xq2,
    std::uint16_t* __restrict__ ds82) {
    // s1: 后范数(逻辑单 CTA 512 线程 -> 物理块 0)
    if (blockIdx.x == 0) {
        chain_norm_stage(reinterpret_cast<const __nv_bfloat162*>(x_in),
                         reinterpret_cast<const __nv_bfloat162*>(norm_w),
                         reinterpret_cast<__nv_bfloat162*>(h), kChainHidden, eps);
    }
    chain_barrier(&g_ternary_mlp_chain_counters[0]);

    // s2: rotquant v3, k=5120(逻辑 blocks=5 -> 物理块 0..2, 映射与独立发射逐块一致)
    if (blockIdx.x < 3) {
        chain_rotq_v3_stage(h, xq1, ds81, gu_signs, kChainHidden / 1024, kChainHidden, 1,
                            gu_perm_hd, gu_perm_nk, gu_perm_rep);
    }
    chain_barrier(&g_ternary_mlp_chain_counters[1]);

    // s3: gate_up GEMV(行 34816, 全网格跨步; 行归属不改变逐行算术)
    {
        const int lane        = static_cast<int>(threadIdx.x) & 31;
        const int warp0       = static_cast<int>(blockIdx.x) * 8 + (static_cast<int>(threadIdx.x) >> 5);
        const int total_warps = static_cast<int>(gridDim.x) * 8;
        for (int row = warp0; row < kChainGuRows; row += total_warps) {
            chain_gemv_row(lane, row, xq1, ds81, gu_codes, gu_scales, g, kChainGuGroups, nullptr);
        }
    }
    chain_barrier(&g_ternary_mlp_chain_counters[2]);

    // s4: silu(对 8704 对; 映射与引擎 dim0_split 同工作量: 前 2176 线程 x 4 对)
    {
        constexpr int kPairsPerThread = 4;
        const std::int64_t tid = blockIdx.x * static_cast<std::int64_t>(kChainThreads) + threadIdx.x;
        const std::int64_t stride =
            static_cast<std::int64_t>(gridDim.x) * kChainThreads * kPairsPerThread;
        const std::int64_t n2 = kChainAct / 2;
        const auto* gate2     = reinterpret_cast<const __nv_bfloat162*>(g);
        const auto* up2       = reinterpret_cast<const __nv_bfloat162*>(g + kChainAct);
        auto* out2            = reinterpret_cast<__nv_bfloat162*>(act);
        for (std::int64_t j = tid * kPairsPerThread; j < n2; j += stride) {
            const __nv_bfloat162 g0 = gate2[j];
            const __nv_bfloat162 u0 = up2[j];
            if (j + 3 < n2) {
                const __nv_bfloat162 g1 = gate2[j + 1];
                const __nv_bfloat162 u1 = up2[j + 1];
                const __nv_bfloat162 g2 = gate2[j + 2];
                const __nv_bfloat162 u2 = up2[j + 2];
                const __nv_bfloat162 g3 = gate2[j + 3];
                const __nv_bfloat162 u3 = up2[j + 3];
                out2[j]                 = chain_silu_mul_pair(g0, u0);
                out2[j + 1]             = chain_silu_mul_pair(g1, u1);
                out2[j + 2]             = chain_silu_mul_pair(g2, u2);
                out2[j + 3]             = chain_silu_mul_pair(g3, u3);
            } else {
                out2[j] = chain_silu_mul_pair(g0, u0);
                if (j + 1 < n2) { out2[j + 1] = chain_silu_mul_pair(gate2[j + 1], up2[j + 1]); }
                if (j + 2 < n2) { out2[j + 2] = chain_silu_mul_pair(gate2[j + 2], up2[j + 2]); }
            }
        }
    }
    chain_barrier(&g_ternary_mlp_chain_counters[3]);

    // s5: rotquant v3, k=17408(逻辑 blocks=17 -> 物理块 0..8)
    if (blockIdx.x < 9) {
        chain_rotq_v3_stage(act, xq2, ds82, dn_signs, kChainAct / 1024, kChainAct, 1, dn_perm_hd,
                            dn_perm_nk, dn_perm_rep);
    }
    chain_barrier(&g_ternary_mlp_chain_counters[4]);

    // s6: down GEMV + 残差 epilogue(原地: residual == out == x_inout)
    {
        const int lane        = static_cast<int>(threadIdx.x) & 31;
        const int warp0       = static_cast<int>(blockIdx.x) * 8 + (static_cast<int>(threadIdx.x) >> 5);
        const int total_warps = static_cast<int>(gridDim.x) * 8;
        for (int row = warp0; row < kChainDnRows; row += total_warps) {
            chain_gemv_row(lane, row, xq2, ds82, dn_codes, dn_scales, x_inout, kChainDnGroups,
                           x_inout);
        }
    }
}

__global__ __launch_bounds__(kChainThreads) void ternary_mlp_chain_kernel(
    const __nv_bfloat16* __restrict__ x_in, const __nv_bfloat16* __restrict__ norm_w, float eps,
    const std::uint8_t* __restrict__ gu_codes, const std::uint8_t* __restrict__ gu_scales,
    const std::uint8_t* __restrict__ dn_codes, const std::uint8_t* __restrict__ dn_scales,
    const float* __restrict__ gu_signs, const float* __restrict__ dn_signs, int gu_perm_hd,
    int gu_perm_nk, int gu_perm_rep, int dn_perm_hd, int dn_perm_nk, int dn_perm_rep,
    __nv_bfloat16* __restrict__ x_inout, __nv_bfloat16* __restrict__ h,
    __nv_bfloat16* __restrict__ g, __nv_bfloat16* __restrict__ act, std::int8_t* __restrict__ xq1,
    std::uint16_t* __restrict__ ds81, std::int8_t* __restrict__ xq2,
    std::uint16_t* __restrict__ ds82) {
    chain_run_mlp_stages(x_in, norm_w, eps, gu_codes, gu_scales, dn_codes, dn_scales, gu_signs,
                         dn_signs, gu_perm_hd, gu_perm_nk, gu_perm_rep, dn_perm_hd, dn_perm_nk,
                         dn_perm_rep, x_inout, h, g, act, xq1, ds81, xq2, ds82);
}

// 准入: 与 folded_dp4a_admits(ternary_rotation.cpp)同族判据 + 大形状/折叠/几何无 P 校验。
bool chain_weight_admits(const Weight& w, std::int32_t rows, std::int32_t cols) {
    if (w.qtype != QType::PQ2_0_G128 || w.n != rows || w.k != cols ||
        w.padded_shape[0] != rows || w.padded_shape[1] != cols || w.group != 128 ||
        w.layout != QuantLayout::RowSplit || w.scale_dtype != DType::FP16 || w.ndim != 2 ||
        w.shape[0] != w.n || w.shape[1] != w.k || w.qdata == nullptr || w.scales == nullptr) {
        return false;
    }
    if (!ternary_weight_is_folded(w) || w.hadamard_n_blk != cols / 1024) { return false; }
    // P 置换分支: 链核按 v3 逐字支持, 但准入限定几何无 P(与 MLP 权重实际载荷一致);
    // 有 P 时回落原路径, 避免把未验证的置换路径带进链。
    const TernaryRotationGeometry geometry = ternary_rotation_geometry(w);
    return geometry.perm_rep <= 1;
}

} // namespace

bool ternary_mlp_chain(const Tensor& x, const Tensor& post_norm, const Weight& gate_up_weight,
                       const Weight& down_weight, float eps, WorkspaceArena& workspace,
                       cudaStream_t stream) {
#if defined(NINFER_HIP)
    if (!ternary_mlp_chain_enabled()) { return false; }
    // 与 folded_dp4a_admits 同族的三道总闸: 旋转总闸、dp4a rung、旋转量化融合闸。
    if (!ternary_rotation_enabled() || !ternary_pq2_dp4a_enabled() ||
        !ternary_rotq_fuse_enabled()) {
        return false;
    }
    if (x.dtype != DType::BF16 || post_norm.dtype != DType::BF16 || !x.is_contiguous() ||
        !post_norm.is_contiguous() || x.ne[0] != kChainHidden || x.ne[1] != 1 || x.ne[2] != 1 ||
        x.ne[3] != 1 || post_norm.ne[0] != kChainHidden) {
        return false;
    }
    if (!chain_weight_admits(gate_up_weight, kChainGuRows, kChainHidden) ||
        !chain_weight_admits(down_weight, kChainDnRows, kChainAct)) {
        return false;
    }
    const unsigned grid = chain_grid();
    if (grid == 0) { return false; } // 物理 CU 查询失败 => 无法保证共驻留, 回落原路径

    auto scope = workspace.scope();
    const DeviceSpan h_span   = workspace.alloc_bytes(kChainHidden * 2);
    const DeviceSpan g_span   = workspace.alloc_bytes(kChainGuRows * 2);
    const DeviceSpan act_span = workspace.alloc_bytes(kChainAct * 2);
    const DeviceSpan xq1_span = workspace.alloc_bytes(kChainHidden);
    const DeviceSpan ds81_span = workspace.alloc_bytes((kChainHidden / 32) * 2);
    const DeviceSpan xq2_span  = workspace.alloc_bytes(kChainAct);
    const DeviceSpan ds82_span = workspace.alloc_bytes((kChainAct / 32) * 2);

    const TernaryRotationGeometry gu_geo = ternary_rotation_geometry(gate_up_weight);
    const TernaryRotationGeometry dn_geo = ternary_rotation_geometry(down_weight);

    ternary_mlp_chain_kernel<<<grid, kChainThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data),
        static_cast<const __nv_bfloat16*>(post_norm.data), eps,
        static_cast<const std::uint8_t*>(gate_up_weight.qdata),
        static_cast<const std::uint8_t*>(gate_up_weight.scales),
        static_cast<const std::uint8_t*>(down_weight.qdata),
        static_cast<const std::uint8_t*>(down_weight.scales), gate_up_weight.hadamard_signs,
        down_weight.hadamard_signs, gu_geo.perm_hd, gu_geo.perm_nk, gu_geo.perm_rep, dn_geo.perm_hd,
        dn_geo.perm_nk, dn_geo.perm_rep, static_cast<__nv_bfloat16*>(x.data),
        static_cast<__nv_bfloat16*>(h_span.data), static_cast<__nv_bfloat16*>(g_span.data),
        static_cast<__nv_bfloat16*>(act_span.data), static_cast<std::int8_t*>(xq1_span.data),
        static_cast<std::uint16_t*>(ds81_span.data), static_cast<std::int8_t*>(xq2_span.data),
        static_cast<std::uint16_t*>(ds82_span.data));
    CUDA_CHECK(cudaGetLastError());
    return true;
#else
    (void)x;
    (void)post_norm;
    (void)gate_up_weight;
    (void)down_weight;
    (void)eps;
    (void)workspace;
    (void)stream;
    return false; // dp4a/perm 内建是 AMDGCN 专有, CUDA 构建恒走原路径
#endif
}

// ---- T5 片 2a: 前置链(o_proj)host 入口 ----

namespace {

// 公共准入: 环境门 + 三道总闸(旋转/dp4a/rotq 融合) + 残差融合闸(x 原地残差是链核的数值形式)
// + act/x 形状 + o_proj 折叠权重几何。act = X 侧激活(attn: sigmoid 后 / gdn: gated_rmsnorm 后),
// x = 残差 in-out 的 hidden。
bool out_proj_chain_admits(const Tensor& act, const Weight& o_proj_weight, const Tensor& x) {
    if (!ternary_out_proj_chain_enabled()) { return false; }
    if (!ternary_rotation_enabled() || !ternary_pq2_dp4a_enabled() ||
        !ternary_rotq_fuse_enabled() || !ternary_res_fuse_enabled()) {
        return false;
    }
    if (act.dtype != DType::BF16 || !act.is_contiguous() || act.ne[0] != kOutProjK ||
        act.ne[1] != 1 || act.ne[2] != 1 || act.ne[3] != 1) {
        return false;
    }
    if (x.dtype != DType::BF16 || !x.is_contiguous() || x.ne[0] != kOutProjRows || x.ne[1] != 1 ||
        x.ne[2] != 1 || x.ne[3] != 1) {
        return false;
    }
    // 网格门槛 = gdn 支 gated_rmsnorm 的行覆盖(48 行 / 每 CTA 8 warp = 6 CTA)。小卡上网格
    // 收窄到不足 6 CTA 时行覆盖不全, 必须回落(chain_grid()=0 = CU 查询失败, 也在此拒绝)。
    const unsigned grid = chain_grid();
    if (grid < 6) { return false; }
    return chain_weight_admits(o_proj_weight, kOutProjRows, kOutProjK);
}

// 核发射公共段: 激活闸门(sigmoid/gated_rmsnorm) -> rotq v3(k=6144) -> o_proj GEMV+残差,
// 工作区 = xq(6144B) + ds8(384B), 与引擎 dp4a 路径同尺寸(声明容量已覆盖, 见 layouts_impl.h)。
void out_proj_chain_launch(int gdn_mode, const __nv_bfloat16* gate, __nv_bfloat16* act,
                           const __nv_bfloat16* gr_x, const __nv_bfloat16* gr_w,
                           const __nv_bfloat16* gr_z, float eps, const Weight& o_proj_weight,
                           __nv_bfloat16* x, WorkspaceArena& workspace, cudaStream_t stream) {
    auto scope = workspace.scope();
    const DeviceSpan xq_span  = workspace.alloc_bytes(kOutProjK);
    const DeviceSpan ds8_span = workspace.alloc_bytes((kOutProjK / 32) * 2);
    const TernaryRotationGeometry geo = ternary_rotation_geometry(o_proj_weight);
    ternary_out_proj_chain_kernel<<<chain_grid(), kChainThreads, 0, stream>>>(
        gdn_mode, gate, act, gr_x, gr_w, gr_z, eps,
        static_cast<const std::uint8_t*>(o_proj_weight.qdata),
        static_cast<const std::uint8_t*>(o_proj_weight.scales), o_proj_weight.hadamard_signs,
        geo.perm_hd, geo.perm_nk, geo.perm_rep, x, static_cast<std::int8_t*>(xq_span.data),
        static_cast<std::uint16_t*>(ds8_span.data));
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

bool ternary_out_proj_chain_attn(const Tensor& gate, Tensor& act, const Weight& o_proj_weight,
                                 Tensor& x, WorkspaceArena& workspace, cudaStream_t stream) {
#if defined(NINFER_HIP)
    if (!out_proj_chain_admits(act, o_proj_weight, x)) { return false; }
    if (gate.dtype != DType::BF16 || !gate.is_contiguous() || gate.ne[0] != kOutProjK ||
        gate.ne[1] != 1 || gate.ne[2] != 1 || gate.ne[3] != 1) {
        return false;
    }
    out_proj_chain_launch(0, static_cast<const __nv_bfloat16*>(gate.data),
                          static_cast<__nv_bfloat16*>(act.data), nullptr, nullptr, nullptr, 0.0f,
                          o_proj_weight, static_cast<__nv_bfloat16*>(x.data), workspace, stream);
    return true;
#else
    (void)gate;
    (void)act;
    (void)o_proj_weight;
    (void)x;
    (void)workspace;
    (void)stream;
    return false; // dp4a/perm 内建是 AMDGCN 专有, CUDA 构建恒走原路径
#endif
}

bool ternary_out_proj_chain_gdn(const Tensor& o, const Tensor& norm_weight, const Tensor& z,
                                Tensor& on, const Weight& o_proj_weight, Tensor& x, float eps,
                                WorkspaceArena& workspace, cudaStream_t stream) {
#if defined(NINFER_HIP)
    // T == 1 硬门先于展平视图({128,48,T} -> {6144,T} 的 view 在 T!=1 时 numel 不匹配会抛)。
    if (o.ne[2] != 1 || on.ne[2] != 1 || z.ne[2] != 1) { return false; }
    Tensor on_flat = on.view({kOutProjK, 1});
    if (!out_proj_chain_admits(on_flat, o_proj_weight, x)) { return false; }
    const auto d128_ok = [](const Tensor& t) {
        return t.dtype == DType::BF16 && t.is_contiguous() && t.ne[0] == kGrD &&
               t.ne[1] == kGrRows && t.ne[2] == 1 && t.ne[3] == 1;
    };
    if (!d128_ok(o) || !d128_ok(z) || !d128_ok(on)) { return false; }
    if (norm_weight.dtype != DType::BF16 || !norm_weight.is_contiguous() ||
        norm_weight.ne[0] != kGrD || norm_weight.ne[1] != 1 || norm_weight.ne[2] != 1 ||
        norm_weight.ne[3] != 1) {
        return false;
    }
    // 引擎 d=128 gated_rmsnorm 走 bf16x2 warp 核的前提 = 四个地址 2 元素对齐; 不对齐时引擎
    // 走 generic 核(不同归约次序 = 非逐位), 链核不冒充, 回落。
    const auto aligned2 = [](const void* p) {
        return (reinterpret_cast<std::uintptr_t>(p) & (alignof(__nv_bfloat162) - 1)) == 0;
    };
    if (!aligned2(o.data) || !aligned2(z.data) || !aligned2(on.data) ||
        !aligned2(norm_weight.data)) {
        return false;
    }
    out_proj_chain_launch(1, nullptr, static_cast<__nv_bfloat16*>(on.data),
                          static_cast<const __nv_bfloat16*>(o.data),
                          static_cast<const __nv_bfloat16*>(norm_weight.data),
                          static_cast<const __nv_bfloat16*>(z.data), eps, o_proj_weight,
                          static_cast<__nv_bfloat16*>(x.data), workspace, stream);
    return true;
#else
    (void)o;
    (void)norm_weight;
    (void)z;
    (void)on;
    (void)o_proj_weight;
    (void)x;
    (void)eps;
    (void)workspace;
    (void)stream;
    return false; // dp4a/perm 内建是 AMDGCN 专有, CUDA 构建恒走原路径
#endif
}

// ---- T5 片 2b: 统一尾链 host 入口 ----

namespace {

// pending 旗标(宿主侧, 捕获期静态窗口; 单会话静态约束同链核): 统一尾链发射成功 = true,
// 由 mlp_tail 的链入口(ternary_mlp_chain_consume_pending)消费一次。图回放不跑宿主代码,
// 捕获期每层"设一次/消一次"自成闭环; 未准入时旗标保持 false, MLP 链照常发射。
// 带层身份(post_norm 缓冲地址, 每层唯一且跨捕获稳定): 消费端核对命中才跳过, 且无论命中
// 与否都清(捕获中途异常退出留下的粘滞旗标不可能误跳过别层的 MLP = 不静默出错)。
bool g_tail_chain_pending      = false;
const void* g_tail_pending_norm = nullptr;

// MLP 侧准入(与 ternary_mlp_chain 的准入同族, 只判定不发射; 含 MLP 链门——任一相关门关
// 即整体回落, 逃生门语义逐门可退)。
bool tail_chain_mlp_admits(const Tensor& x, const Tensor& post_norm,
                           const Weight& gate_up_weight, const Weight& down_weight) {
    if (!ternary_mlp_chain_enabled() || !ternary_rotation_enabled() ||
        !ternary_pq2_dp4a_enabled() || !ternary_rotq_fuse_enabled()) {
        return false;
    }
    if (x.dtype != DType::BF16 || post_norm.dtype != DType::BF16 || !x.is_contiguous() ||
        !post_norm.is_contiguous() || x.ne[0] != kChainHidden || x.ne[1] != 1 || x.ne[2] != 1 ||
        x.ne[3] != 1 || post_norm.ne[0] != kChainHidden) {
        return false;
    }
    return chain_weight_admits(gate_up_weight, kChainGuRows, kChainHidden) &&
           chain_weight_admits(down_weight, kChainDnRows, kChainAct);
}

// 公共发射: 前置段工作区(xq k=6144 + ds8)+ MLP 段工作区(h/g/act/xq1/ds81/xq2/ds82),
// 九件全在站点 scope 内分配, 站点 scope 结束时一并回收。
void tail_chain_launch(int gdn_mode, const __nv_bfloat16* gate, __nv_bfloat16* op_act,
                       const __nv_bfloat16* gr_x, const __nv_bfloat16* gr_w,
                       const __nv_bfloat16* gr_z, float op_eps, const Weight& o_proj_weight,
                       const Tensor& post_norm, const Weight& gate_up_weight,
                       const Weight& down_weight, __nv_bfloat16* x, float mlp_eps,
                       WorkspaceArena& workspace, cudaStream_t stream) {
    auto scope                          = workspace.scope();
    const DeviceSpan op_xq_span         = workspace.alloc_bytes(kOutProjK);
    const DeviceSpan op_ds8_span        = workspace.alloc_bytes((kOutProjK / 32) * 2);
    const DeviceSpan h_span             = workspace.alloc_bytes(kChainHidden * 2);
    const DeviceSpan g_span             = workspace.alloc_bytes(kChainGuRows * 2);
    const DeviceSpan act_span           = workspace.alloc_bytes(kChainAct * 2);
    const DeviceSpan xq1_span           = workspace.alloc_bytes(kChainHidden);
    const DeviceSpan ds81_span          = workspace.alloc_bytes((kChainHidden / 32) * 2);
    const DeviceSpan xq2_span           = workspace.alloc_bytes(kChainAct);
    const DeviceSpan ds82_span          = workspace.alloc_bytes((kChainAct / 32) * 2);
    const TernaryRotationGeometry op_geo = ternary_rotation_geometry(o_proj_weight);
    const TernaryRotationGeometry gu_geo = ternary_rotation_geometry(gate_up_weight);
    const TernaryRotationGeometry dn_geo = ternary_rotation_geometry(down_weight);
    ternary_tail_chain_kernel<<<chain_grid(), kChainThreads, 0, stream>>>(
        gdn_mode, gate, op_act, gr_x, gr_w, gr_z, op_eps,
        static_cast<const std::uint8_t*>(o_proj_weight.qdata),
        static_cast<const std::uint8_t*>(o_proj_weight.scales), o_proj_weight.hadamard_signs,
        op_geo.perm_hd, op_geo.perm_nk, op_geo.perm_rep,
        static_cast<std::int8_t*>(op_xq_span.data),
        static_cast<std::uint16_t*>(op_ds8_span.data), x,
        static_cast<const __nv_bfloat16*>(post_norm.data), mlp_eps,
        static_cast<const std::uint8_t*>(gate_up_weight.qdata),
        static_cast<const std::uint8_t*>(gate_up_weight.scales),
        static_cast<const std::uint8_t*>(down_weight.qdata),
        static_cast<const std::uint8_t*>(down_weight.scales), gate_up_weight.hadamard_signs,
        down_weight.hadamard_signs, gu_geo.perm_hd, gu_geo.perm_nk, gu_geo.perm_rep,
        dn_geo.perm_hd, dn_geo.perm_nk, dn_geo.perm_rep, x,
        static_cast<__nv_bfloat16*>(h_span.data), static_cast<__nv_bfloat16*>(g_span.data),
        static_cast<__nv_bfloat16*>(act_span.data), static_cast<std::int8_t*>(xq1_span.data),
        static_cast<std::uint16_t*>(ds81_span.data), static_cast<std::int8_t*>(xq2_span.data),
        static_cast<std::uint16_t*>(ds82_span.data));
    CUDA_CHECK(cudaGetLastError());
    g_tail_chain_pending = true;
    g_tail_pending_norm  = post_norm.data;
}

// R1 刀公共发射: 工作区与 tail_chain_launch 逐字同款(九件不变; recurrent 段零共享内存),
// 前置段固定 gdn 支(gate=nullptr, op_act=on, gr_x=o); pending 旗标同款(含 MLP 段)。
void tail_chain_recurrent_launch(
    const __nv_bfloat16* rec_q, const __nv_bfloat16* rec_k, const __nv_bfloat16* rec_v,
    const float* rec_g, const float* rec_beta, const float* rec_states_read,
    float* rec_states_write, const std::int32_t* rec_src_slots,
    const std::int32_t* rec_dst_slots, __nv_bfloat16* rec_out,
    gated_delta_net::head_map rec_heads, std::int64_t rec_slot_stride, float rec_scale,
    int rec_workgroups, const Tensor& norm_weight, const Tensor& z, Tensor& on,
    const Weight& o_proj_weight, Tensor& x, const Tensor& post_norm, const Weight& gate_up_weight,
    const Weight& down_weight, float eps, WorkspaceArena& workspace, cudaStream_t stream) {
    auto scope                          = workspace.scope();
    const DeviceSpan op_xq_span         = workspace.alloc_bytes(kOutProjK);
    const DeviceSpan op_ds8_span        = workspace.alloc_bytes((kOutProjK / 32) * 2);
    const DeviceSpan h_span             = workspace.alloc_bytes(kChainHidden * 2);
    const DeviceSpan g_span             = workspace.alloc_bytes(kChainGuRows * 2);
    const DeviceSpan act_span           = workspace.alloc_bytes(kChainAct * 2);
    const DeviceSpan xq1_span           = workspace.alloc_bytes(kChainHidden);
    const DeviceSpan ds81_span          = workspace.alloc_bytes((kChainHidden / 32) * 2);
    const DeviceSpan xq2_span           = workspace.alloc_bytes(kChainAct);
    const DeviceSpan ds82_span          = workspace.alloc_bytes((kChainAct / 32) * 2);
    const TernaryRotationGeometry op_geo = ternary_rotation_geometry(o_proj_weight);
    const TernaryRotationGeometry gu_geo = ternary_rotation_geometry(gate_up_weight);
    const TernaryRotationGeometry dn_geo = ternary_rotation_geometry(down_weight);
    ternary_tail_chain_recurrent_kernel<<<chain_grid(), kChainThreads, 0, stream>>>(
        rec_q, rec_k, rec_v, rec_g, rec_beta, rec_states_read, rec_states_write, rec_src_slots,
        rec_dst_slots, rec_out, rec_heads, rec_slot_stride, rec_scale, rec_workgroups,
        static_cast<__nv_bfloat16*>(on.data),
        static_cast<const __nv_bfloat16*>(norm_weight.data),
        static_cast<const __nv_bfloat16*>(z.data), eps,
        static_cast<const std::uint8_t*>(o_proj_weight.qdata),
        static_cast<const std::uint8_t*>(o_proj_weight.scales), o_proj_weight.hadamard_signs,
        op_geo.perm_hd, op_geo.perm_nk, op_geo.perm_rep,
        static_cast<std::int8_t*>(op_xq_span.data),
        static_cast<std::uint16_t*>(op_ds8_span.data),
        static_cast<const __nv_bfloat16*>(x.data),
        static_cast<const __nv_bfloat16*>(post_norm.data), eps,
        static_cast<const std::uint8_t*>(gate_up_weight.qdata),
        static_cast<const std::uint8_t*>(gate_up_weight.scales),
        static_cast<const std::uint8_t*>(down_weight.qdata),
        static_cast<const std::uint8_t*>(down_weight.scales), gate_up_weight.hadamard_signs,
        down_weight.hadamard_signs, gu_geo.perm_hd, gu_geo.perm_nk, gu_geo.perm_rep,
        dn_geo.perm_hd, dn_geo.perm_nk, dn_geo.perm_rep,
        static_cast<__nv_bfloat16*>(x.data),
        static_cast<__nv_bfloat16*>(h_span.data), static_cast<__nv_bfloat16*>(g_span.data),
        static_cast<__nv_bfloat16*>(act_span.data), static_cast<std::int8_t*>(xq1_span.data),
        static_cast<std::uint16_t*>(ds81_span.data), static_cast<std::int8_t*>(xq2_span.data),
        static_cast<std::uint16_t*>(ds82_span.data));
    CUDA_CHECK(cudaGetLastError());
    g_tail_chain_pending = true;
    g_tail_pending_norm  = post_norm.data;
}

} // namespace

bool ternary_mlp_chain_consume_pending(const void* norm_identity) {
    const bool hit       = g_tail_chain_pending && g_tail_pending_norm == norm_identity;
    g_tail_chain_pending = false; // 命中与否都清(防粘滞)
    g_tail_pending_norm  = nullptr;
    return hit;
}

bool ternary_tail_chain_attn(const Tensor& gate, Tensor& act, const Weight& o_proj_weight,
                             Tensor& x, const Tensor& post_norm, const Weight& gate_up_weight,
                             const Weight& down_weight, float eps, WorkspaceArena& workspace,
                             cudaStream_t stream) {
#if defined(NINFER_HIP)
    if (!ternary_tail_chain_enabled()) { return false; }
    if (!out_proj_chain_admits(act, o_proj_weight, x)) { return false; }
    if (gate.dtype != DType::BF16 || !gate.is_contiguous() || gate.ne[0] != kOutProjK ||
        gate.ne[1] != 1 || gate.ne[2] != 1 || gate.ne[3] != 1) {
        return false;
    }
    if (!tail_chain_mlp_admits(x, post_norm, gate_up_weight, down_weight)) { return false; }
    tail_chain_launch(0, static_cast<const __nv_bfloat16*>(gate.data),
                      static_cast<__nv_bfloat16*>(act.data), nullptr, nullptr, nullptr, 0.0f,
                      o_proj_weight, post_norm, gate_up_weight, down_weight,
                      static_cast<__nv_bfloat16*>(x.data), eps, workspace, stream);
    return true;
#else
    (void)gate;
    (void)act;
    (void)o_proj_weight;
    (void)x;
    (void)post_norm;
    (void)gate_up_weight;
    (void)down_weight;
    (void)eps;
    (void)workspace;
    (void)stream;
    return false; // dp4a/perm 内建是 AMDGCN 专有, CUDA 构建恒走原路径
#endif
}

bool ternary_tail_chain_gdn(const Tensor& o, const Tensor& norm_weight, const Tensor& z, Tensor& on,
                            const Weight& o_proj_weight, Tensor& x, const Tensor& post_norm,
                            const Weight& gate_up_weight, const Weight& down_weight, float eps,
                            WorkspaceArena& workspace, cudaStream_t stream) {
#if defined(NINFER_HIP)
    if (!ternary_tail_chain_enabled()) { return false; }
    // T == 1 硬门先于展平视图({128,48,T} -> {6144,T} 的 view 在 T!=1 时 numel 不匹配会抛)。
    if (o.ne[2] != 1 || on.ne[2] != 1 || z.ne[2] != 1) { return false; }
    Tensor on_flat = on.view({kOutProjK, 1});
    if (!out_proj_chain_admits(on_flat, o_proj_weight, x)) { return false; }
    const auto d128_ok = [](const Tensor& t) {
        return t.dtype == DType::BF16 && t.is_contiguous() && t.ne[0] == kGrD &&
               t.ne[1] == kGrRows && t.ne[2] == 1 && t.ne[3] == 1;
    };
    if (!d128_ok(o) || !d128_ok(z) || !d128_ok(on)) { return false; }
    if (norm_weight.dtype != DType::BF16 || !norm_weight.is_contiguous() ||
        norm_weight.ne[0] != kGrD || norm_weight.ne[1] != 1 || norm_weight.ne[2] != 1 ||
        norm_weight.ne[3] != 1) {
        return false;
    }
    // 引擎 d=128 gated_rmsnorm 走 bf16x2 warp 核的前提 = 四个地址 2 元素对齐(同片 2a 判定)。
    const auto aligned2 = [](const void* p) {
        return (reinterpret_cast<std::uintptr_t>(p) & (alignof(__nv_bfloat162) - 1)) == 0;
    };
    if (!aligned2(o.data) || !aligned2(z.data) || !aligned2(on.data) ||
        !aligned2(norm_weight.data)) {
        return false;
    }
    if (!tail_chain_mlp_admits(x, post_norm, gate_up_weight, down_weight)) { return false; }
    tail_chain_launch(1, nullptr, static_cast<__nv_bfloat16*>(on.data),
                      static_cast<const __nv_bfloat16*>(o.data),
                      static_cast<const __nv_bfloat16*>(norm_weight.data),
                      static_cast<const __nv_bfloat16*>(z.data), eps, o_proj_weight, post_norm,
                      gate_up_weight, down_weight, static_cast<__nv_bfloat16*>(x.data), eps,
                      workspace, stream);
    return true;
#else
    (void)o;
    (void)norm_weight;
    (void)z;
    (void)on;
    (void)o_proj_weight;
    (void)x;
    (void)post_norm;
    (void)gate_up_weight;
    (void)down_weight;
    (void)eps;
    (void)workspace;
    (void)stream;
    return false; // dp4a/perm 内建是 AMDGCN 专有, CUDA 构建恒走原路径
#endif
}

// R1 刀(2026-10-17 剩余节点组): [GDN recurrent 批更新 + 统一尾链] 2 节点并 1。接线面 =
// text_context_impl GDN Verify 支(快照路径 T=1/width=1/batch=1)的 batch_update 节点。准入 =
// 统一尾链全部前提(经同一组助手逐条复用)+ recurrent 侧: bf16 连续 q/k/v、fp32 连续 g/beta、
// 四维视图 width==1 且 batch==1、head 几何自洽(v%q==0 且 v×kStateDim == o 展开规模)、
// WG 数 ≤ 2×网格。数值契约 = 原两节点路径逐字节(md5 门判)。
bool ternary_tail_chain_gdn_recurrent(const Tensor& q, const Tensor& k, const Tensor& v,
                                      const Tensor& g, const Tensor& beta, float scale,
                                      const Tensor& recurrent_states,
                                      const Tensor& source_state_slots,
                                      const Tensor& destination_state_slots, Tensor& o,
                                      const Tensor& norm_weight, const Tensor& z, Tensor& on,
                                      const Weight& o_proj_weight, Tensor& x,
                                      const Tensor& post_norm, const Weight& gate_up_weight,
                                      const Weight& down_weight, float eps,
                                      WorkspaceArena& workspace, cudaStream_t stream) {
#if defined(NINFER_HIP)
    if (!ternary_rec_tail_chain_enabled()) { return false; }
    // 尾链侧前提与 ternary_tail_chain_gdn 逐条同款(该函数体在此逐字复用)。
    if (!ternary_tail_chain_enabled()) { return false; }
    if (o.ne[2] != 1 || on.ne[2] != 1 || z.ne[2] != 1) { return false; }
    Tensor on_flat = on.view({kOutProjK, 1});
    if (!out_proj_chain_admits(on_flat, o_proj_weight, x)) { return false; }
    const auto d128_ok = [](const Tensor& t) {
        return t.dtype == DType::BF16 && t.is_contiguous() && t.ne[0] == kGrD &&
               t.ne[1] == kGrRows && t.ne[2] == 1 && t.ne[3] == 1;
    };
    if (!d128_ok(o) || !d128_ok(z) || !d128_ok(on)) { return false; }
    if (norm_weight.dtype != DType::BF16 || !norm_weight.is_contiguous() ||
        norm_weight.ne[0] != kGrD || norm_weight.ne[1] != 1 || norm_weight.ne[2] != 1 ||
        norm_weight.ne[3] != 1) {
        return false;
    }
    const auto aligned2 = [](const void* p) {
        return (reinterpret_cast<std::uintptr_t>(p) & (alignof(__nv_bfloat162) - 1)) == 0;
    };
    if (!aligned2(o.data) || !aligned2(z.data) || !aligned2(on.data) ||
        !aligned2(norm_weight.data)) {
        return false;
    }
    if (!tail_chain_mlp_admits(x, post_norm, gate_up_weight, down_weight)) { return false; }
    // recurrent 侧: 快照支 T=1 单列单批形态(width==1 ⇒ 引擎走 update-in-place 单 token)。
    const auto bf16_ok = [](const Tensor& t) { return t.dtype == DType::BF16 && t.is_contiguous(); };
    if (!bf16_ok(q) || !bf16_ok(k) || !bf16_ok(v)) { return false; }
    if (g.dtype != DType::FP32 || beta.dtype != DType::FP32 || !g.is_contiguous() ||
        !beta.is_contiguous()) {
        return false;
    }
    if (q.ne[2] != 1 || q.ne[3] != 1 || k.ne[2] != 1 || k.ne[3] != 1 || v.ne[2] != 1 ||
        v.ne[3] != 1) {
        return false;
    }
    if (q.ne[0] != gated_delta_net::kStateDim || v.ne[0] != gated_delta_net::kStateDim) {
        return false;
    }
    if (recurrent_states.data == nullptr || source_state_slots.data == nullptr ||
        destination_state_slots.data == nullptr) {
        return false;
    }
    const std::int32_t qk_heads = q.ne[1];
    const std::int32_t v_heads  = v.ne[1];
    if (qk_heads <= 0 || v_heads <= 0 || v_heads % qk_heads != 0) { return false; }
    // 输出规模自洽: v_heads × kStateDim == o 的二维展开(128×48)。
    if (static_cast<std::int64_t>(v_heads) * gated_delta_net::kStateDim !=
        static_cast<std::int64_t>(o.ne[0]) * o.ne[1]) {
        return false;
    }
    // WG 数(引擎几何: 每 WG = 4 warp, 链 CTA = 8 warp ⇒ 每 CTA 2 WG; 超限回落)。
    const int workgroups =
        v_heads * (gated_delta_net::kStateDim / gated_delta_net::kBlockDv);
    if (workgroups <= 0 || static_cast<unsigned>(workgroups) > 2u * chain_grid()) {
        return false;
    }
    const gated_delta_net::head_map heads = gated_delta_net::head_map::of(qk_heads, v_heads);
    const std::int64_t slot_stride =
        static_cast<std::int64_t>(gated_delta_net::kStateDim) * gated_delta_net::kStateDim *
        recurrent_states.ne[2];
    tail_chain_recurrent_launch(
        static_cast<const __nv_bfloat16*>(q.data), static_cast<const __nv_bfloat16*>(k.data),
        static_cast<const __nv_bfloat16*>(v.data), static_cast<const float*>(g.data),
        static_cast<const float*>(beta.data), static_cast<const float*>(recurrent_states.data),
        static_cast<float*>(recurrent_states.data),
        static_cast<const std::int32_t*>(source_state_slots.data),
        static_cast<const std::int32_t*>(destination_state_slots.data),
        static_cast<__nv_bfloat16*>(o.data), heads, slot_stride, scale, workgroups, norm_weight, z,
        on, o_proj_weight, x, post_norm, gate_up_weight, down_weight, eps, workspace, stream);
    return true;
#else
    (void)q;
    (void)k;
    (void)v;
    (void)g;
    (void)beta;
    (void)scale;
    (void)recurrent_states;
    (void)source_state_slots;
    (void)destination_state_slots;
    (void)o;
    (void)norm_weight;
    (void)z;
    (void)on;
    (void)o_proj_weight;
    (void)x;
    (void)post_norm;
    (void)gate_up_weight;
    (void)down_weight;
    (void)eps;
    (void)workspace;
    (void)stream;
    return false; // dp4a/perm 内建是 AMDGCN 专有, CUDA 构建恒走原路径
#endif
}

// ---- T5 片 3: 输入投影链 host 入口 ----

namespace {

// 段权重: 与 chain_weight_admits 同族 + k=5120 固定 + 旋转基行数吻合(n_blk=5)。
bool input_chain_weight_admits(const Weight& w, std::int32_t rows) {
    return chain_weight_admits(w, rows, kInProjK) && w.hadamard_n_blk == kInProjNBlk;
}

// bf16 [rows, 1] 连续张量。
bool input_chain_bf16_row(const Tensor& t, std::int32_t rows) {
    return t.dtype == DType::BF16 && t.is_contiguous() && t.ne[0] == rows && t.ne[1] == 1 &&
           t.ne[2] == 1 && t.ne[3] == 1;
}

// fp32 [rows, 1] 连续张量(a_log/dt_bias/g/beta 同族)。
bool input_chain_fp32_col(const Tensor& t, std::int32_t rows) {
    return t.dtype == DType::FP32 && t.is_contiguous() && t.ne[0] == rows && t.ne[1] == 1 &&
           t.ne[2] == 1 && t.ne[3] == 1 && t.data != nullptr;
}

// 闸门 bf16 权重(gdn_gating_proj.cpp require_bf16_weight 的准入版)。
bool input_chain_gate_weight(const Weight& w) {
    return w.qtype == QType::BF16_CTRL && w.layout == QuantLayout::Contiguous && w.ndim == 2 &&
           w.n == kGatingN && w.k == kInProjK && w.shape[0] == kGatingN &&
           w.shape[1] == kInProjK && w.padded_shape[0] == kGatingN &&
           w.padded_shape[1] == kInProjK && w.qhigh == nullptr && w.scales == nullptr &&
           w.qdata != nullptr &&
           (reinterpret_cast<std::uintptr_t>(w.qdata) & (alignof(__nv_bfloat162) - 1)) == 0;
}

bool input_chain_aligned2(const void* p) {
    return (reinterpret_cast<std::uintptr_t>(p) & (alignof(__nv_bfloat162) - 1)) == 0;
}

// 形状/类型/对齐准入(不含环境门与物理网格): 激活/归一权重/输出 h 的 [k,1] bf16 形态。
bool input_chain_shape_admits(const Tensor& x_raw, const Tensor& norm_w, const Tensor& h) {
    if (!input_chain_bf16_row(x_raw, kInProjK) || !input_chain_bf16_row(h, kInProjK)) {
        return false;
    }
    if (norm_w.dtype != DType::BF16 || !norm_w.is_contiguous() || norm_w.ne[0] != kInProjK ||
        norm_w.ne[1] != 1 || norm_w.ne[2] != 1 || norm_w.ne[3] != 1) {
        return false;
    }
    // 引擎 d>3072 的 CTA norm 走 bf16x2 路线的前提 = 三地址 2 元素对齐; 链核同前提。
    return input_chain_aligned2(x_raw.data) && input_chain_aligned2(norm_w.data) &&
           input_chain_aligned2(h.data);
}

// 公共准入: 环境门 + 四总闸 + 形状 + 物理网格门槛。
// (segn 融合闸按值解析: 引擎的 merge 关掉时链核不冒充, 回落逐段派发。)
bool input_chain_common(const Tensor& x_raw, const Tensor& norm_w, const Tensor& h,
                        unsigned min_grid) {
    if (!ternary_input_chain_enabled()) { return false; }
    if (!ternary_rotation_enabled() || !ternary_pq2_dp4a_enabled() ||
        !ternary_rotq_fuse_enabled() || !ternary_segn_fuse_enabled()) {
        return false;
    }
    if (!input_chain_shape_admits(x_raw, norm_w, h)) { return false; }
    return chain_grid() >= min_grid;
}

// attn 型输入投影链的发射参数封装。
struct InputChainLaunch {
    const Tensor* x_raw   = nullptr;
    const Tensor* norm_w  = nullptr;
    float eps             = 0.0f;
    Tensor* h             = nullptr;
    const Weight* rot_w   = nullptr;
    ChainSegTable table{};
    int rows_total = 0;
};

// 段表填写: codes/scales/out/row_begin(与 launch_pq2_dp4a_segn 的表构造逐字同序)。
void fill_chain_seg_table(ChainSegTable& table, const Weight* const* ws, Tensor* const* outs,
                          const std::int32_t* rows, std::int32_t nsegs) {
    std::int32_t total = 0;
    for (std::int32_t i = 0; i < nsegs; ++i) {
        table.codes[i]    = static_cast<const std::uint8_t*>(ws[i]->qdata);
        table.scales[i]   = static_cast<const std::uint8_t*>(ws[i]->scales);
        table.out[i]      = static_cast<__nv_bfloat16*>(outs[i]->data);
        table.row_begin[i] = total;
        total += rows[i];
    }
    for (std::int32_t i = nsegs; i < 4; ++i) { table.row_begin[i] = 0x7FFFFFFF; }
    table.nsegs = nsegs;
}

bool input_chain_launch(const InputChainLaunch& args, WorkspaceArena& workspace,
                        cudaStream_t stream) {
    auto scope = workspace.scope();
    const DeviceSpan xq_span  = workspace.alloc_bytes(kInProjK);
    const DeviceSpan ds8_span = workspace.alloc_bytes((kInProjK / 32) * 2);
    const TernaryRotationGeometry geo = ternary_rotation_geometry(*args.rot_w);
    ternary_input_chain_kernel<<<chain_grid(), kChainThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(args.x_raw->data),
        static_cast<const __nv_bfloat16*>(args.norm_w->data), args.eps,
        static_cast<__nv_bfloat16*>(args.h->data), args.rot_w->hadamard_signs, kInProjNBlk,
        geo.perm_hd, geo.perm_nk, geo.perm_rep, args.table, args.rows_total,
        static_cast<std::int8_t*>(xq_span.data), static_cast<std::uint16_t*>(ds8_span.data));
    CUDA_CHECK(cudaGetLastError());
    return true;
}

} // namespace

bool ternary_input_chain_attn(const Tensor& x_raw, const Tensor& norm_w, float eps,
                              const Weight& query_key_w, const Weight& gate_value_w, Tensor& q,
                              Tensor& gate, Tensor& k, Tensor& v, Tensor& h,
                              WorkspaceArena& workspace, cudaStream_t stream) {
#if defined(NINFER_HIP)
    if (!input_chain_common(x_raw, norm_w, h, kInProjRotqGrid)) { return false; }
    if (!input_chain_bf16_row(q, kAttnQRows) || !input_chain_bf16_row(gate, kAttnQRows) ||
        !input_chain_bf16_row(k, kAttnKvRows) || !input_chain_bf16_row(v, kAttnKvRows)) {
        return false;
    }
    if (!input_chain_weight_admits(query_key_w, kAttnQRows + kAttnKvRows) ||
        !input_chain_weight_admits(gate_value_w, kAttnQRows + kAttnKvRows)) {
        return false;
    }
    // 段构造与 launch_ternary_attn 逐字同构(q|k 来自 query_key 父, gate|v 来自 gate_value 父,
    // 四段顺序 = q, k, gate, v)。
    const Weight q_head = ternary_row_view(query_key_w, 0, kAttnQRows);
    const Weight k_tail = ternary_row_view(query_key_w, kAttnQRows, kAttnKvRows);
    const Weight g_head = ternary_row_view(gate_value_w, 0, kAttnQRows);
    const Weight v_tail = ternary_row_view(gate_value_w, kAttnQRows, kAttnKvRows);
    InputChainLaunch args{};
    args.x_raw    = &x_raw;
    args.norm_w   = &norm_w;
    args.eps      = eps;
    args.h        = &h;
    args.rot_w    = &query_key_w;
    const Weight* const ws[4]  = {&q_head, &k_tail, &g_head, &v_tail};
    Tensor* const outs[4]      = {&q, &k, &gate, &v};
    const std::int32_t rows[4] = {kAttnQRows, kAttnKvRows, kAttnQRows, kAttnKvRows};
    fill_chain_seg_table(args.table, ws, outs, rows, 4);
    args.rows_total = 2 * (kAttnQRows + kAttnKvRows);
    return input_chain_launch(args, workspace, stream);
#else
    (void)x_raw;
    (void)norm_w;
    (void)eps;
    (void)query_key_w;
    (void)gate_value_w;
    (void)q;
    (void)gate;
    (void)k;
    (void)v;
    (void)h;
    (void)workspace;
    (void)stream;
    return false; // dp4a/perm 内建是 AMDGCN 专有, CUDA 构建恒走原路径
#endif
}

bool ternary_norm_gating_chain(const Tensor& x_raw, const Tensor& norm_w, float eps,
                               const Weight& a_w, const Weight& b_w, const Tensor& a_log,
                               const Tensor& dt_bias, Tensor& h, Tensor& g, Tensor& beta,
                               cudaStream_t stream) {
#if defined(NINFER_HIP)
    // T=1 是硬准入(形状门隐含: [k,1] bf16), norm 体与闸门体都按单列写。
    if (!ternary_input_chain_enabled()) { return false; }
    if (!input_chain_shape_admits(x_raw, norm_w, h)) { return false; }
    // 网格门槛 = 闸门行数: 96 行一 CTA 一行, 小卡收窄会漏行(kGatingRows 之上的 CTA 空转)。
    if (chain_grid() < kGatingMinGrid) { return false; }
    if (!input_chain_gate_weight(a_w) || !input_chain_gate_weight(b_w)) { return false; }
    if (!input_chain_fp32_col(a_log, kGatingN) || !input_chain_fp32_col(dt_bias, kGatingN) ||
        !input_chain_fp32_col(g, kGatingN) || !input_chain_fp32_col(beta, kGatingN)) {
        return false;
    }
    ternary_norm_gating_chain_kernel<<<chain_grid(), kChainThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x_raw.data),
        static_cast<const __nv_bfloat16*>(norm_w.data), eps,
        static_cast<__nv_bfloat16*>(h.data), static_cast<const __nv_bfloat16*>(a_w.qdata),
        static_cast<const __nv_bfloat16*>(b_w.qdata), static_cast<const float*>(a_log.data),
        static_cast<const float*>(dt_bias.data), static_cast<float*>(g.data),
        static_cast<float*>(beta.data));
    CUDA_CHECK(cudaGetLastError());
    return true;
#else
    (void)x_raw;
    (void)norm_w;
    (void)eps;
    (void)a_w;
    (void)b_w;
    (void)a_log;
    (void)dt_bias;
    (void)h;
    (void)g;
    (void)beta;
    (void)stream;
    return false; // dp4a/perm 内建是 AMDGCN 专有, CUDA 构建恒走原路径
#endif
}

bool ternary_segn_chain_split(const Tensor& activation, const Weight& qk_w,
                              const Weight& value_z_w, Tensor& qkv, Tensor& z,
                              std::int32_t qk_rows, std::int32_t value_rows, std::int32_t z_rows,
                              TernaryS8Scratch scratch, cudaStream_t stream) {
#if defined(NINFER_HIP)
    if (!ternary_input_chain_enabled()) { return false; }
    if (!ternary_rotation_enabled() || !ternary_pq2_dp4a_enabled() ||
        !ternary_rotq_fuse_enabled() || !ternary_segn_fuse_enabled()) {
        return false;
    }
    // 本刀前提 = 延迟旋转: 激活未旋转, 由链核就地旋转+量化(未延迟时旋转不能在核内重做一遍)。
    if (!scratch.rot_deferred() || scratch.dp4a_xq == nullptr || scratch.dp4a_ds8 == nullptr) {
        return false;
    }
    if (scratch.rot_n_blk != kInProjNBlk) { return false; } // k=5120 的 signs 行数(几何自证)
    if (!input_chain_bf16_row(activation, kInProjK)) { return false; }
    if (!input_chain_bf16_row(qkv, qk_rows + value_rows) || !input_chain_bf16_row(z, z_rows)) {
        return false;
    }
    if (chain_grid() < kInProjRotqGrid) { return false; }
    if (!input_chain_weight_admits(qk_w, qk_rows) ||
        !input_chain_weight_admits(value_z_w, value_rows + z_rows)) {
        return false;
    }
    // 段构造与 launch_ternary_split 逐字同构(qk 直切; value|z 同父两行区, qkv 两半共用父行步长)。
    Tensor qk_out    = qkv.slice(0, 0, qk_rows);
    Tensor value_out = qkv.slice(0, qk_rows, value_rows);
    const Weight value_head = ternary_row_view(value_z_w, 0, value_rows);
    const Weight value_tail = ternary_row_view(value_z_w, value_rows, z_rows);
    ChainSegTable table{};
    const Weight* const ws[3]  = {&qk_w, &value_head, &value_tail};
    Tensor* const outs[3]      = {&qk_out, &value_out, &z};
    const std::int32_t rows[3] = {qk_rows, value_rows, z_rows};
    fill_chain_seg_table(table, ws, outs, rows, 3);
    ternary_segn_chain_kernel<<<chain_grid(), kChainThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(activation.data), scratch.rot_signs, scratch.rot_n_blk,
        scratch.rot_perm_hd, scratch.rot_perm_nk, scratch.rot_perm_rep, table,
        qk_rows + value_rows + z_rows, scratch.dp4a_xq, scratch.dp4a_ds8);
    CUDA_CHECK(cudaGetLastError());
    return true;
#else
    (void)activation;
    (void)qk_w;
    (void)value_z_w;
    (void)qkv;
    (void)z;
    (void)qk_rows;
    (void)value_rows;
    (void)z_rows;
    (void)scratch;
    (void)stream;
    return false; // dp4a/perm 内建是 AMDGCN 专有, CUDA 构建恒走原路径
#endif
}

// S2a(2026-10-17 剩余节点组): [rotq v3 + segn GEMV + gdn 投影卷积] 三合链入口。
// 接线面 = gdn_input_proj_conv_snapshot 的 ternary 分支(快照路径): launch_ternary_split +
// gdn_projected_conv_snapshot_launch 两节点 → 1。准入 = 刀二全部条件(经同一组助手逐条复
// 用) + conv 侧仅限 27B 快照形态(T=1 单列, 10240 = 2048+2048+6144)。conv 状态读写同缓冲
// (引擎快照支本就传同一 conv_states)。数值契约 = 两节点原路径逐字节(md5 门判)。
bool ternary_segn_conv_chain_split(const Tensor& activation, const Weight& qk_w,
                                   const Weight& value_z_w, Tensor& qkv, Tensor& z,
                                   std::int32_t qk_rows, std::int32_t value_rows,
                                   std::int32_t z_rows, const Tensor& conv_weight,
                                   Tensor& conv_states, const Tensor& valid_columns,
                                   const Tensor& initial_state_slots,
                                   const Tensor& snapshot_base_slots, Tensor& query, Tensor& key,
                                   Tensor& value, std::int32_t query_rows, std::int32_t key_rows,
                                   TernaryS8Scratch scratch, cudaStream_t stream) {
#if defined(NINFER_HIP)
    if (!ternary_segn_conv_chain_enabled()) { return false; }
    // 与刀二逐条对齐的四闸(链核自带 rotq+量化, 前提是全路径配置一致)。
    if (!ternary_input_chain_enabled()) { return false; }
    if (!ternary_rotation_enabled() || !ternary_pq2_dp4a_enabled() ||
        !ternary_rotq_fuse_enabled() || !ternary_segn_fuse_enabled()) {
        return false;
    }
    // 刀二同款前提: 旋转/量化延迟态 + 自带缓冲 + k=5120 的 signs 行数(几何自证)。
    if (!scratch.rot_deferred() || scratch.dp4a_xq == nullptr || scratch.dp4a_ds8 == nullptr) {
        return false;
    }
    if (scratch.rot_n_blk != kInProjNBlk) { return false; }
    if (!input_chain_bf16_row(activation, kInProjK)) { return false; }
    if (!input_chain_bf16_row(qkv, qk_rows + value_rows) || !input_chain_bf16_row(z, z_rows)) {
        return false;
    }
    if (chain_grid() < kInProjRotqGrid) { return false; }
    if (!input_chain_weight_admits(qk_w, qk_rows) ||
        !input_chain_weight_admits(value_z_w, value_rows + z_rows)) {
        return false;
    }
    // conv 侧(快照形态专有): 27B 几何 = gdn_projected_conv_kernel 的注册形状 + T=1 单列。
    constexpr std::int32_t kChannels = 2048 + 2048 + 6144;
    if (qk_rows != 4096 || value_rows != 6144 || z_rows != 6144) { return false; }
    if (query_rows != 2048 || key_rows != 2048) { return false; }
    if (query.ne[0] != query_rows || key.ne[0] != key_rows || value.ne[0] != 6144) { return false; }
    if (conv_weight.data == nullptr || conv_states.data == nullptr ||
        initial_state_slots.data == nullptr || snapshot_base_slots.data == nullptr) {
        return false;
    }
    // 段构造与 launch_ternary_split / 刀二逐字同构(qk 直切; value|z 同父两行区)。
    Tensor qk_out    = qkv.slice(0, 0, qk_rows);
    Tensor value_out = qkv.slice(0, qk_rows, value_rows);
    const Weight value_head = ternary_row_view(value_z_w, 0, value_rows);
    const Weight value_tail = ternary_row_view(value_z_w, value_rows, z_rows);
    ChainSegTable table{};
    const Weight* const ws[3]  = {&qk_w, &value_head, &value_tail};
    Tensor* const outs[3]      = {&qk_out, &value_out, &z};
    const std::int32_t rows[3] = {qk_rows, value_rows, z_rows};
    fill_chain_seg_table(table, ws, outs, rows, 3);
    ternary_segn_conv_chain_kernel<<<chain_grid(), kChainThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(activation.data), scratch.rot_signs, scratch.rot_n_blk,
        scratch.rot_perm_hd, scratch.rot_perm_nk, scratch.rot_perm_rep, table,
        qk_rows + value_rows + z_rows, scratch.dp4a_xq, scratch.dp4a_ds8,
        static_cast<const __nv_bfloat16*>(qkv.data),
        static_cast<const __nv_bfloat16*>(conv_weight.data),
        static_cast<const __nv_bfloat16*>(conv_states.data),
        valid_columns.data == nullptr
            ? nullptr
            : static_cast<const std::int32_t*>(valid_columns.data),
        static_cast<const std::int32_t*>(initial_state_slots.data),
        static_cast<__nv_bfloat16*>(conv_states.data),
        static_cast<const std::int32_t*>(snapshot_base_slots.data),
        static_cast<__nv_bfloat16*>(query.data), static_cast<__nv_bfloat16*>(key.data),
        static_cast<__nv_bfloat16*>(value.data), kChannels, query_rows, key_rows);
    CUDA_CHECK(cudaGetLastError());
    return true;
#else
    (void)activation;
    (void)qk_w;
    (void)value_z_w;
    (void)qkv;
    (void)z;
    (void)qk_rows;
    (void)value_rows;
    (void)z_rows;
    (void)conv_weight;
    (void)conv_states;
    (void)valid_columns;
    (void)initial_state_slots;
    (void)snapshot_base_slots;
    (void)query;
    (void)key;
    (void)value;
    (void)query_rows;
    (void)key_rows;
    (void)scratch;
    (void)stream;
    return false; // dp4a/perm 内建是 AMDGCN 专有, CUDA 构建恒走原路径
#endif
}

} // namespace ninfer::ops::detail
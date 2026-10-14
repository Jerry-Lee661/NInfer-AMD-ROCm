#pragma once

// Host-safe half of the int8 rung (#14): the token threshold, the scratch descriptor and its sizes.
//
// Separate from ternary_rowsplit_mma_s8.cuh on purpose: that header carries the CUDA kernel and
// device-only syntax, and it must never be included by a host translation unit. ternary_dispatch.cpp
// and ternary_rotation.cpp are host TUs, and including the kernel header there drags
// cuda_pipeline_helpers.h into MSVC, which fails with "unexpected volatile" -- measured, that is
// exactly how the first build of this rung broke.

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <string>

namespace ninfer::ops::detail {

// A/B and rollback switch for the int8 rung, in the same style as NINFER_TERNARY_MMA / _HADAMARD.
// It is needed because this rung CHANGES THE NUMERICS (activations get quantized to int8), so it must
// be comparable inside one binary and switchable off without a rebuild:
//   NINFER_TERNARY_S8=0   -> stay on the bf16 rungs
// Read once, because the choice decides which kernel enters the captured CUDA graph.
[[nodiscard]] inline bool ternary_s8_enabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("NINFER_TERNARY_S8");
        return value == nullptr || std::string(value) != "0";
    }();
    return enabled;
}

// First token count at which the int8 rung beats both bf16 rungs. Measured on the real shape mix
// (tscale_bench, 2026-09-20, quantization pass included): T=32 is a tie (43.25 vs 42.49 small_t),
// T=40 s8 wins by 1.20x, T=48 by 1.22x, T=64 by 1.30x; below T=32 the small-t rung's 10.2 ms per
// 8-token pass is still cheaper.
inline constexpr int kTernaryS8MinTokens = 33;

// Round 17 experiment (sm_86 decode): NINFER_S8_DECODE=1 admits the decode window (T = 2..8)
// to the int8 rung. The round-15 architecture research tied the decode round cost to the base-3
// decode ALU chain (162-270G ops/round, predicted 19-32 ms vs measured 25-28 ms) -- the int8 mma
// consumes the code plane directly and removes that chain. The s8 kernel is fully token-guarded
// (zero-filled staging, guarded stores), so small T is correct, just mma-wasteful. Off by default:
// int8 activation quantization changes results, so the PPL gate must pass before flipping on.
[[nodiscard]] inline bool ternary_s8_decode_admits(std::int32_t tokens) {
    static const bool decode_gate = [] {
        const char* value = std::getenv("NINFER_S8_DECODE");
        return value != nullptr && std::string(value) == "1";
    }();
    return decode_gate && tokens >= 2;
}

// Activation-quantization scratch for the int8 rung: one int8 code row per token (token-major, the
// same layout as the activation it is built from) plus one fp32 scale per token.
//
// The caller owns this memory. The ternary linear op runs inside captured CUDA graphs, so a lazy
// cudaMalloc at launch time is illegal; it must come from the op's workspace arena, and the arena's
// capacity must count it (see ternary_rotation_workspace_bytes()).
struct TernaryS8Scratch {
    std::int8_t* codes  = nullptr;
    float*       scales = nullptr;
    // PQ2_0 decode GEMV 的 dp4a 配方 scratch(NINFER_PQ2_DP4A=1, T=1 专用):
    // q8_1 预量化激活(int8 码 + 每 32 元素 fp16 scale), PrismML mmvq 同款。
    // 空指针 = 该 rung 不可用, launch 保持 bf16 arm。
    std::int8_t*   dp4a_xq  = nullptr;
    std::uint16_t* dp4a_ds8 = nullptr;
    // 旋转延迟(M2, megakernel): 非空 = 传下来的激活是"未旋转"的, dp4a_xq 必须由融合核
    // (rotate+quant 合一)就地生成, 而不是读已旋转的 bf16 缓冲。由
    // folded_activation_or_defer() 判定并填写, 随 scratch 值传递(无全局窗口, 图捕获安全);
    // 几何取自填写它的那个权重, 与激活的折叠基同源。
    const float* rot_signs    = nullptr;
    std::int32_t rot_n_blk    = 0;
    std::int32_t rot_perm_hd  = 0;
    std::int32_t rot_perm_nk  = 0;
    std::int32_t rot_perm_rep = 1;
    [[nodiscard]] bool rot_deferred() const noexcept { return rot_signs != nullptr; }
};

inline constexpr std::size_t ternary_s8_codes_bytes(std::int32_t k, std::int32_t tokens) {
    return static_cast<std::size_t>(k) * static_cast<std::size_t>(tokens);
}

inline constexpr std::size_t ternary_s8_scales_bytes(std::int32_t tokens) {
    return static_cast<std::size_t>(tokens) * sizeof(float);
}

// dp4a rung 的 scratch 字节数。tokens 参数留给未来的 verify-tile 刀; 当前 rung 只服务
// T=1, 分配端与 workspace 容量端都按 tokens=1 计(ternary_rotation_workspace_bytes)。
inline constexpr std::size_t ternary_pq2_dp4a_xq_bytes(std::int32_t k, std::int32_t tokens) {
    return static_cast<std::size_t>(k) * static_cast<std::size_t>(tokens);
}

inline constexpr std::size_t ternary_pq2_dp4a_ds8_bytes(std::int32_t k, std::int32_t tokens) {
    return static_cast<std::size_t>(k / 32) * sizeof(std::uint16_t) *
           static_cast<std::size_t>(tokens);
}

// A/B 门(2026-10-03 默认开): dp4a 配方(T=1..4 PQ2 decode)覆盖收口 + 数值门绿
// (T2 连贯 PASS / T3 无回归 / serve mtp0 +5.4-6.2%, 见 bench/evidence/pb1/pb3)后
// 翻入默认路径。NINFER_PQ2_DP4A=0 回退 bf16 arm, 同二进制 A/B 保留。
// 与 NINFER_TERNARY_S8 同纪律: 读一次, 图捕获语义安全。
[[nodiscard]] inline bool ternary_pq2_dp4a_enabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("NINFER_PQ2_DP4A");
        return value == nullptr || std::string(value) != "0";
    }();
    return enabled;
}

// 量化上提门(2026-10-08, megakernel M1a): NINFER_DP4A_QHOIST=0 回退。默认开依据:
// 节点 1675→1531(−144),nospec 42.9→43.5(+1.4%),d2 27.14(md5 3857bdb9 逐位一致,
// 验收 1.81 不变)——纯发射数优化,数值逐位不变。
[[nodiscard]] inline bool ternary_pq2_dp4a_qhoist_enabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("NINFER_DP4A_QHOIST");
        return value == nullptr || std::string(value) != "0";
    }();
    return enabled;
}

// 双段合并门(2026-10-08, megakernel M2c): 开 = 同一父权重的两个行区一段发射(attn q|k、
// gdn value|z), 消一半 launch 并把 GEMV 推上大尺寸臂(冷 DRAM 探针:逐发带宽 164→591 GB/s
// 随 1K→17K 行爬坡)。逐行数学与分片发射逐位一致。NINFER_SEG2_FUSE=0 回退分片发射。
[[nodiscard]] inline bool ternary_seg2_fuse_enabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("NINFER_SEG2_FUSE");
        return value == nullptr || std::string(value) != "0";
    }();
    return enabled;
}

// segn 表驱动合并门(2026-10-08, megakernel M2d): 开 = T=1 的多个切片(可跨父权重)一段
// 发射(attn 4→1,gdn 输入 2→1)。结构与已探针定案的 T=1 专用形态同构(只多 3 次整数比较)。
// NINFER_SEGN_FUSE=0 回退分片发射(经 seg2/strided)。
[[nodiscard]] inline bool ternary_segn_fuse_enabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("NINFER_SEGN_FUSE");
        return value == nullptr || std::string(value) != "0";
    }();
    return enabled;
}

// 残差 epilogue 融合门(2026-10-08, megakernel M2 刀二): 开 = dp4a GEMV/tile 在 epilogue
// 里直接把残差加进写出的行(ternary linear_add 原地 residual += W*x), 省一个 residual_add
// 节点(每步 128)。数值契约: 舍入点与 compose 路径逐位一致(先舍 bf16 投影结果, 再加, 再舍)。
// NINFER_RES_FUSE=0 回退 compose, 同二进制 A/B。
[[nodiscard]] inline bool ternary_res_fuse_enabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("NINFER_RES_FUSE");
        return value == nullptr || std::string(value) != "0";
    }();
    return enabled;
}

// 旋转延迟融合门(2026-10-08 立项;**2026-10-09 v2(LDS 转置 absmax)翻默认开**): 开 =
// dp4a 量化核就地做 Hadamard(rotate+quant 合一),省 ternary_rotate_bf16 的 launch 与
// bf16 往返。v1(butterfly absmax)判负 −8.3%;v2 = 32×33 LDS 转置替代 shuffle 蝶形
// (探针位一致 + 引擎 md5 含 d2 全逐位一致),nospec 46.44→47.29/47.16(+1.8%),
// 节点 1227→**970**(−257),d2 27.52 验收 1.81 无回归。NINFER_ROTQ_FUSE=0 回退。
[[nodiscard]] inline bool ternary_rotq_fuse_enabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("NINFER_ROTQ_FUSE");
        return value == nullptr || std::string(value) != "0";
    }();
    return enabled;
}

} // namespace ninfer::ops::detail

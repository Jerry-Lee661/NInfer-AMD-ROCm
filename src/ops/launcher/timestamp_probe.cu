// P-C② 诊断件(2026-10-03): 设备侧时间戳写, 供 MTP 轮内分段计价。
// 为什么不用 hipEvent: HIP 捕获期 record 的事件只在 capture 当下触发, replay 不重触发
// (实测 elapsed 冻结在 capture 期入队墙钟)。s_memtime 写入 pinned 主存槽, 图 replay
// 每轮重触发, 宿主任意时刻可读。

#include <cuda_runtime.h>

#include <cstdint>

#include "core/device.h"

// 诊断件, 符号归 targets/qwen3_6/detail(mtp_impl.h 的声明所在), 不进 ops 公共面。
namespace ninfer::targets::qwen3_6::detail {
namespace {

__device__ __forceinline__ std::uint64_t device_timestamp_ticks() {
    // gfx1100 无 s_memtime(RDNA3 移除, builtin 缺 target feature); clock64() 由
    // 编译器映射到本架构可用的周期计数序列。
    return clock64();
}

__global__ void mtp_timestamp_probe_kernel(std::uint64_t* slots, int slot) {
    slots[slot] = device_timestamp_ticks();
}

}  // namespace

void mtp_timestamp_probe_launch(std::uint64_t* device_slots, int slot, cudaStream_t stream) {
    mtp_timestamp_probe_kernel<<<1, 1, 0, stream>>>(device_slots, slot);
    CUDA_CHECK(cudaGetLastError());
}

}  // namespace ninfer::targets::qwen3_6::detail

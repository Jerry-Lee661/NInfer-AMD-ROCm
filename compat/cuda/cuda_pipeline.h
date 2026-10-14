#pragma once
// HIP backend shim: synchronous stand-ins for the CUDA async pipeline intrinsics.
// cp.async has no RDNA3 equivalent; callers keep their double-buffer structure and
// the copies simply happen at the copy site instead of being deferred to wait_group.
#include <cstddef>

#include <hip/hip_runtime.h>

__device__ inline void __pipeline_memcpy_async(void* dst, const void* src, std::size_t size) {
    char* d = static_cast<char*>(dst);
    const char* s = static_cast<const char*>(src);
    for (std::size_t i = 0; i < size; ++i) d[i] = s[i];
}

__device__ inline void __pipeline_commit() {}

__device__ inline void __pipeline_wait_prior(std::size_t) {}

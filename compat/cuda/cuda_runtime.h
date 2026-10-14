#pragma once
// HIP backend shim: CUDA runtime API surface over HIP.
//
// Compiled only when NINFER_BACKEND=HIP (CMake adds compat/cuda as the first include
// dir and defines NINFER_HIP=1). Every CUDA runtime symbol NInfer consumes maps here;
// signatures marked inline in this header are the ones where HIP and CUDA disagree
// (graph instantiate/update arity, launch configs, PDL) — everything else is a
// one-to-one rename verified against the HIP 7.1 headers on the target machine.

#include <hip/hip_runtime.h>

#include <cstddef>

// CUDA 12 kernel-parameter qualifier; HIP has no counterpart and drops the optimization.
#define __grid_constant__
// CUDA 12.4 / 12.7 per-kernel register budget qualifier; HIP has no counterpart, so it expands
// away and the register budget is dropped.
#define __maxnreg__(n)

// Upstream pins CUDA 12.8; only consumed as a version label by request logging.
#define CUDART_VERSION 12080

// --- Types -------------------------------------------------------------------
using cudaError_t = hipError_t;
using cudaStream_t = hipStream_t;
using cudaEvent_t = hipEvent_t;
using cudaGraph_t = hipGraph_t;
using cudaGraphExec_t = hipGraphExec_t;
using cudaGraphNode_t = hipGraphNode_t;
using cudaGraphExecUpdateResult = hipGraphExecUpdateResult;
using cudaDeviceProp = hipDeviceProp_t;
using cudaUUID_t = hipUUID;

// --- Enum values -------------------------------------------------------------
#define cudaSuccess hipSuccess
#define cudaErrorInvalidValue hipErrorInvalidValue
#define cudaErrorInvalidConfiguration hipErrorInvalidConfiguration
#define cudaErrorNotReady hipErrorNotReady
#define cudaMemcpyHostToDevice hipMemcpyHostToDevice
#define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost
#define cudaMemcpyDeviceToDevice hipMemcpyDeviceToDevice
#define cudaStreamNonBlocking hipStreamNonBlocking
#define cudaEventDisableTiming hipEventDisableTiming
#define cudaStreamCaptureModeThreadLocal hipStreamCaptureModeThreadLocal
#define cudaStreamCaptureModeGlobal hipStreamCaptureModeGlobal
#define cudaErrorNoDevice hipErrorNoDevice
#define cudaErrorInsufficientDriver hipErrorInsufficientDriver
#define cudaGraphExecUpdateSuccess hipGraphExecUpdateSuccess
#define cudaFuncAttributeMaxDynamicSharedMemorySize hipFuncAttributeMaxDynamicSharedMemorySize

// --- One-to-one runtime functions --------------------------------------------
#define cudaMalloc hipMalloc
#define cudaFree hipFree
#define cudaMallocHost hipMallocHost
#define cudaFreeHost hipFreeHost
#define cudaMemcpy hipMemcpy
#define cudaMemcpyAsync hipMemcpyAsync
#define cudaMemcpy2D hipMemcpy2D
#define cudaMemcpy2DAsync hipMemcpy2DAsync
#define cudaMemset hipMemset
#define cudaMemsetAsync hipMemsetAsync
#define cudaMemset2DAsync hipMemset2DAsync
#define cudaMemcpyToSymbol hipMemcpyToSymbol
#define cudaMemcpyFromSymbol hipMemcpyFromSymbol
#define cudaMemGetInfo hipMemGetInfo
#define cudaGetDeviceCount hipGetDeviceCount
#define cudaSetDevice hipSetDevice
#define cudaGetDeviceProperties hipGetDeviceProperties
#define cudaGetErrorName hipGetErrorName
#define cudaGetErrorString hipGetErrorString
#define cudaGetLastError hipGetLastError
#define cudaDriverGetVersion hipDriverGetVersion
#define cudaRuntimeGetVersion hipRuntimeGetVersion
#define cudaGetDevice hipGetDevice
#define cudaDeviceGetAttribute hipDeviceGetAttribute
#define cudaDeviceSynchronize hipDeviceSynchronize
#define cudaDevAttrMultiProcessorCount hipDeviceAttributeMultiprocessorCount
#define cudaDevAttrClockRate hipDeviceAttributeClockRate
#define cudaDevAttrMaxSharedMemoryPerBlockOptin hipDeviceAttributeMaxSharedMemoryPerBlock
#define cudaDevAttrComputeCapabilityMajor hipDeviceAttributeComputeCapabilityMajor
#define cudaDevAttrComputeCapabilityMinor hipDeviceAttributeComputeCapabilityMinor
#define cudaStreamCreate hipStreamCreate
#define cudaStreamCreateWithFlags hipStreamCreateWithFlags
#define cudaStreamDestroy hipStreamDestroy
#define cudaStreamSynchronize hipStreamSynchronize
#define cudaStreamWaitEvent hipStreamWaitEvent
#define cudaEventCreate hipEventCreate
#define cudaEventCreateWithFlags hipEventCreateWithFlags
#define cudaEventDestroy hipEventDestroy
#define cudaEventRecord hipEventRecord
#define cudaEventQuery hipEventQuery
#define cudaEventSynchronize hipEventSynchronize
#define cudaEventElapsedTime hipEventElapsedTime
#define cudaStreamBeginCapture hipStreamBeginCapture
#define cudaStreamEndCapture hipStreamEndCapture
#define cudaGraphDestroy hipGraphDestroy
#define cudaGraphGetNodes hipGraphGetNodes
#define cudaGraphLaunch hipGraphLaunch
#define cudaGraphUpload hipGraphUpload
#define cudaGraphExecDestroy hipGraphExecDestroy

// --- Graph node introspection (2026-10-08, NINFER_GRAPH_NODES inventory) ------
// CUDA's graph-node query surface mapped onto the HIP equivalents one to one.
typedef hipGraphNodeType cudaGraphNodeType;
#define cudaGraphNodeGetType hipGraphNodeGetType
typedef hipKernelNodeParams cudaKernelNodeParams;
#define cudaGraphKernelNodeGetParams hipGraphKernelNodeGetParams
#define cudaGraphNodeTypeKernel hipGraphNodeTypeKernel

// CUDA accepts a function pointer implicitly; HIP takes const void*, which a function
// pointer does not convert to implicitly in C++. Route through an explicit wrapper.
template <class KernelFunc>
inline hipError_t cudaFuncSetAttribute(KernelFunc* func, hipFuncAttribute attr, int value) {
    return hipFuncSetAttribute(reinterpret_cast<const void*>(func), attr, value);
}

// --- Graph APIs with different arity -----------------------------------------
// CUDA 12: cudaGraphInstantiate(exec, graph, flags). HIP: hipGraphInstantiateWithFlags.
inline hipError_t cudaGraphInstantiate(hipGraphExec_t* exec, hipGraph_t graph,
                                       unsigned long long flags) {
    return hipGraphInstantiateWithFlags(exec, graph, flags);
}

// CUDA's deprecated 5-argument form, still what the graph-replay tests call. The error-node and
// log-buffer out-params have no HIP counterpart here: hipGraphInstantiateWithFlags reports
// through the return code alone, so the extra arguments are dropped exactly as CUDA 12 drops them.
inline hipError_t cudaGraphInstantiate(hipGraphExec_t* exec, hipGraph_t graph,
                                       hipGraphNode_t /*errorNode*/, char* /*logBuffer*/,
                                       std::size_t /*bufferSize*/) {
    return hipGraphInstantiateWithFlags(exec, graph, 0);
}

// CUDA 12.8: cudaGraphExecUpdate(exec, graph, resultInfo*) with
// cudaGraphExecUpdateResultInfo{ result, ... }. HIP keeps the 4-ary legacy signature.
struct cudaGraphExecUpdateResultInfo {
    hipGraphExecUpdateResult result;
    hipGraphNode_t errorNode;
};

inline hipError_t cudaGraphExecUpdate(hipGraphExec_t exec, hipGraph_t graph,
                                      cudaGraphExecUpdateResultInfo* resultInfo) {
    hipGraphNode_t errorNode = nullptr;
    hipGraphExecUpdateResult result = hipGraphExecUpdateSuccess;
    const hipError_t err = hipGraphExecUpdate(exec, graph, &errorNode, &result);
    if (resultInfo != nullptr) {
        resultInfo->result = result;
        resultInfo->errorNode = errorNode;
    }
    return err;
}

// --- Launch configs ------------------------------------------------------------
// HIP has no programmatic-stream-serialization attribute (RDNA3 has no PDL); that
// attribute is silently dropped, which degrades PDL launches to plain stream ordering.
enum cudaLaunchAttributeID {
    cudaLaunchAttributeIgnore = 0,
    cudaLaunchAttributeCooperative = 1,
    cudaLaunchAttributeProgrammaticStreamSerialization = 2,
};

struct cudaLaunchAttribute {
    cudaLaunchAttributeID id;
    union {
        int cooperative;
        int programmaticStreamSerializationAllowed;
    } val;
};

struct cudaLaunchConfig_t {
    dim3 gridDim;
    dim3 blockDim;
    std::size_t dynamicSmemBytes = 0;
    cudaStream_t stream = nullptr;
    cudaLaunchAttribute* attrs = nullptr;
    unsigned numAttrs = 0;
};

template <class... ExpTypes, class... ActTypes>
inline hipError_t cudaLaunchKernelEx(const cudaLaunchConfig_t* config,
                                     void (*kernel)(ExpTypes...), ActTypes&&... args) {
    hipLaunchConfig_t hipConfig{};
    hipConfig.gridDim = config->gridDim;
    hipConfig.blockDim = config->blockDim;
    hipConfig.dynamicSmemBytes = config->dynamicSmemBytes;
    hipConfig.stream = config->stream;

    hipLaunchAttribute hipAttrs[2];
    unsigned numHipAttrs = 0;
    for (unsigned i = 0; i < config->numAttrs; ++i) {
        if (config->attrs[i].id == cudaLaunchAttributeCooperative) {
            hipAttrs[numHipAttrs].id = hipLaunchAttributeCooperative;
            hipAttrs[numHipAttrs].val.cooperative = config->attrs[i].val.cooperative;
            ++numHipAttrs;
        }
        // ProgrammaticStreamSerialization: dropped, see comment above.
    }
    hipConfig.attrs = hipAttrs;
    hipConfig.numAttrs = numHipAttrs;

    void* kernelArgs[] = {const_cast<void*>(static_cast<const void*>(&args))...};
    return hipLaunchKernelExC(&hipConfig, reinterpret_cast<const void*>(kernel), kernelArgs);
}

// --- Programmatic dependent launch device intrinsics ---------------------------
// Degrade to stream-order semantics: the trigger is a no-op and the consumer wait
// returns immediately (stream ordering already guarantees producer completion).
__device__ inline void cudaTriggerProgrammaticLaunchCompletion() {}

__device__ inline void cudaGridDependencySynchronize() {}

// The bf16 API rides along with the runtime header (mirroring CUDA's umbrella
// behavior) so translation units that only include <cuda_runtime.h> still see
// __float2bfloat16 & co., whose HIP-side names were renamed away above.
#include "cuda_bf16.h"

// Debian-family libstdc++ keeps the C math functions out of the global namespace in
// device translation units (Ubuntu's packaging leaks them, which upstream relies on).
// Templates rather than using-declarations: on Debian the ::fmaf/::max names do not
// exist at all, so `using` would be a hard error; templates merely add overloads and
// never clash with Ubuntu's non-template leaks.
#include <cmath>
namespace ninfer_hip_math_shim {
template <class T, class U>
static inline auto max(T a, U b) -> decltype(a > b ? a : b) { return a > b ? a : b; }
template <class T, class U>
static inline auto min(T a, U b) -> decltype(a < b ? a : b) { return a < b ? a : b; }
template <class T, class U, class V>
static inline auto fmaf(T a, U b, V c) -> decltype(a * b + c) { return a * b + c; }
template <class T, class U, class V>
static inline auto fma(T a, U b, V c) -> decltype(a * b + c) { return a * b + c; }
}  // namespace ninfer_hip_math_shim
using ninfer_hip_math_shim::max;
using ninfer_hip_math_shim::min;
using ninfer_hip_math_shim::fmaf;
using ninfer_hip_math_shim::fma;

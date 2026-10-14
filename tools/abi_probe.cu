// Cross-toolchain ABI probe: compiled on Kali (/opt/hip71 headers), run on X99 (distro runtime).
// Prints the device properties the engine's CC gate depends on, plus the dynamic shared memory
// ceiling the causal-prompt tiles were halved against.
#include <hip/hip_runtime.h>
#include <cstdio>

namespace {

__global__ void smem_probe_kernel(float* out) { out[threadIdx.x] = 0.0F; }

// 4b16c70 cut the prompt tiles from 64/64 to 32/32 because gfx1100's LDS is 64 KiB, but the
// halving was never measured: it assumed the ceiling excludes an exactly-64-KiB request. 64/32
// needs precisely 65536 B of dynamic smem, so the accept-and-launch pair below is what decides
// whether that half-step can be given back. 98304 B is the sm_86 size and must fail here -- if it
// does not, this probe is not measuring what it claims to.
void probe_smem(float* buf, int bytes) {
    const hipError_t attr = hipFuncSetAttribute(
        reinterpret_cast<const void*>(smem_probe_kernel), hipFuncAttributeMaxDynamicSharedMemorySize,
        bytes);
    if (attr != hipSuccess) {
        std::printf("  smem %7d B: setattr=%s\n", bytes, hipGetErrorString(attr));
        return;
    }
    smem_probe_kernel<<<1, 32, bytes>>>(buf);
    const hipError_t run = hipDeviceSynchronize();
    std::printf("  smem %7d B: setattr=ok launch=%s\n", bytes,
                run == hipSuccess ? "ok" : hipGetErrorString(run));
}

} // namespace

int main() {
    hipDeviceProp_t props{};
    hipError_t err = hipGetDeviceProperties(&props, 0);
    if (err != hipSuccess) {
        std::printf("hipGetDeviceProperties failed: %s\n", hipGetErrorString(err));
        return 2;
    }
    int attr_major = -1, attr_minor = -1, attr_mp = -1;
    hipDeviceGetAttribute(&attr_major, hipDeviceAttributeComputeCapabilityMajor, 0);
    hipDeviceGetAttribute(&attr_minor, hipDeviceAttributeComputeCapabilityMinor, 0);
    hipDeviceGetAttribute(&attr_mp, hipDeviceAttributeMultiprocessorCount, 0);
    std::printf("struct:   major=%d minor=%d mp=%d gmem=%.1f GiB name=%s\n", props.major,
                props.minor, props.multiProcessorCount,
                static_cast<double>(props.totalGlobalMem) / 1073741824.0, props.name);
    std::printf("attribute: major=%d minor=%d mp=%d\n", attr_major, attr_minor, attr_mp);
    std::printf("sm() would be: %d (gate wants 86/89/110)\n", props.major * 10 + props.minor);
    // HIP 7.1 exposes no "...PerBlockOptin" attribute, which is why compat/cuda maps
    // cudaDevAttrMaxSharedMemoryPerBlockOptin onto the plain per-block one below.
    int smem_block = -1, smem_mp = -1;
    hipDeviceGetAttribute(&smem_block, hipDeviceAttributeMaxSharedMemoryPerBlock, 0);
    hipDeviceGetAttribute(&smem_mp, hipDeviceAttributeMaxSharedMemoryPerMultiprocessor, 0);
    std::printf("smem: attr perBlock=%d perMP=%d | struct perBlock=%zu perMP=%zu\n", smem_block,
                smem_mp, props.sharedMemPerBlock, props.sharedMemPerMultiprocessor);
    float* buf = nullptr;
    if (hipMalloc(&buf, 32 * sizeof(float)) != hipSuccess) {
        std::printf("hipMalloc failed, skipping the smem ladder\n");
        return 0;
    }
    for (const int bytes : {49152, 55296, 65536, 98304}) { probe_smem(buf, bytes); }
    (void)hipFree(buf);
    return 0;
}

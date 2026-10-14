// Decode-step dispatch cost probe: is the per-op launch path what caps decode t/s?
//
// With CUDA Graphs bypassed the engine issues roughly 800-1,300 single-kernel launches per decode
// step. perf-survey attributes 4-12 ms/step to that, which is the basis for the "hipGraph revival
// is worth 2-5x" line in the P3 backlog. This probe measures the same three shapes directly and
// separately, so the backlog can be ordered against a number instead of an estimate:
//
//   1. bare launches on a stream           -> the cost the bypassed engine actually pays
//   2. one captured graph, many launches   -> what a replay saves per step
//   3. capture + instantiate + upload      -> one-time cost the engine would add at startup
//   4. cudaGraphExecUpdate round trips     -> whether replay-SSM style updates work on HIP 7.1
//      at all, which is the open question behind the --no-cuda-graph bypass
//
// Build (standalone, nothing of libninfer is linked):
//   HIP,  X99 from ~/ninfer:
//     hipcc -O2 -std=c++20 -DNINFER_HIP=1 -I compat/cuda -I src -I include \
//       --offload-arch=gfx1100 tools/hip_dispatch_probe.cu -o /tmp/hip_dispatch_probe
//   CUDA reference (same source, real CUDA graph API):
//     nvcc -O2 -std=c++20 -arch=sm_86 -I src -I include tools/hip_dispatch_probe.cu -o /tmp/dispatch_probe
//
// Exit code 0 = every measured shape ran; a graph API that is unsupported reports SKIP instead and
// still exits 0, because "not available on this stack" is a result, not a failure.

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <vector>

#include <cuda_runtime.h>

#define PROBE_CHECK(expr)                                                                          \
    do {                                                                                           \
        const cudaError_t error__ = (expr);                                                        \
        if (error__ != cudaSuccess) {                                                              \
            std::fprintf(stderr, "probe error at %s:%d: %s\n", __FILE__, __LINE__,                 \
                         cudaGetErrorString(error__));                                             \
            return 2;                                                                              \
        }                                                                                          \
    } while (false)

namespace {

// A kernel with enough substance to not be dropped, small enough that the measurement is
// dominated by dispatch rather than by the work itself.
__global__ void probe_kernel(float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = out[i] * 1.000001f + 1.0e-6f;
    }
}

constexpr int kElements  = 8192;
constexpr int kGrid      = kElements / 256;
constexpr int kStepLaunch = 1000;  // stands in for the 800-1,300 launches of one decode step
constexpr int kRepeat    = 40;

using Clock = std::chrono::steady_clock;

double ms_since(const Clock::time_point& start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

// Median over repeats: dispatch cost is noisy and a single cold run misleads.
template <class Run>
double median_ms(Run&& run, int samples) {
    std::vector<double> v;
    v.reserve(samples);
    for (int i = 0; i < samples; ++i) {
        const Clock::time_point t0 = Clock::now();
        run();
        v.push_back(ms_since(t0));
    }
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

}  // namespace

int main() {
    cudaDeviceProp prop{};
    PROBE_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::printf("dispatch probe on %s (%d MPs)\n", prop.name, prop.multiProcessorCount);

    float* buf = nullptr;
    PROBE_CHECK(cudaMalloc(&buf, kElements * sizeof(float)));
    PROBE_CHECK(cudaMemset(buf, 0, kElements * sizeof(float)));

    // The exec-update step re-points the same kernels at this second buffer, so the update is a
    // real parameter change rather than a no-op re-upload of identical parameters.
    float* buf_repointed = nullptr;
    PROBE_CHECK(cudaMalloc(&buf_repointed, kElements * sizeof(float)));
    PROBE_CHECK(cudaMemset(buf_repointed, 0, kElements * sizeof(float)));

    // 1. Bare launches, one stream, sync once at the end: what the bypassed engine pays.
    const double bare_ms = median_ms([&] {
        for (int i = 0; i < kStepLaunch; ++i) {
            probe_kernel<<<kGrid, 256>>>(buf, kElements);
        }
        cudaStreamSynchronize(nullptr);
    }, kRepeat);
    std::printf("bare launches        %8.3f ms / %d ops  -> %6.3f us/op   x1000 = %6.1f ms/step\n",
                bare_ms, kStepLaunch, bare_ms * 1000.0 / kStepLaunch, bare_ms);

    // 2. The same work captured into one graph and replayed.
    cudaStream_t stream = nullptr;
    cudaError_t err     = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    if (err != cudaSuccess) {
        std::printf("graph path         SKIP (stream create: %s)\n", cudaGetErrorString(err));
        return 0;
    }

    cudaGraph_t graph     = nullptr;
    cudaGraphExec_t exec  = nullptr;
    err = cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal);
    if (err != cudaSuccess) {
        std::printf("graph path         SKIP (begin capture: %s)\n", cudaGetErrorString(err));
        return 0;
    }
    for (int i = 0; i < kStepLaunch; ++i) {
        probe_kernel<<<kGrid, 256, 0, stream>>>(buf, kElements);
    }
    PROBE_CHECK(cudaStreamEndCapture(stream, &graph));

    const Clock::time_point t_inst = Clock::now();
    PROBE_CHECK(cudaGraphInstantiate(&exec, graph, 0));
    const double instantiate_once_ms = ms_since(t_inst);
    PROBE_CHECK(cudaGraphUpload(exec, stream));

    const double replay_ms = median_ms([&] {
        cudaGraphLaunch(exec, stream);
        cudaStreamSynchronize(stream);
    }, kRepeat);
    std::printf("graph replay       %8.3f ms / %d ops  -> %6.3f us/op   x1000 = %6.1f ms/step\n",
                replay_ms, kStepLaunch, replay_ms * 1000.0 / kStepLaunch, replay_ms);
    std::printf("one-time costs      instantiate %.2f ms (startup only)\n", instantiate_once_ms);
    std::printf("dispatch saving     %.2fx per %d-op step\n", bare_ms / replay_ms, kStepLaunch);

    // 3. ExecUpdate with re-pointed buffers: what the ReplaySSM/decode path does every step, and
    //    the call that forced --no-cuda-graph during P2. The topology, node count and kernel
    //    functions stay identical; only the first argument changes, so a rejection here is a
    //    parameter-update limitation rather than a topology-change limitation. HIP reports the
    //    reason through hipGraphExecUpdateResult (0 success, 1 error, 2 topology changed,
    //    3 node type, 4 function, 5 parameters changed in an unsupported way, 6 not supported,
    //    7 unsupported function change).
    cudaGraph_t graph2 = nullptr;
    cudaGraphExec_t exec2 = nullptr;
    PROBE_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    for (int i = 0; i < kStepLaunch; ++i) {
        probe_kernel<<<kGrid, 256, 0, stream>>>(buf_repointed, kElements);
    }
    PROBE_CHECK(cudaStreamEndCapture(stream, &graph2));
    PROBE_CHECK(cudaGraphInstantiate(&exec2, graph2, 0));

    const double update_ms = median_ms([&] {
        cudaGraphExecUpdateResultInfo info{};
        cudaGraphExecUpdate(exec, graph2, &info);
    }, kRepeat);
    cudaGraphExecUpdateResultInfo probe_info{};
    const cudaError_t update_err = cudaGraphExecUpdate(exec, graph2, &probe_info);
    std::printf("exec update        %8.3f ms per round -> %s (result=%d errorNode=%s)\n", update_ms,
                update_err == cudaSuccess && probe_info.result == cudaGraphExecUpdateSuccess
                    ? "accepted"
                    : "rejected",
                static_cast<int>(probe_info.result),
                probe_info.errorNode == nullptr ? "none" : "reported");

    const double replay_after_update_ms = median_ms([&] {
        cudaGraphLaunch(exec, stream);
        cudaStreamSynchronize(stream);
    }, kRepeat);
    std::printf("replay post-update %8.3f ms / %d ops (update costs %.3f us/op)\n",
                replay_after_update_ms, kStepLaunch,
                update_ms * 1000.0 / kStepLaunch);

    cudaGraphExecDestroy(exec);
    cudaGraphExecDestroy(exec2);
    cudaGraphDestroy(graph);
    cudaGraphDestroy(graph2);
    cudaStreamDestroy(stream);
    cudaFree(buf);
    std::printf("done\n");
    return 0;
}

#include "core/decode_graph.h"

#include "core/device.h"

#include <cxxabi.h>

#include <cstdio>
#include <cstdlib>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace ninfer {
namespace {

void log_cuda_error(const char* op, cudaError_t err) noexcept {
    if (err != cudaSuccess) {
        std::fprintf(stderr, "CUDA cleanup failed during %s: %s: %s\n", op, cudaGetErrorName(err),
                     cudaGetErrorString(err));
    }
}

void destroy_graph_exec(cudaGraphExec_t& exec) noexcept {
    if (exec != nullptr) {
        log_cuda_error("cudaGraphExecDestroy", cudaGraphExecDestroy(exec));
        exec = nullptr;
    }
}

void destroy_graph(cudaGraph_t& graph) noexcept {
    if (graph != nullptr) {
        log_cuda_error("cudaGraphDestroy", cudaGraphDestroy(graph));
        graph = nullptr;
    }
}

void discard_capture(cudaStream_t stream) noexcept {
    cudaGraph_t discard = nullptr;
    log_cuda_error("cudaStreamEndCapture(discard)", cudaStreamEndCapture(stream, &discard));
    destroy_graph(discard);
}

} // namespace

DecodeGraphDefinition::~DecodeGraphDefinition() { reset(); }

DecodeGraphDefinition::DecodeGraphDefinition(DecodeGraphDefinition&& other) noexcept
    : graph_(other.graph_) {
    other.graph_ = nullptr;
}

DecodeGraphDefinition& DecodeGraphDefinition::operator=(DecodeGraphDefinition&& other) noexcept {
    if (this == &other) { return *this; }

    reset();
    graph_ = other.graph_;

    other.graph_ = nullptr;
    return *this;
}

void DecodeGraphDefinition::capture(cudaStream_t stream, const std::function<void()>& body) {
    reset();

    CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));

    try {
        body();
    } catch (...) {
        discard_capture(stream);
        throw;
    }

    cudaGraph_t graph = nullptr;

    cudaError_t err = cudaStreamEndCapture(stream, &graph);
    if (err != cudaSuccess) {
        destroy_graph(graph);
        CUDA_CHECK(err);
    }

    graph_ = graph;
}

bool DecodeGraphDefinition::ready() const noexcept { return graph_ != nullptr; }

void DecodeGraphDefinition::reset() noexcept { destroy_graph(graph_); }

DecodeGraphExecutable::~DecodeGraphExecutable() { reset(); }

DecodeGraphExecutable::DecodeGraphExecutable(DecodeGraphExecutable&& other) noexcept
    : exec_(other.exec_) {
    other.exec_ = nullptr;
}

DecodeGraphExecutable& DecodeGraphExecutable::operator=(DecodeGraphExecutable&& other) noexcept {
    if (this == &other) { return *this; }

    reset();
    exec_       = other.exec_;
    other.exec_ = nullptr;
    return *this;
}

void DecodeGraphExecutable::instantiate(const DecodeGraphDefinition& definition) {
    if (!definition.ready()) {
        throw std::logic_error("cannot instantiate an empty CUDA Graph definition");
    }
    reset();

    cudaGraphExec_t exec  = nullptr;
    const cudaError_t err = cudaGraphInstantiate(&exec, definition.graph_, 0);
    if (err != cudaSuccess) {
        destroy_graph_exec(exec);
        CUDA_CHECK(err);
    }
    exec_ = exec;
    // 诊断(2026-10-08, NINFER_GRAPH_NODES): 每步/每轮的内核发射数与逐节点清单。
    // megakernel 融合排布的仪器:1 = 计数,2 = 计数 + 按内核名分组(hipKernelNameRefByPtr)。
    static const int dump_mode = []() {
        const char* v = std::getenv("NINFER_GRAPH_NODES");
        return v == nullptr ? 0 : std::atoi(v);
    }();
    if (dump_mode >= 1) {
        std::size_t nodes = 0;
        if (cudaGraphGetNodes(definition.graph_, nullptr, &nodes) == cudaSuccess) {
            std::fprintf(stderr, "[graph-nodes] %zu nodes\n", nodes);
            std::fflush(stderr);
        }
        if (dump_mode >= 2 && nodes > 0) {
            std::vector<cudaGraphNode_t> list(nodes);
            if (cudaGraphGetNodes(definition.graph_, list.data(), &nodes) == cudaSuccess) {
                std::map<std::string, std::size_t> tally;
                std::size_t nonkernel = 0;
                for (const cudaGraphNode_t node : list) {
                    cudaGraphNodeType type = cudaGraphNodeTypeKernel;
                    if (cudaGraphNodeGetType(node, &type) != cudaSuccess ||
                        type != cudaGraphNodeTypeKernel) {
                        ++nonkernel;
                        continue;
                    }
                    cudaKernelNodeParams params{};
                    if (cudaGraphKernelNodeGetParams(node, &params) != cudaSuccess ||
                        params.func == nullptr) {
                        ++nonkernel;
                        continue;
                    }
                    const char* mangled = hipKernelNameRefByPtr(params.func, nullptr);
                    std::string name = mangled != nullptr ? mangled : "<null>";
                    // Demangle for readability; fall back to the mangled form.
                    int status = -1;
                    char* demangled = abi::__cxa_demangle(name.c_str(), nullptr, nullptr, &status);
                    if (status == 0 && demangled != nullptr) {
                        name = demangled;
                        std::free(demangled);
                    }
                    // 去参数列表,保留函数名与模板头;先剥 "(anonymous namespace)"(匿名
                    // 命名空间的核名字里含它,按 '(' 截断会剁成 "ninfer::ops::detail::",
                    // 306 节点失名的真凶,2026-10-09)。
                    const std::string kAnon = "(anonymous namespace)";
                    for (std::size_t pos = name.find(kAnon); pos != std::string::npos;
                         pos = name.find(kAnon)) {
                        name.erase(pos, kAnon.size());
                    }
                    const std::size_t paren = name.find('(');
                    if (paren != std::string::npos) { name = name.substr(0, paren); }
                    ++tally[name];
                }
                for (const auto& entry : tally) {
                    std::fprintf(stderr, "[node-tally] %6zu  %s\n", entry.second,
                                 entry.first.c_str());
                }
                std::fprintf(stderr, "[node-tally] non-kernel/unknown nodes: %zu\n", nonkernel);
                std::fflush(stderr);
            }
        }
    }
}

void DecodeGraphExecutable::update(const DecodeGraphDefinition& definition) {
    if (!ready() || !definition.ready()) {
        throw std::logic_error("CUDA Graph update requires a definition and executable");
    }

    cudaGraphExecUpdateResultInfo result{};
    const cudaError_t err = cudaGraphExecUpdate(exec_, definition.graph_, &result);
    if (err != cudaSuccess || result.result != cudaGraphExecUpdateSuccess) {
        // 清掉失败 update 的粘滞 last-error,否则它会被后续无关节点的
        // CUDA_CHECK(cudaGetLastError()) 捞出来(recurrent.cu:127 实录)。
        (void)cudaGetLastError();
        // gfx1100 port:跨形状更新(注释在案的 batch 变化实测,以及 window/smem
        // 变化的同族场景)在 ROCm 上注定 update 失败。降级为重实例化:语义等价
        // (exec_ 由新定义重建),代价 = 一次慢于 update 的实例化。调用方在发射
        // 间隙切换 profile,无 in-flight hazard。
        instantiate(definition);
    }
}

void DecodeGraphExecutable::upload(cudaStream_t stream) {
    if (!ready()) { throw std::logic_error("cannot upload an empty CUDA Graph executable"); }
    CUDA_CHECK(cudaGraphUpload(exec_, stream));
}

void DecodeGraphExecutable::launch(cudaStream_t stream) {
    if (!ready()) { throw std::logic_error("cannot launch an empty CUDA Graph executable"); }
    CUDA_CHECK(cudaGraphLaunch(exec_, stream));
}

bool DecodeGraphExecutable::ready() const noexcept { return exec_ != nullptr; }

void DecodeGraphExecutable::reset() noexcept { destroy_graph_exec(exec_); }

} // namespace ninfer

# NInfer-AMD-ROCm

A research port of the [NInfer](https://github.com/Neroued/ninfer) single-GPU inference engine to **AMD gfx1100 (Radeon RX 7900 XTX / XT, RDNA 3)** on Linux + ROCm.

The tree is based on [Don-Chad/ninfer-3090](https://github.com/Don-Chad/ninfer-3090) v0.6.1 (itself a community retarget of the upstream RTX 5090 engine to sm_86). The port keeps CUDA sources untouched and bridges them with a `compat/cuda` HIP shim, so the diff against the base fork stays reviewable and rebaseable.

## What runs on one RX 7900 XTX today

- **Qwen3.8-27B** official groupwise-int artifacts (v2 container, revision `eec39564`) and **Bonsai-2-27B ternary (PQ2)** artifacts: text generation and perplexity
- OpenAI- and Anthropic-compatible `serve` app with graph-captured decode (hipGraph), concurrent cohorts and prefix reuse
- **INT8 KV cache** routing: the largest single win of the port so far
- **MTP speculative decoding** (d2..d5), numerically correct, with acceptance telemetry
- Ternary PQ2 dot-product path via `sudot4`, plus a decode-graph node-fusion ("megakernel") series that cut per-step graph nodes from 1371 to 970

Headline numbers (single RX 7900 XTX, greedy, INT8 KV): **45.8 t/s serve / 47.3-48.8 t/s CLI** non-speculative decode. Tables, methodology and trajectory: [BENCHMARKS.md](BENCHMARKS.md). Port architecture and platform constraints: [docs/amd-port.md](docs/amd-port.md).

## Status

Research snapshot `v0.1.0`. gfx1100 is the only tested target (2x RX 7900 XTX, distro ROCm HIP 7.1 / clang 21, Ubuntu). Community project maintained on a best-effort basis, like its base fork: issues and PRs welcome, support not guaranteed.

## Build

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DNINFER_BACKEND=HIP -DCMAKE_HIP_ARCHITECTURES=gfx1100 \
  -DCMAKE_CXX_SCAN_FOR_MODULES=OFF
ninja -C build -j 12
```

- `CMAKE_CXX_SCAN_FOR_MODULES=OFF` is mandatory: C++20 module scanning otherwise triggers full-rebuild loops.
- The `w8_small_t.cu` translation unit is the heaviest in the tree (hours on a 12-thread machine); it is not a hang.
- Tested against a distro ROCm install (HIP headers under `/usr`, no `/opt/rocm`), HIP 7.1 / clang 21.

## Run

```bash
# CLI generation
HIP_VISIBLE_DEVICES=0 ./build/apps/ninfer /path/to/model.ninfer \
  --prompt "..." --max-context 8192 --max-new 256 --kv-dtype int8

# Perplexity
./build/apps/ninfer-perplexity /path/to/model.ninfer --text corpus.txt \
  --context 2048 --stride 1024 --kv-dtype int8

# Serve (OpenAI/Anthropic-compatible); see docs/serving.md for the full flag surface
./build/apps/ninfer-serve /path/to/model.ninfer --kv-dtype int8 ...
```

This fork reads the official **v2** container (revision `eec39564`) and community-built Bonsai ternary artifacts; the upstream **v3** container is not supported by this tree. Model cards: `model-cards/`.

## Known limitations

- MTP verify rounds serialize on GDN (linear-attention) layers. With the current round cost, speculative decoding nets at or slightly below non-speculative decode on short-context workloads. Correctness is unaffected; optimization is ongoing.
- DFlash2 speculative mode requires artifact payloads that current public artifacts do not carry.
- No FP4/FP8 tensor-core paths (RDNA 3 has none); NVFP4 targets are excluded from the HIP build.
- The approximate-attention "MK4" fast path (+7.4% decode) is lossy (4 keys share one softmax probability) and excluded from all published baselines.

## Attribution and license

Apache-2.0: see [LICENSE](LICENSE) and [NOTICE.md](NOTICE.md). Lineage: [Neroued/ninfer](https://github.com/Neroued/ninfer) -> [Don-Chad/ninfer-3090](https://github.com/Don-Chad/ninfer-3090) -> this port. The base fork's README is preserved at [docs/README-3090-base.md](docs/README-3090-base.md).

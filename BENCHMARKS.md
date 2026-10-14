# Benchmarks

All numbers: single Radeon RX 7900 XTX (gfx1100, 24 GB), Ubuntu 26.04, distro ROCm (HIP 7.1.52801, clang 21), greedy decoding, INT8 KV cache, graph-captured decode, Bonsai-2-27B ternary PQ2 artifact unless noted. Medians of repeated runs on an otherwise idle GPU.

## Current baseline (2026-10-14)

| Path | Mode | tok/s | Notes |
|---|---|---|---|
| CLI | no speculation | 47.3-48.8 | 970 nodes per decode step, ~20 ms/step |
| CLI | MTP d2 | 27.52 | 1.81 accepted tokens/round |
| serve | no speculation | 45.77 | single-request stream |
| serve | MTP d2 | 26.26 | 2.50 tokens/round, 75% acceptance |

## Trajectory of the port

| Date | CLI nospec tok/s | Main change |
|---|---|---|
| 2026-09-20 | 5.0 | first correct end-to-end decode (cp_async shim degrades to sync copies; mma emulated via shuffle+FMA) |
| 2026-09-21 | 33.4 | first performance pass on the decode path |
| 2026-10-07 | 42.1 | INT8 KV cache routing: routes decode attention to lean/SIMT kernels under the 32 KB LDS launch-tax threshold; biggest single step of the port |
| 2026-10-08 | 46.5 | decode-graph node fusion M1/M2 series (segment merges seg2..4) |
| 2026-10-09 | 47.3-48.8 | rotate+quant fusion v2 (1227 -> 970 nodes/step) |

## Same-card context (other stacks, for calibration)

| Stack | Mode | tok/s | Source |
|---|---|---|---|
| PrismML-Eng/llama.cpp (ternary) | tg128, no speculation | 57.77 | self-measured on the same card |
| kvmem-llama.cpp `kvmem-bonsai` branch | single stream, no speculation | 50.2 | SakuraSoul-Leaf/TernaryBonsai-KVMem-ROCm README |
| same | MTP draft=1 | 61.0 | ditto |
| same | 2-context aggregate | 74.2 | ditto |

Hardware context: RX 7900 XTX memory bandwidth (~960 GB/s) is on par with RTX 3090 (~936 GB/s). The remaining gap to 3090-class Bonsai results (94-171 t/s no-speculation, content dependent) is software structure (kernel dispatch overhead, serial GDN verification, GEMV layout residue), not bandwidth.

## Methodology notes

- Greedy decoding everywhere; stochastic sampling swings roughly +/-10% on this engine and is not used for measurement.
- "Acceptance" = accepted tokens per verify round.
- The approximate-attention MK4 fast path is lossy and excluded from every baseline above.
- The INT8 KV cache is enabled in every number above (`--kv-dtype int8`).

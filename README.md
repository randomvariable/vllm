# vLLM — homelabs fork

This is a [vLLM](https://github.com/vllm-project/vllm) fork whose default branch
**`homelabs-main`** is the union of everything needed to build and run a
**super-optimised vLLM on homelab hardware**. It deliberately diverges from
upstream `main`; the upstream README is preserved at [`VLLM.md`](VLLM.md).

## Targets

| Target | GPU | Backend | Runtime image |
|---|---|---|---|
| **DGX Spark** | Grace Blackwell, sm_121a (arm64) | CUDA 13 | `vllm-spark-runtime` |
| **Strix Halo** | AMD Radeon 8060S, gfx1151 / RDNA3.5 (x86_64, 40 CU, UMA) | ROCm 7.14 | `vllm-strix-runtime` |

One branch builds both. The build recipes live in [`homelab/`](homelab/):

- [`homelab/spark.Dockerfile`](homelab/spark.Dockerfile) — CUDA 13 / sm_121a
- [`homelab/strix.Dockerfile`](homelab/strix.Dockerfile) — ROCm 7.14 / gfx1151

Both are BuildKit-cache-optimised, two-stage `builder → runtime` Dockerfiles that
`COPY` this source tree (no inner clone) and carry **no infrastructure specifics**
— the source ref and the (strix) ROCm base image are `ARG`s.

## Building

The Dockerfiles build straight from a checkout of this branch:

```sh
# DGX Spark (CUDA)
docker build -f homelab/spark.Dockerfile -t vllm-spark-runtime:local .

# Strix Halo (ROCm) -- needs /dev/kfd + /dev/dri at *run* time, not build time
docker build -f homelab/strix.Dockerfile -t vllm-strix-runtime:local .
```

Automated builds run from the private
[`vllm-runtime`](https://github.com/randomvariable/vllm-runtime) repo, whose
Tekton pipelines pull this branch and push the images to the homelab registry.

## What's carried (vs upstream)

**gfx1151 / Strix Halo enablement & tuning** (the bulk of the divergence):

- RDNA3 W4A16 GPTQ GEMM kernels enabled on gfx1151 (`#46186`), with the scalar
  W4A16 path dispatched to gfx1151 (WMMA prefill stays gfx1100-only — its kernel
  is tuned for 96 CU, gfx1151 is 40 CU).
- AITER enabled on gfx1151 (Triton flash-attention — explicitly Strix-Halo-tuned,
  ~1.5–1.9× decode — unified-attention, RMSNorm/RoPE, BF16 MoE). FP8/FP4 and MLA
  stay gated off per-op (gfx1151 has no FP8 tensor cores).
- A pure-Triton `topk_softmax` fallback so MoE routing works on AITER-less ROCm
  (gfx1151), where the only other implementations are AITER (disabled) and the
  CUDA-only `_moe_C` op.
- gfx1151 fused_moe autotuned configs, `-ffast-math` on the HIP build, AMD APU
  UMA VRAM reporting from sysfs, KFD-topology GPU detection, consumer-RDNA
  encoder-cache-profiling skip, and the AITER LDS-overflow → Triton fallback.
- ROCm build requirements bumped to 7.14 / torch 2.12.0+rocm7.14.0 (the release
  HSA runtime initialises the 8060S cleanly; the nightly HSA segfaults in
  `GpuAgent::InitDma`).
- The FlashInfer + serving runtime set (incl. the cu130 JIT cache, nixl, ray,
  tiktoken) inlined into `requirements/cuda.txt` so the spark image installs
  straight from the file with no build-time patching.

**GB10 / laguna / poolside** fork work (the original `gb10-main` lineage) is
retained.

## Branch strategy

- **`homelabs-main`** (default) — the consolidated, easy-to-build homelab branch.
  Diverges from upstream by design.
- **Upstream contributions** — individual patches are kept upstream-compliant and
  proposed from their own branches based off `upstream/main`, not from
  `homelabs-main`.

## Provenance

`homelabs-main` was formerly `gb10-main` (renamed once it covered both DGX Spark
and Strix Halo). Each carried change is a granular commit citing its source
(upstream PR or llama.cpp/kyuz0 optimisation) with AI-assistance attribution.

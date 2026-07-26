# vLLM for Homelabs

A [vLLM](https://github.com/vllm-project/vllm) fork tuned to serve large
language models fast on homelab GPUs. The default branch **`homelabs-main`**
builds one serving stack for two hardware targets and ships the optimisations
that make each of them run well.

| Target | GPU | Backend | Runtime image |
|---|---|---|---|
| **NVIDIA DGX Spark** | Grace Blackwell, `sm_121a` (arm64) | CUDA 13 | `vllm-spark-runtime` |
| **AMD Strix Halo** | Radeon 8060S, `gfx1151` / RDNA3.5 (x86_64, 40 CU, 96 GB UMA) | ROCm 7.14 | `vllm-strix-runtime` |

The upstream vLLM README is preserved at [`VLLM.md`](VLLM.md).

## Quick Start

Build the image for your hardware straight from a checkout of this branch, then
serve a model. The Dockerfiles `COPY` this source tree (no inner clone) and are
BuildKit-cache-optimised two-stage `builder → runtime` builds.

### Step 1: Build the Runtime Image

DGX Spark (CUDA):

```sh
docker build -f homelab/spark.Dockerfile -t vllm-spark-runtime:local .
```

Strix Halo (ROCm). The build is host-agnostic; the GPU (`/dev/kfd`, `/dev/dri`)
is only needed at *run* time:

```sh
docker build -f homelab/strix.Dockerfile -t vllm-strix-runtime:local .
```

### Step 2: Serve a Model

```sh
docker run --rm -it \
  --device /dev/kfd --device /dev/dri \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -p 8000:8000 \
  vllm-strix-runtime:local \
  serve Qwen/Qwen3-8B --host 0.0.0.0 --port 8000
```

Query the OpenAI-compatible endpoint:

```sh
curl http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model": "Qwen/Qwen3-8B", "messages": [{"role": "user", "content": "Hello!"}]}'
```

## Served Models

Any model vLLM supports will run; these are validated and tuned on both targets
as representative examples:

- **Qwen 3.6** (GDN hybrid MoE — interleaved full + Gated DeltaNet linear
  attention) — dense 27B and 35B-A3B, BF16 and quantised.
- **GPT-OSS** — MoE with SwiGLU activation (MXFP4 on CUDA, W4A16/BF16 on ROCm).
- **GGUF models** via the bundled [GGUF plugin](#gguf).

## Optimizations

What `homelabs-main` adds over upstream, framed by the benefit:

**Strix Halo (gfx1151) enablement and tuning** — the bulk of the divergence:

- **AITER on gfx1151** — AMD's Strix-Halo-tuned Triton flash-attention
  (~1.5–1.9× decode), unified-attention, RMSNorm/RoPE, and BF16 MoE. FP8/FP4 and
  MLA stay off (gfx1151 has no FP8 tensor cores).
- **Working MoE routing** — a pure-Triton `topk_softmax` so MoE serves on
  AITER-less ROCm builds.
- **W4A16 GPTQ on gfx1151** — native HIP scalar W4A16 GEMM for quantised dense
  models.
- **Autotuned fused-MoE configs, `-ffast-math` HIP build, APU UMA VRAM
  reporting, KFD-topology GPU detection** — and other RDNA3.5 build/runtime
  fixes so the 8060S initialises and serves cleanly.

**Speculative decoding and runtime knobs** (opt-in):

- DSpark confidence-scheduled verification, online INT8 W8A8 MoE, GDN FLA
  prefill fusions, and runtime tuning env vars (NUMA binding, mmap control,
  parallel weight loading). Each is gated behind an env var and off by default.

**GGUF:**

- The [vLLM GGUF plugin](https://github.com/vllm-project/vllm-gguf-plugin) is
  bundled with the runtime images, so GGUF checkpoints serve out of the box.

Optimisation kernels ported from [ROCm/ATOM](https://github.com/ROCm/ATOM) are
kept (attributed, MIT) under [`homelab/atom_reference/`](homelab/atom_reference/)
as reference material.

## Documentation

- [`homelab/`](homelab/) — the build recipes (`spark.Dockerfile`,
  `strix.Dockerfile`) and ATOM reference kernels.
- [`docs/`](docs/) — upstream vLLM documentation (usage, serving, contributing).
- Automated builds are produced by an external CI harness that pulls this
  branch.

## For Contributors

`homelabs-main` deliberately diverges from upstream `main` — it is the union of
everything needed for an easy, fast homelab build. Individual optimisations are
kept upstream-compliant and proposed to `vllm-project/vllm` from their own
branches based off `upstream/main`, never from `homelabs-main`.

`homelabs-main` was formerly `gb10-main`, renamed once it covered both DGX Spark
and Strix Halo. Each carried change is a granular commit citing its source
(upstream PR, llama.cpp, ATOM, or ROCmFPX) with AI-assistance attribution.

## License

Apache 2.0, as upstream vLLM. Bundled ATOM reference kernels are MIT
(© Advanced Micro Devices). See [`LICENSE`](LICENSE).

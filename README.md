# vLLM for Homelabs

A [vLLM](https://github.com/vllm-project/vllm) fork tuned to serve large
language models fast on homelab GPUs. The default branch **`homelabs-main`**
builds one serving stack for two hardware targets and ships the optimisations
that make each of them run well.

| Target | GPU | Backend | Runtime image |
| --- | --- | --- | --- |
| **NVIDIA DGX Spark** | Grace Blackwell, `sm_121a` (arm64) | CUDA 13 | `vllm-spark-runtime` |
| **AMD Strix Halo** | Radeon 8060S, `gfx1151` / RDNA3.5 (x86_64, 40 CU, unified memory) | ROCm 7.14 | `vllm-strix-runtime` |

Strix Halo is a unified-memory APU: the GPU draws from system RAM rather than
dedicated VRAM, and how much is actually usable depends on your BIOS/GTT
configuration (on our own hardware the GTT pool reports roughly 62 GiB usable).
vLLM reads the sysfs GTT pool rather than the small HIP-reported VRAM aperture,
so it sizes the KV cache against the real budget.

The upstream vLLM README is preserved at [`VLLM.md`](VLLM.md).

## Quick Start

Build the image for your hardware straight from a checkout of this branch, then
serve a model. The Dockerfiles `COPY` this source tree (no inner clone) and are
BuildKit-cache-optimised two-stage `builder → runtime` builds.

### Step 1: Build the Runtime Image

DGX Spark (CUDA), built natively on an arm64 machine:

```sh
docker build -f homelab/spark.Dockerfile -t vllm-spark-runtime:local .
```

DGX Spark, cross-compiled from an x86_64 machine. The builder stage runs on
amd64 and cross-compiles to aarch64, which is how the Spark images are normally
produced — use this unless you are building on Spark hardware itself:

```sh
docker build -f homelab/spark-cross.Dockerfile -t vllm-spark-runtime:local .
```

Strix Halo (ROCm). The build is host-agnostic; the GPU (`/dev/kfd`, `/dev/dri`)
is only needed at *run* time:

```sh
docker build -f homelab/strix.Dockerfile -t vllm-strix-runtime:local .
```

#### Interactive toolbox variants

`homelab/spark-toolbox.Dockerfile` and `homelab/strix-toolbox.Dockerfile` layer a
TUI model launcher, Hugging Face download tooling, and GPU diagnostics on top of
the corresponding runtime image, for standalone `docker run -it` use. They keep
the headless contract — given arguments, or with no TTY, they exec `vllm "$@"`
just like the base image. Point `BASE_IMAGE` at the runtime image you built:

```sh
docker build -f homelab/spark-toolbox.Dockerfile \
  --build-arg BASE_IMAGE=vllm-spark-runtime:local \
  -t vllm-spark-toolbox:local .
```

Use a runtime image for serving, and a toolbox image when you want an
interactive shell with model-management tooling in it.

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

Any model vLLM supports will run; these are the representative examples this
fork is exercised against:

- **Qwen 3.6** (GDN hybrid MoE — interleaved full + Gated DeltaNet linear
  attention) — dense 27B and 35B-A3B, BF16 and quantised.
- **GPT-OSS** — MoE with SwiGLU activation (MXFP4 on CUDA; W4A16/BF16 on ROCm,
  where the Strix Halo path is a pending migration rather than a validated one).
- **DeepSeek-V4** — sparse MLA on DGX Spark, via the vendored FlashInfer fork.
- **GGUF models** via the bundled
  [GGUF plugin](https://github.com/vllm-project/vllm-gguf-plugin).

## Optimizations

What `homelabs-main` adds over upstream, framed by the benefit:

**Strix Halo (gfx1151) enablement** — the bulk of the divergence:

- **AITER on gfx1151** — AMD's Strix-Halo-tuned Triton flash-attention,
  unified-attention, RMSNorm/RoPE, and BF16 MoE. FP8/FP4 and MLA stay off
  (gfx1151 has no FP8 tensor cores). We have not benchmarked the speedup
  ourselves; treat AMD's own performance claims as unverified here.
- **Working MoE routing** — a pure-Triton `topk_softmax` so MoE serves on
  AITER-less ROCm builds.
- **W4A16 GPTQ on gfx1151** — native HIP scalar W4A16 GEMM for quantised dense
  models.
- **`-ffast-math` HIP build, APU UMA memory reporting (sysfs GTT rather than the
  HIP VRAM aperture), KFD-topology GPU detection** — and other RDNA3.5
  build/runtime fixes so the 8060S initialises and serves cleanly.

No fused-MoE kernel configs for the 8060S were tuned here. The two
`Radeon_8060S_Graphics` int4_w4a16 configs in the tree come from upstream — one
is already in upstream `main`, the other is a carried upstream pull request.
Strix Halo kernel tuning has not been done yet.

**DGX Spark (sm_121a) enablement:**

- **Native SM120/SM121 CUTLASS FP8 grouped MoE** — a grouped-GEMM MoE path built
  for the Blackwell consumer/Spark targets. Ported and building; **not yet
  benchmark-qualified**, so it is not claimed as a performance win.
- **Vendored FlashInfer fork** — carries a DeepSeek-V4 sparse-MLA `TOPK=256`
  decode fix, built as a submodule so the Spark images ship working sparse-MLA
  kernels.
- **B12X native FP4 MoE** — an MXFP4 and NVFP4 expert backend for `sm_121a`,
  capability-gated and selectable with `--moe-backend b12x`. NVFP4 consumes
  ModelOpt checkpoints directly. It leads NVFP4 auto-selection on `sm_121a`,
  and is distinct from `--moe-backend flashinfer_b12x`, which uses
  FlashInfer's vendored copy of the kernels. It declines models that need a
  SWIGLU clamp rather than silently dropping the clamp.
- **DeepGEMM cross-build support** — the vendored DeepGEMM host extension builds
  against AArch64 Python/Torch/CUDA instead of silently packaging an x86-64
  binding into the arm64 wheel.

**Memory control for unified-memory devices:**

- **`--gpu-memory-utilization-gb`** — an absolute per-worker GiB budget for total
  engine residency, as an alternative to the fractional
  `--gpu-memory-utilization`. On GB10 and Strix Halo the GPU and host share one
  physical pool, so a fraction of total device memory is a host-unsafe control.
  See [Conserving Memory](docs/configuration/conserving_memory.md#gpu-memory-budget).
  This is allocation planning, not an allocator-enforced quota; real-hardware
  unified-memory residency validation is still outstanding.

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
  `spark-cross.Dockerfile`, `strix.Dockerfile`, and the two `*-toolbox`
  variants) and ATOM reference kernels.
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

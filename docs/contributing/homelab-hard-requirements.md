# Homelab Hard Requirements

Mandatory rules for the `homelabs-main` fork, each one learned the hard way.
They override convenience. Violating them has caused broken images and wasted
multi-hour build cycles.

## FlashInfer is MANDATORY in DGX Spark (CUDA sm_121a) images

DGX Spark serving runs on FlashInfer sm_12x kernels — `deepseek-v4-flash-dspark` (sparse MLA) is the current deployment, and any NVFP4 or AWQ MoE model landing on Spark needs the same kernel surface. A FlashInfer-less DGX image is pointless for this hardware. **Never strip `flashinfer-python` / `flashinfer-cubin` / `flashinfer-jit-cache` from `requirements/cuda.txt` at image-build time.**

## Verify dependency claims by actual resolution, with the repo's indexes

Before declaring two packages incompatible — especially before baking an exclusion into a Dockerfile — run a real resolution using the EXACT `--extra-index-url` lines from the repo's `requirements/*.txt`:

```bash
uv pip compile requirements/cuda.txt --index-strategy unsafe-best-match
```

Package-metadata reading alone is insufficient: constraints move across versions and indexes. An incompatibility claim not reproduced by a resolver with the correct indexes is not a fact — do not act on it. `flashinfer-cubin` and `flashinfer-jit-cache` are **NOT on PyPI**; they exist only on the flashinfer.ai index declared in `requirements/cuda.txt`, so a resolution run without it fails misleadingly and makes the pinned FlashInfer look incompatible with torch/CUDA when it is not.

## Verify every `VLLM_*` env var against `vllm/envs.py` before use

Docker `ENV` entries that do not exist in `vllm/envs.py` are silent no-ops. Example that bit us: `VLLM_USE_FLASHINFER` does NOT exist; the sampler gate is `VLLM_USE_FLASHINFER_SAMPLER` (default `True`, `envs.py`). Grep `vllm/envs.py` for the exact variable name before adding it to any Dockerfile, script, or deployment manifest.

## `README.md` must not overclaim

The fork `README.md` is a public, user-facing document: no node names, registry hosts, or cluster/CI specifics. Keep it current when fork capabilities, supported hardware, or build commands change, and never claim work that has not been done — carried-upstream configs are not "ours", ported-and-building is not "performance-qualified", and a number nobody measured is not a benchmark. Correct or remove a claim as soon as it stops being true.

## Dockerfiles are build-only

`homelab/*.Dockerfile` must BUILD only — no verification assertions, probes, or gate checks in the build path, which fail correct builds on technicalities (e.g. `grep -q` on CMakeCache.txt key formats, readelf arch checks, zipfile membership checks). Verification is a runtime concern: run it against the deployed image on real hardware.

## No performance regression in production migrations

When migrating a production serving deployment to a new image/build, the target must **match or beat** the current deployment's performance. A migration that boots on a slower fallback path (e.g. Triton instead of the tuned kernel backend) is not done — it is a correctness baseline only. Establish the performance-parity requirement BEFORE planning the migration, identify exactly which kernel/backend delivers the current performance, and gate the swap on matching it. (2026-07-26: "i won't accept any regression in performance.")

## Both targets must run every attention type on the most optimized path

CUDA `sm_121a` (DGX Spark / GB10) and ROCm `gfx1151` (Strix Halo) must each run EVERY attention type on the most optimized available path, not merely "run at all". In scope: dense causal GQA/MQA, sliding-window/global hybrids, attention sinks, MLA and sparse/compressed MLA, and linear/recurrent state (Gated DeltaNet, Mamba2, Kimi Delta Attention).

- A model that serves only by falling back to a slower generic backend is a GAP to close, not "support".
- When triaging optimization work, prefer items that close a target × attention-type gap.
- Do NOT dismiss an optimization because it benefits only one of the two targets.

## Ports must be upstream-compatible

Any code ported into this fork from another fork/overlay (e.g. aidendle94 DSV4, bjk110, ATOM) must be written in **upstream-compatible style**: follow the target area's existing upstream patterns (oracle enums/mappings, capability gating, `is_supported_config`/`_supports_current_device` probes, optional-dependency probes, envs.py declarations), so the work could be proposed as an upstream PR. No hacky fork-only patches, no divergent one-off wiring. (2026-07-26: "make any ports upstream compatible.")

## When a deployment is idle, swap and iterate live

If the user says a production deployment is not in use, treat the migration as a live test loop: swap to the new image immediately and iterate on the real deployment until it works, rather than staging a separate canary. Rollback is the manifest revert. (2026-07-26: "completely swap out and test until we get working.")

## Rebase on upstream at least daily, and before implementation work

Keep `homelabs-main` close to `upstream/main` so fork changes stay small, mergeable, and always land on current upstream code.

- Rebase onto `upstream/main` **at least once per working day**, and **before starting any new implementation/fixer work** on a feature.
- Procedure: confirm the `upstream` remote points at `vllm-project/vllm`, then `git fetch upstream && git rebase upstream/main`, then force-push with lease (`git push --force-with-lease`).
- Resolve conflicts by **preserving fork-unique work** — the SM120/SM121 CUTLASS grouped-MoE port, the vendored FlashInfer submodule and its build wiring, B12X MXFP4 integration, DeepGEMM/Spark cross-build changes, and the `homelab/` Dockerfiles. Never drop these to make a rebase "clean".
- If a rebase hits non-trivial conflicts, stop and resolve them with fork context rather than blindly taking upstream or fork sides.

## Compiled extensions live in the checkout, not the image

The gfx1151 devtools image has an **empty** `/src/vllm`; build extensions land
in the host checkout. First check that the devloop container bind-mounts the
checkout. A missing `vllm._C` can have other causes, so then verify with the
sanctioned explicit `.venv/bin/python` or devtools interpreter. Prefer the
Strix devloop's mandatory `cargo make doctor`, `cargo make setup`,
`cargo make build`, and `cargo make test` sequence. For other incremental
C++/HIP/CUDA work, follow the [Incremental Build Setup](./incremental_build.md)
guidance rather than duplicating its CMake commands.

## Optimise the development lifecycle daily, and build incrementally

A slow edit-test loop is treated as a defect, not a cost of doing business. Rebuilding a whole image to test a source change is never acceptable as the routine path. (2026-07-30: "the development practice should force fast incremental build whenever we need to... aggressively optimise the development lifecycle daily.")

- Use the [Incremental Build Setup](./incremental_build.md) guidance for C++/HIP/CUDA changes. Strix work uses the mandatory cargo-make devloop above.
- **Review the loop daily**, alongside the upstream rebase above. If any routine step has become slow, fix the tooling before continuing feature work.
- **Environment staleness must fail closed.** A harness that cannot find the source it is meant to test, or that silently falls back to an installed package or a fallback kernel, must error rather than report success. Prove *which* files a run actually loaded — resolved package path and compiled extension imports — before trusting a pass. Test harnesses that overlay source into a container must run pytest with `--import-mode=importlib`, otherwise the mount shadows the installed package and its compiled extensions vanish silently. Devloop probes and pytest Python invocations must use safe-path mode (`python -P`) in addition to pytest `--import-mode=importlib`, so mounted checkout metadata cannot shadow image metadata.
- **Do not add another language** to solve build orchestration. This tree is already Python, C++/HIP/CUDA and Rust; dev tooling belongs in Rust (`cargo-make`, already a dependency via `setuptools-rust`). A hermetic build system such as Bazel or Pants is not justified by polyglot pain alone: the recurring costs here have been SDK packaging, cross-compilation toolchain files and artefact staleness, none of which it addresses, and it would fight the ROCm SDK's layout harder than CMake does.

## Keep `.git` and doc trees out of the Spark build context

CI clones this fork fresh for every run, so a root `.git` in the build context
differs byte-for-byte between runs even when the commit is identical
(`FETCH_HEAD`, reflogs, pack checksums). With `.git` in context the
`COPY . /src/vllm` layer digest changed every build, which invalidated every
compile layer beneath it: on 2026-08-12 that was 67 min for the vLLM wheel and
118 min for the FlashInfer AOT set, on every single build, including rebuilds
of an unchanged commit.

`.dockerignore` therefore excludes `.git`, `.github`, `docs`, `examples` and
`benchmarks`, and the two things the build genuinely needed from git arrive as
build args: `VLLM_SOURCE_COMMIT` and `VLLM_SCM_VERSION`
(`0.1.dev1+g<9-char sha>`, reproducing what setuptools-scm derived from the
shallow clone). **Do not reintroduce `.git` into the context** to "fix"
versioning — set the build args instead.

Related ordering rule: the FlashInfer AOT stage sits **before** the full source
`COPY`, with narrow copies of `third_party/flashinfer` and
`tools/flashinfer-build.sh`. It has no other dependency on the repository, and
placing it after the full copy meant any edit anywhere in the fork rebuilt its
~2 h layer.

## ccache only survives a clean buildkitd shutdown

BuildKit persists a cache mount when buildkitd releases the ref; a run that is
cancelled or evicted loses that window and the next daemon start deletes the
orphaned mutable refs. That is why `/vllm-spark-ccache-cross` kept reporting
`0 files` at stage entry while `//root/.cache/uv` — untouched, because its
layer was cached — survived from July. The pipeline allows a 120 s termination
grace period so the step's trap can stop buildkitd cleanly. Prefer letting a
bad build fail on its own over cancelling it, and expect a cold compile after
any cancellation.

`CCACHE_DEPEND=true` is set because both compile paths already emit dependency
files (CMake `-MD`, FlashInfer `nvcc --generate-dependencies-with-compile`). On
the 2026-08-12 build, 2740 of 2755 FlashInfer lookups fell out of direct mode
into the slower preprocessed lookup.

## What the FlashInfer AOT matrix actually buys (measured)

`tools/flashinfer-build.sh` runs `python3 -m flashinfer.aot` with no arguments,
so it builds FlashInfer's full default config: fp16 **and** bf16, four FA3 and
three FA2 head-dim pairs, sliding-window and logits-soft-cap variants, plus the
Gemma (head_dim 256), OAI-OSS (head_dim 64 + SWA), XQA and comm kernel sets —
3413 compile units, the single largest stage in the build. The Dockerfile
strips the `flashinfer-jit-cache==` pin from `requirements/cuda.txt` and
installs the wheel this stage produces, so these units are what ships.

Measured on a live `deepseek-v4-flash` rank 0, only three FlashInfer native
modules were mapped, and only two of them came from that wheel:

| Loaded module | Origin |
| --- | --- |
| `sampling.so`, `trtllm_utils.so` | the AOT wheel built here |
| `sparse_mla_sm120.so` | JIT-compiled at runtime into `FLASHINFER_WORKSPACE_BASE` |

The MoE path does not use FlashInfer at all. `--moe-backend flashinfer_b12x`
resolves to the vendored `third_party/b12x` submodule — the logs show
`Using 'B12X_MXFP4' Mxfp4 MoE backend`, `Using B12xExperts`, and CuTeDSL
kernels such as `W4A16FusedMoeKernel` compiled during inference.

**JIT fallback is enabled**, not disabled: `jit_monitor_mode` is `warn`, and
both `sparse_mla_sm120` and the CuTeDSL MoE kernels are compiled at runtime in
the normal course of serving. A kernel absent from the AOT set therefore costs
a first-call compile spike, not a crash. Do not justify the matrix size by
claiming a missing kernel is a hard failure — that is measurably untrue.

The matrix is nonetheless kept full **by decision**: this is the shared homelab
Spark runtime rather than a per-model artifact, and the alternative to build
minutes on an otherwise idle builder is latency spikes on live traffic whenever
a new shape appears. Attack build time through caching and parallelism, which
are model-independent, rather than by narrowing the kernel matrix.

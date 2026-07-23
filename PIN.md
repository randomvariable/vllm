# Fork base pin

**Base SHA:** `e5949f10009c8b1803e2e37f5610b4dd047d432f`
**Upstream context:** vLLM tag `v0.26.0rc1` (release candidate commit on
`vllm-project/vllm`, tagged 2026-07-23). Chosen as a recent, tagged,
known-stable-looking main-line commit per the standing directive to anchor the
fork on vLLM tip-of-tree.
**Branch:** `gb10-main` (replaces the misleading `homelab-v0.25.1` name; the old
branch was never on the v0.25.1 tag — it sat on `752a3a504`, PR #48330 bugfix).
**Date pinned:** 2026-07-24

## Custom commits carried on top of the base (cherry-picked, in order)

1. `build: add native SM121 targets` — adds `12.1` to `CUDA_SUPPORTED_ARCHS`
   (+ build.sh / docker-bake / versions.json / flashinfer-build.sh arch refs).
2. `fix: clamp UMA CUDA graph estimate` — negative-cudagraph-estimate clamp
   (`max(..., 0)`) via `get_device_memory_info()`; DGX Spark GB10 UMA free
   memory can rise during CUDA graph capture (project memory #3655/#3679).
3. `fix: complete UMA memory and CUDA gating port` — adds
   `get_device_memory_info` helper to `vllm/utils/mem_utils.py`, gating port.
4. `fix: port UMA handling to V2 runner` — ports the same UMA clamp to the V2
   `vllm/v1/worker/gpu/model_runner.py`.

All four cherry-picked cleanly onto `v0.26.0rc1` (no conflicts).

## Pin-advance cadence

"Track main" means: use vLLM main as the source of truth for *what* to pin, then
pin a specific tagged/known-good SHA — do NOT auto-float to tip. Advance the pin
deliberately, re-validate the GB10 build/serve surface after each advance, and
update this file (SHA, tag context, date) each time.

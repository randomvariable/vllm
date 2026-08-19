# GEMM workspace address stability under CUDA graphs

Branch: `fix/gemm-workspace-address-stability`
Upstream: [vllm-project/vllm#52553](https://github.com/vllm-project/vllm/pull/52553)
(root-causes SM120 NVFP4 serving crashes in upstream #52540, and some of #34948)

## Problem

A CUDA graph captures the **raw device pointer** of whatever workspace buffer
was current at capture time. Anything that moves a workspace after capture
turns every subsequent replay into a use-after-free: the caching allocator has
already handed the freed block to unrelated tensors, so kernels read scratch
from — and write scratch over — live data.

Two such moves exist:

1. **`SM100Workspace.ensure_size`** calls `self._workspace_buf.resize_()`.
   `resize_()` frees the old storage and moves the buffer.
2. **FlashInfer's cuDNN GEMM execute paths** grow their shared 32 MiB cache
   workspace with `resize_()` when an execution plan asks for more. Fixed
   upstream in [flashinfer-ai/flashinfer#4553](https://github.com/flashinfer-ai/flashinfer/pull/4553),
   which is not in our pinned release, so vLLM needs a guard.

Symptom is an illegal memory access or misaligned address, or a wedged GPU,
minutes into serving — not at capture. Upstream reports a 256-byte overflow
past the 32 MiB default during capture being enough.

## Fix

**Retire, never resize.** `ensure_size` allocates a new buffer and keeps the
old one alive in `_retired_bufs`. Graphs captured against a retired buffer stay
correct because its capture-time size was sufficient for the shapes baked into
those graphs.

**Pre-grow FlashInfer's shared buffer** before any capture, in
`kernel_warmup`, so it never has to move later.

Both allocate with `zeros`, not `empty`: split-KV and cuDNN split-K plans keep
semaphore/accumulator regions in the workspace and expect them zeroed. A fresh
`cudaMalloc` happens to return zeroed pages, but a recycled dirty block from
the caching allocator does not, and kernels then spin or corrupt.

## Applicability here

`SM100Workspace` in this fork matches upstream's pre-state exactly, including
the `resize_()` call. The FlashInfer half targets SM120 NVFP4 directly, which
is our serving path.

## Tests

`tests/v1/attention/test_gemm_workspace_stability.py`:

- `presize_flashinfer_gemm_workspaces` is a no-op when FlashInfer is absent,
  requests `zero_init`, and falls back cleanly on an older signature that does
  not accept it — all runnable on CPU with a stubbed module.
- The workspace-retirement invariant (old buffer stays alive and keeps its
  device pointer across a grow) is GPU-gated: `SM100Workspace` allocates on
  `"cuda"` in its constructor, matching upstream, so this one runs on the
  image build rather than locally.

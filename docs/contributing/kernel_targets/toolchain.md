# Toolchain, Target Suffixes and Dispatch Gating

A kernel that is written correctly, tiled correctly and validated correctly can
still be absent at runtime, because it was compiled for a target the device does
not load or gated behind a capability test that does not fire. This page covers
the build and dispatch layer for both targets, and the specific ways it goes
wrong in this fork.

Facts are labelled **Confirmed** (vendor documentation with a citation, or code
in this tree with a path) or **Hypothesis** (inference not yet verified here).

## The two failure modes

They look identical from the outside and have completely different fixes.

1. **Wrong binary.** The kernel was compiled for an architecture the device
   cannot load. The gate in source may be perfectly correct; there is simply no
   code. Fixed in the build system.
2. **Wrong gate.** The binary exists, but a capability check in Python or C++
   selects a different path. Fixed in source.

The diagnostic that separates them: check whether the compiled artefact contains
code for the device's architecture at all. If it does, the problem is the gate.

## NVIDIA: target suffixes

**Confirmed.** The target suffix decides which devices a binary will run on, and
the naming is not self-explanatory:

| Suffix | Meaning | Runs on |
| --- | --- | --- |
| none, e.g. `compute_120` | Baseline features only | 12.0 and later |
| `f`, e.g. `compute_120f` | Family-specific | 12.0 and 12.1 |
| `a`, e.g. `compute_121a` | Architecture-specific | 12.1 only |

**Confirmed.** Each is a superset of the one above it in features and a subset in
portability; `compute_100f` correspondingly covers 10.0 and 10.3.
Architecture-specific targets appeared with 9.0, family-specific ones with 10.0.

**Configuration (in tree).** The vLLM wheel, GGUF plugin wheel, runtime stage,
and native `homelab/spark.Dockerfile` set `TORCH_CUDA_ARCH_LIST="12.1a"`. The
FlashInfer cross wheel additionally retains `12.0f` with `12.1a`, because its
CUDA 12.9+ runtime resolver requires both AOT module variants.

**Confirmed.** An `a` target will not load on a device of any other minor
capability: a `12.1a` binary does not run on SM120 at all. This is a deployment
failure distinct from a capability gate being wrong in source — the gate can be
correct and the kernel still absent. If you need one artefact to cover both,
`12.0f` is the family target that spans 12.0 and 12.1.

### Diagnosing a missing CUDA kernel

```bash
# What architectures does this object actually contain?
cuobjdump --list-elf <path-to-.so-or-.o>

# What does the device report?
nvidia-smi --query-gpu=compute_cap --format=csv
```

If the reported capability has no matching entry in the ELF list, you have a
build problem, not a gating problem.

## AMD: architecture strings

**Confirmed (in tree).** `CMakeLists.txt` lists the supported HIP architectures
in `HIP_SUPPORTED_ARCHS`, which includes `gfx1151` alongside the gfx9, gfx103x,
gfx110x, gfx115x and gfx12xx families.

**Confirmed (in tree).** Architecture gating happens at translation-unit
granularity in the build. The RDNA3-family scalar W4A16 GEMM sources
(`csrc/rocm/q_gemm_rdna3.cu`, `csrc/rocm/moe_q_gemm_rdna3.cu`) are added when
`VLLM_GPU_ARCHES` matches `gfx1100|gfx1151`, while the WMMA prefill translation
unit (`csrc/rocm/q_gemm_rdna3_wmma.cu`) is added only for `gfx1100`.

That asymmetry is worth internalising: **being in the arch list does not mean
every kernel is built for you.** A file-level `if(VLLM_GPU_ARCHES MATCHES ...)`
is invisible from Python, and the resulting absence looks exactly like a runtime
gate rejecting the device.

### Diagnosing a missing HIP kernel

```bash
# What architectures does this object contain?
roc-obj-ls <path-to-.so>

# What does the device report?
rocminfo | grep -i gfx
```

Then check `CMakeLists.txt` for a target-conditional block around the source
file, before assuming the problem is in Python.

## Runtime gating in this tree

**Confirmed (in tree).** ROCm capability predicates live in
`vllm/platforms/rocm.py`. The ones that matter for gfx1151:

| Predicate | Behaviour for gfx1151 |
| --- | --- |
| `on_gfx1151()` | True (`_ON_GFX1151` tests for `gfx1151` in the GCN arch string) |
| `supports_fp8()` | False — returns true only for gfx9 and gfx12x |
| `is_fp8_fnuz()` | False — tests for `gfx94`, so gfx1151 gets `torch.float8_e4m3fn` |
| `supports_mx()` | False — tests for `gfx95` |
| `use_custom_allreduce()` | False — tests for `gfx94`/`gfx95` |

**Confirmed (in tree).** `vllm/_aiter_ops.py` gates AITER availability on
`on_mi3xx() or on_gfx1151()`, with its docstring recording the boundary: AITER's
Triton flash-attention, unified-attention, RMSNorm/RoPE and BF16 MoE kernels
have gfx1151 paths as of ROCm 7.14, while FP8/FP4 and MLA remain gated off
per-op.

**Confirmed (in tree).** CUDA-side capability gating for the Blackwell split is
overwhelmingly `capability.major == 10`, which is the correct test for anything
requiring SM100 features and is *false* on `sm_121a`. Instances include
`vllm/v1/attention/backends/fa_utils.py` (FA4 selection),
`vllm/v1/attention/backends/flashinfer.py`, and the MLA backends under
`vllm/v1/attention/backends/mla/`.

### Writing a gate that is right on both targets

The recurring mistake is gating on generation rather than on capability.

| Intent | Wrong test | Right test |
| --- | --- | --- |
| "Needs `tcgen05.mma`" | "is Blackwell" | `capability.major == 10` |
| "Blackwell or newer" | `capability.major == 10` | `capability >= (10, 0)` |
| "Has FP8 tensor cores" (ROCm) | "is ROCm" or "is RDNA3.5" | `current_platform.supports_fp8()` |
| "Has a matrix unit" (ROCm) | `on_mi3xx()` | An explicit predicate covering both MFMA and WMMA parts |
| "Can run this kernel at all" | Architecture family | A `can_implement()` returning a reason string |

**Confirmed (in tree).** The `can_implement()` pattern is already used for the
RDNA3 W4A16 kernel: `vllm/model_executor/kernels/linear/mixed_precision/`
`rdna3_w4a16.py` returns `(False, "RDNA3 W4A16 kernel requires gfx1100 or
gfx1151")` when the predicate fails. A gate that returns a reason is
dramatically easier to debug than a boolean, because the fallback path can log
*why*.

### Gate widening is the dangerous edit

The single most tempting change when a kernel does not dispatch on gfx1151 is to
widen the capability predicate. Sometimes that is correct. Often it is how a
silent numerical failure enters the tree — most sharply for FP8, where
[the encoding differs](numerics.md#fp8-two-problems-not-one) and admitting
gfx1151 to an `e4m3fnuz` path produces wrong numbers rather than slow ones.

Before widening a gate, answer:

1. What hardware capability does this gate actually stand for?
2. Does gfx1151 have that capability, or does it have something adjacent?
3. If adjacent — does the kernel's arithmetic depend on the difference?
4. What reference comparison will demonstrate that it does not?

If you cannot answer 4, do not widen the gate.

## Compiler flags and resource introspection

### Reading resource usage

This is the highest-value diagnostic on the ROCm side, because
[hardware counters are unavailable](memory_model.md#profiling-on-this-target).

```bash
# Report per-kernel VGPR/SGPR/LDS usage at compile time.
hipcc -Rpass-analysis=kernel-resource-usage ...
```

The same information is in the compiled object's metadata (`.vgpr_count`,
`.sgpr_count`, `.group_segment_fixed_size`). Feed the VGPR count into the
occupancy arithmetic in
[Execution model](execution_model.md#occupancy-is-a-staircase-not-a-slope);
the raw count is not the number that matters, the block-rounded quotient is.

On the CUDA side the equivalent is `nvcc -Xptxas -v` or `--resource-usage`, and
the occupancy consequence is smoother because registers are not block-allocated
the same way.

### Wave size selection

**Confirmed.** A shader is compiled for one wave size and runs at that size
(ISA §2.1); see
[Execution model](execution_model.md#wave-size-is-a-build-decision). The
toolchain flag is `-mwavefrontsize64` (and its negation), and HIP exposes the
compiled size through `warpSize` — which is *not* a compile-time constant across
architectures on ROCm, unlike CUDA's 32.

**Practical consequence.** Code ported from CUDA that hardcodes 32 in reduction
loops, mask widths or shared-memory sizing will be wrong if built for wave64 and
right if built for wave32, with no diagnostic either way. Search ported kernels
for literal 32s that mean "warp width".

### Triton on this target

**Confirmed (in tree).** Triton is a first-class path here, not a fallback of
last resort: this fork's online INT8 MoE dispatches to AITER's Triton kernels
(`vllm/model_executor/layers/quantization/online_int8_moe.py`), and the
mixed-precision W4A16 kernels carry gfx1151-specific autotune tables
(`vllm/model_executor/kernels/linear/mixed_precision/triton_w4a16.py`,
`rdna_hybrid_w4a16.py`).

Two Triton-specific traps on gfx1151:

- **Sub-16 tile dimensions fail codegen** with `no matching matrix core
  intrinsic for wmma version 1`. See
  [Matrix units](matrix_units.md#the-16-element-floor).
- **Autotune tables do not transfer between architectures.** The tuned tables in
  this tree are annotated with the part they were tuned on — gfx1151 (40 CUs,
  32-wide wavefronts) and gfx1201 (32 CUs, 32-wide wavefronts) have separate
  entries for good reason. Importing a gfx942 table wholesale imports its
  sub-16 dimensions along with everything else.

## Environment-variable gating

**Confirmed (in tree).** Optional paths on the ROCm side are gated behind
`VLLM_ROCM_*` environment variables declared in `vllm/envs.py`, defaulting off.
The online INT8 MoE path is gated behind `VLLM_ROCM_USE_AITER_ONLINE_INT8_MOE`.

**Confirmed (per this repository's contribution rules).** Environment variables
must be declared in `vllm/envs.py` and documented wherever env vars are listed,
and any new user-facing option needs both a Google-style docstring on the config
field (which generates the reference page) and prose in the guide that owns the
area.

**Practical consequence for kernel work.** A new target-specific kernel path
almost always wants an env-var opt-in during its first release, because:

- It lets the default stay conservative while the numerics are still being
  validated.
- It makes A/B benchmarking trivial and honest — the same binary, two
  configurations, which is exactly what the
  [parity control](verification.md#include-control-configurations) needs.
- It gives operators an escape hatch when a model hits a shape the kernel
  handles badly.

Document the mutual exclusion explicitly. If your new path conflicts with an
existing one, say which wins and why, and cross-reference from the established
option so readers find the alternative.

## Confirming what actually ran

**This is the step most often skipped and most often decisive.** Both targets
silently fall back: gated features select generic paths, and the kernel you
believe you are benchmarking may not be the kernel that ran.

Ways to confirm dispatch, roughly in order of cost:

1. **Log the selection.** Most backends in this tree log the chosen path at
   INFO. Raise the log level and read it.
2. **Return a reason from the gate.** A `can_implement()` returning
   `(False, "<reason>")` tells you immediately which condition failed.
3. **Kernel trace.** `rocprofv3 --kernel-trace` on ROCm, or the PyTorch profiler
   / Nsight Systems on CUDA (see [profiling](../profiling.md)). The kernel names
   in the trace are ground truth.
4. **Deliberate breakage.** Temporarily make the kernel you expect to run fail
   loudly. If nothing fails, it was not running.

!!! danger "A missing kernel and a slow kernel look the same"
    Per this fork's standing requirements, both targets must run every attention
    type on the most optimized available path. A model that serves only by
    falling back to a slower generic backend is a gap to close, not support.
    When you find a kernel dispatching to a generic fallback on one of the
    targets, that is a finding — not the resolution.

## Checklist

- [ ] Does the compiled artefact contain code for this device's architecture?
- [ ] Is there a target-conditional block in `CMakeLists.txt` excluding the
      source file?
- [ ] Does the runtime gate test a capability or a generation?
- [ ] If widening a gate: what reference comparison proves the numerics still
      hold?
- [ ] Does the kernel hardcode 32 anywhere that means "wave width"?
- [ ] Is the autotune table tuned for this architecture, or imported?
- [ ] Is any new env var declared in `vllm/envs.py` and documented in prose?
- [ ] Have you confirmed which backend actually ran, rather than which one you
      intended?

## Related pages

- [Porting workflow: SM120 to gfx1151](porting_sm120_to_gfx1151.md)
- [Execution model: warps, waves and launch geometry](execution_model.md)
- [Matrix units: `mma.sync` versus WMMA](matrix_units.md)
- [Numerics, atomics and validation](numerics.md)
- [SM120/SM121 constraints](sm120.md)
- [Verifying kernel correctness and performance](verification.md)

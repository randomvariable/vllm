# Porting Workflow: SM120 to gfx1151

Most kernels in vLLM are written NVIDIA-first. This page is the ordered
procedure for bringing one of them to AMD Strix Halo (`gfx1151`, RDNA3.5) from
NVIDIA GB10 (`sm_121a`, consumer-class Blackwell), plus a catalogue of the
failure modes that recur.

It assumes you have read
[Execution model](execution_model.md), [Matrix units](matrix_units.md),
[Memory model](memory_model.md) and [Numerics](numerics.md). Those pages explain
*why*; this one is *what to do, in what order*.

Facts are labelled **Confirmed** (vendor documentation with a citation, or code
in this tree with a path) or **Hypothesis** (inference not yet measured here).
Section numbers of the form `ISA §x.y` refer to AMD's *"RDNA3.5" Instruction Set
Architecture Reference Guide* (23 July 2024), which is not vendored here.

## Before you write anything

Per this repository's contribution policy, check for duplicate work first:

```bash
gh issue view <issue_number> --repo vllm-project/vllm --comments
gh pr list --repo vllm-project/vllm --state open --search "<issue_number> in:body"
gh pr list --repo vllm-project/vllm --state open --search "<short area keywords>"
```

If an open PR already addresses the same port, do not open another.

Then answer three questions about the source kernel, because the answers decide
whether a port is even the right shape of work:

1. **What capability was it exploiting?** If the answer is FP8 tensor cores,
   block scaling, TMA or structured sparsity, there is no equivalent on gfx1151
   and the port is a *reimplementation* with different numerics, not a
   translation. Say so up front.
2. **What is the shape regime?** A prefill-shaped GEMM and a decode-shaped GEMV
   have different right answers on this target, and the matrix unit is not
   automatically the winner for the latter. See
   [Matrix units](matrix_units.md#non-matrix-paths-worth-knowing).
3. **What is it competing against?** If a generic Triton path already serves
   this operation on gfx1151, the bar is beating that path, measured
   [at saturation with a parity control](verification.md).

## Step 1: Inventory the source kernel

Write down, explicitly, what the SM120 kernel assumes. Most of these assumptions
are implicit in the source and become visible only when they break.

| Assumption to extract | Where to find it | Why it matters |
| --- | --- | --- |
| Warp width | Reduction loops, mask literals, `32` constants | Becomes wave32 or wave64 (ISA §2.1) |
| Block dimensions | Launch config | Must be a multiple of wave size |
| Shared memory per block | `extern __shared__`, static arrays, `cudaFuncSetAttribute` | Ceiling drops to 64 KB (ISA §1.2.2.1) |
| Registers per thread | `-Xptxas -v`, `__launch_bounds__` | Occupancy becomes a staircase |
| Matrix instruction shape | `mma.sync` shape, fragment types, CUTLASS tile | Becomes 16×16×16 (ISA §7.9) |
| Operand layout | Fragment layout code, TN/NN assumptions | Must be re-derived for WMMA |
| Data types | Accumulator and operand types | FP8/FP6/FP4 unavailable |
| Scale-factor scheme | Block-scaled types, SF vector size | No hardware equivalent |
| Async copy structure | `cp.async`, TMA, producer warps | Becomes a VGPR-staged software pipeline |
| Cache hints | `.cs`, `.lu`, `__ldg` | Becomes `SLC`/`DLC` (ISA §4.1.1) |
| Denormal mode | `-ftz`, `.ftz` modifiers | Becomes `MODE.fp_denorm` (ISA §13.2) |
| Atomic use | `atomicAdd` on float | Different denormal and rounding rules (ISA §13.1, §13.2) |
| Bounds handling | Predication, `if (idx < n)` | Hardware returns zero instead of faulting (ISA §9.4.1) |

The output of this step is a written list. It is the input to every subsequent
step, and it is what you will put in the PR description.

## Step 2: Decide the target shape before writing code

Do the arithmetic on paper first. It is much cheaper than discovering the tile
does not fit after the kernel is written.

1. **Pick the wave size.** Default to wave32: it matches warp32 semantics most
   closely and is the only mode where dual-issue VALU is legal (ISA §7.6).
2. **Size the LDS tile against 64 KB.** Not 128 KB (that is per WGP, not per
   work-group), and not the SM120 figure (ISA §1.2.2.1, §12.1).
3. **Decompose into 16×16×16 steps.** Every tile dimension must be a multiple of
   16 (ISA §7.9, Table 33). Check that the accumulator fragment size per
   work-item is something you can afford.
4. **Estimate the VGPR count** — accumulators, staging registers, addresses,
   indices — and round it up to the 24-register allocation block for wave32 on a
   1536-VGPR SIMD. Divide the register file by the result. That quotient is your
   waves per SIMD (ISA §3.3.2.1).
5. **Sanity-check latency hiding.** If step 4 gives you 3 waves per SIMD and the
   kernel is memory-bound on system memory, the tile is too big. Shrink and
   iterate.

Only now start writing.

## Step 3: Port the structure, not the code

Work outward from the matrix operation.

1. **Matrix op.** Use `rocwmma` or Triton's `tl.dot` rather than hand-placing
   fragments. If you must go to the intrinsic level, re-derive the layout — do
   not translate the `mma.sync` mapping. AMD's Matrix Instruction Calculator
   (<https://github.com/RadeonOpenCompute/amd_matrix_instruction_calculator>,
   referenced from ISA §7.9) generates the element-to-register mapping when the
   vendor guide is unavailable.
2. **Staging.** Replace the `cp.async`/TMA pipeline with an explicit
   load-to-VGPR, wait, store-to-LDS, barrier sequence. Keep the depth shallow
   initially; deepen only if measurement justifies the register cost.
3. **Swizzle.** Re-derive against 64 DWORD banks (ISA §12.1), for both the store
   and load sides. Do not port the constants.
4. **Addressing.** Use address-space-qualified pointers so loads lower to
   `global_load_*`, not `flat_load_*` (ISA §11.1). Verify in the disassembly.
5. **Epilogue.** If the source used block scaling, this is where the software
   scale application lands.
6. **Bounds handling.** Keep explicit predication. Do not rely on the hardware's
   read-zero behaviour (ISA §9.4.1).

## Step 4: Gate it correctly

Write the capability gate to express what the kernel actually needs, and make it
return a reason. See
[Toolchain](toolchain.md#writing-a-gate-that-is-right-on-both-targets). Check
whether the source file needs a target-conditional block in `CMakeLists.txt`,
and whether the arch is in `HIP_SUPPORTED_ARCHS`.

If the path is new and its numerics are not yet fully validated, gate it behind a
`VLLM_ROCM_*` environment variable defaulting off, declared in `vllm/envs.py`
and documented in prose.

## Step 5: Validate before you optimise

Do not tune a kernel that is not yet known to be correct — you will tune it into
a local minimum around a bug.

Apply the [verification bar](verification.md) in full:

- Compare against a float64 computation or an independent implementation, with a
  tolerance derived from the precision claimed, decided *before* the comparison.
- Include shapes that are not multiples of the tile size and assert on the
  masked region.
- Include the numerics-specific tests in
  [Numerics](numerics.md#test-the-paths-that-silently-degrade).
- For selection kernels, compare selected indices, not scores.
- Confirm which backend actually ran.

## Step 6: Measure honestly

Both GPUs ramp from an idle clock floor, which invalidates the obvious
benchmarking method. Interleave arms within a single measurement loop, include a
parity control and a known-slow arm, and measure at saturation. The full
methodology is in
[Verification](verification.md#performance-interleave-the-arms); it applies
unchanged here and is not repeated.

Two additions specific to a port:

- **Benchmark against the path that currently serves**, not against the SM120
  kernel's throughput. An FA4 or SM100 number this target cannot reach is not a
  baseline, it is a distraction.
- **Report the occupancy arithmetic alongside the timing.** With hardware
  counters unavailable (see
  [Memory model](memory_model.md#profiling-on-this-target)), the VGPR count and
  its block-rounded quotient are the only occupancy evidence you have. Include
  them.

## Step 7: Document in the same change

Docs ship with the code, never as a follow-up. A new user-facing option or
behaviour needs prose in the guide that owns the area plus a Google-style
docstring on the config field, and any new environment variable must be declared
in `vllm/envs.py` and documented where env vars are listed.

For a kernel port specifically, record:

- Which capability the source kernel used and what replaced it.
- The numeric consequences, if the format changed.
- The shape regime it is good for, and the one it is not.
- What it falls back to when the gate rejects the device.

## Failure-mode catalogue

Ordered by how often they occur in practice, not by severity.

### Sub-16 tile dimension

**Symptom.** Triton codegen fails with `no matching matrix core intrinsic for
wmma version 1`. The message names the intrinsic, not the config, so it reads
like a backend bug.

**Cause.** A tile dimension below 16, frequently inherited from an MI300X
autotune table where gfx9's MFMA accepts more shapes. This fork hit it with a
config carrying `BV=8`.

**Fix.** Audit every dimension in every autotune configuration. All must be
multiples of 16 (ISA §7.9, Table 33).

### Tile does not fit LDS

**Symptom.** Launch failure, or the compiler reporting an LDS allocation over
the limit.

**Cause.** Tiled against SM120's 99 KB per block, or against gfx1151's 128 KB
per-WGP figure rather than the 64 KB per-work-group limit.

**Fix.** Re-tile against 64 KB (ISA §1.2.2.1, §12.1).

### Wrong fragment layout

**Symptom.** No error. Output is the right shape, the right rough magnitude, and
wrong.

**Cause.** The `mma.sync` lane-to-element mapping was carried over. WMMA requires
lanes 0-15 replicated into 16-31 (and into 32-47, 48-63 under wave64) for A and
B, with A column-major and the others row-major in the canonical 16×16×16 layout
(ISA §7.9).

**Fix.** Use a fragment API, or re-derive from the Matrix Instruction Calculator.
Catch it with a reference comparison.

### Missing WMMA hazard spacing

**Symptom.** Correct at small K, wrong at large K. Or correct in a debug build,
wrong with optimisation.

**Cause.** Back-to-back dependent WMMA instructions where the first's D matrix
overlaps the second's A or B, without an intervening independent VALU
instruction or `V_NOP` (ISA §7.9.1). This is a correctness requirement, not a
performance one.

**Fix.** Insert the spacing. Only applies to hand-written assembly or inline asm;
compiler-generated code handles it.

### FP8 gate widened

**Symptom.** No error. Model quality degrades; activations look plausible.

**Cause.** A capability gate widened to admit gfx1151 to an FP8 path. gfx1151 has
no FP8 tensor cores, and where FP8 is handled at all it uses `e4m3fn`, not
gfx942's `e4m3fnuz` — the same bit pattern denotes a different number
(`vllm/platforms/rocm.py`).

**Fix.** Do not widen the gate. Re-quantise to a supported type and re-derive the
scale factors. See [Numerics](numerics.md#fp8-two-problems-not-one).

### Block-scale group size mismatch

**Symptom.** No error. Magnitudes are perturbed in a way that looks like a
tolerance problem.

**Cause.** Scale factors grouped at the wrong granularity when reimplementing
block scaling in software — `nv_float4_t` uses vector size 16 while the `mx_*`
family uses 32, and sparse variants double it.

**Fix.** Verify the group size against
[SM120 constraints](sm120.md#block-scaled-and-narrow-precision-types) and test
with a tensor whose scale factors vary sharply between adjacent groups.

### Out-of-range reads passing silently

**Symptom.** Correct at tile-aligned shapes, subtly wrong at others. Often passes
`allclose` because the errors are small and localised.

**Cause.** Out-of-range buffer loads return zero rather than faulting, and
multi-DWORD accesses are checked per DWORD so a partial overrun returns a mix of
real and zero data (ISA §9.4.1). LDS behaves the same way (ISA §3.3.4.1).

**Fix.** Explicit predication, plus edge-shape tests that assert on the masked
region rather than on a whole-tensor norm.

### Misaligned LDS access rewriting addresses

**Symptom.** No error, no memory violation, wrong data.

**Cause.** Under the default alignment mode, misaligned LDS accesses have their
low address bits zeroed to force alignment, silently reading or writing
elsewhere; only `UNALIGNED` mode handles them properly, and misaligned atomics
report `MEMVIOL` (ISA §12.7, §3.3.4.1).

**Fix.** Keep the alignment invariant when changing element types. An
FP8-to-BF16 conversion doubles element size and can break a previously aligned
stride.

### Hardcoded warp width

**Symptom.** Wrong results only in a wave64 build. Nothing in a wave32 build.

**Cause.** A literal `32` meaning "warp width" in a reduction, mask or
shared-memory size. HIP's `warpSize` is not a compile-time constant across
architectures the way CUDA's 32 is.

**Fix.** Search ported kernels for literal 32s. Note also that
`ds_permute`/`ds_bpermute` operate across 32 lanes at a time even under wave64,
each half acting as an independent wave32 (ISA §12.5.2), so a butterfly wider
than 32 lanes has no direct translation.

### Barrier neutralised by single-wave groups

**Symptom.** Data-dependent corruption that appears only at some shapes.

**Cause.** Work-groups consisting of a single wave have their barrier
instructions treated as no-ops and allocate no barrier resource (ISA §2.3,
§5.5). A tile configuration that collapses to one wave per group makes every
barrier meaningless.

**Fix.** Do not depend on barriers in configurations that may produce single-wave
groups, or exclude those configurations.

### Flat addressing eating the wait granularity

**Symptom.** No correctness problem. Throughput well below expectation, with
`s_waitcnt 0` scattered through the disassembly.

**Cause.** Generic pointers lowering to `FLAT`, which increments both
`VMcnt`/`VScnt` and `LGKMcnt` and after which the only sensible wait is zero
(ISA §11.1, §11.1.1).

**Fix.** Qualify address spaces so loads lower to `global_load_*`.

### Occupancy cliff from one extra register

**Symptom.** A small, apparently harmless change costs high single digits or
more, but only at production grid sizes.

**Cause.** VGPR count crossed an allocation block boundary, dropping waves per
SIMD (ISA §3.3.2.1). This fork measured 216 → 224 VGPRs costing 4-17% on a
saturated grid, and nothing at low occupancy.

**Fix.** Compute the distance to the next boundary rather than guessing; see
[RDNA3.5 constraints](rdna35.md#vgpr-allocation-granularity-and-occupancy).

### NaN row in attention output

**Symptom.** Whole rows of NaN in attention output on gfx1151 but not on
`sm_121a`.

**Cause.** Usually a NaN entering the tile — an uninitialised LDS region, or a
partial out-of-range load returning a mix of real and zero data (ISA §9.4.1) —
or genuine `-inf` arithmetic: `-INF + INF` is a QNaN under IEEE-754, which a
fully-masked row or a masked seed meeting `+inf` will produce on either target.

**Fix.** Trace where the NaN enters rather than assuming the mask logic is
broken. Note that the atomic min/max NaN ordering in ISA §13.3 is *not* the
cause here — it is scoped to float atomics, not the VALU arithmetic a row max
lowers to. See [Numerics](numerics.md#nan-ordering-in-atomic-min-and-max).

### NaN silently dropped by an atomic reduction

**Symptom.** A quantisation scale or bound looks plausible, but downstream
tensors contain NaN.

**Cause.** Quiet NaN loses the selection in float atomic min and max (ISA
§13.3), so an amax pass built on atomic max discards NaN inputs rather than
reporting them.

**Fix.** Check for NaN explicitly rather than inferring it from a min/max
result.

### Benchmarked the wrong kernel

**Symptom.** A speedup that does not reproduce, or a change with no measurable
effect.

**Cause.** The kernel under test never dispatched; a gate selected a generic
fallback. Both targets fall back silently.

**Fix.** Confirm dispatch before interpreting any number. See
[Toolchain](toolchain.md#confirming-what-actually-ran).

## Quick mapping reference

Consolidated from the individual pages, for use while reading source.

| SM120 construct | gfx1151 equivalent | Caveat |
| --- | --- | --- |
| `threadIdx`, `blockIdx`, `blockDim` | Same names in HIP | Direct |
| `warpSize` (always 32) | `warpSize` (32 or 64, build-dependent) | Not a portable constant |
| `__syncthreads()` | `__syncthreads()` → `s_barrier` | No-op for single-wave groups (ISA §5.5) |
| `__syncwarp()` | Not needed | No independent thread scheduling |
| `__shfl_xor_sync` | `ds_bpermute_b32`, DPP16/DPP8 | 32-lane range even under wave64 (ISA §12.5.2) |
| `__ballot_sync` | `v_cmp_*` to SGPR pair | Width follows wave size |
| `__ldg` / `ld.global.nc` | `GLC`/`SLC`/`DLC` bits | Different semantics (ISA §4.1.1) |
| `cp.async`, TMA | VGPR staging + `ds_store_*` | Costs registers |
| `mma.sync.aligned` | `V_WMMA_*` | 16×16×16 only (ISA §7.9) |
| FP8 / FP6 / FP4 operands | BF16/FP16, or IU8/IU4 + software scale | Different numerics |
| Block scale factors | Software scaling in epilogue | Group boundary is yours to enforce |
| Structured sparsity | Nothing | Densify |
| `atomicAdd` (float) | `global_atomic_add_f32` etc. | Forced flush, forced RNE (ISA §13.1, §13.2) |
| `-ftz=true` | `MODE.fp_denorm` | Atomics ignore it (ISA §13.2) |
| Cluster / distributed shared memory | Nothing | Flatten the algorithm |
| `cudaFuncSetAttribute` for >48 KB shared | 64 KB hard cap per work-group | Re-tile (ISA §1.2.2.1) |
| `compute_121a` target suffix | `gfx1151` arch string | Both are exclusive; neither is a superset |
| `cuobjdump --list-elf` | `roc-obj-ls` | Same diagnostic purpose |
| `-Xptxas -v` | `-Rpass-analysis=kernel-resource-usage` | Occupancy consequence differs |
| Nsight Compute counters | Unavailable on gfx1151 | Kernel trace only |

## PR requirements for a port

Per this repository's contribution policy, AI-assisted work must state in the PR
description why it is not duplicating an existing PR, which test commands were
run and their results, model evaluation results when the change affects output
or serving, and that AI assistance was used. A human submitter must understand
and defend every changed line.

For a kernel port specifically, also include:

- The inventory from step 1 — what the source kernel assumed.
- What replaced each unavailable capability.
- The tolerance you used and its derivation.
- The parity control's result alongside any speedup claim.
- The VGPR count and resulting waves per SIMD.
- Confirmation of which backend actually dispatched during benchmarking.

## Related pages

- [Execution model: warps, waves and launch geometry](execution_model.md)
- [Matrix units: `mma.sync` versus WMMA](matrix_units.md)
- [Memory model, LDS and unified memory](memory_model.md)
- [Numerics, atomics and validation](numerics.md)
- [Toolchain, target suffixes and dispatch gating](toolchain.md)
- [Verifying kernel correctness and performance](verification.md)

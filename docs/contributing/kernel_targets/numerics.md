# Numerics: Atomics, NaN Handling and Narrow Formats

Numerical differences between the two targets do not announce themselves. A
kernel with the wrong FP8 encoding, the wrong NaN ordering in a masked softmax
or the wrong denormal behaviour in a reduction produces output that looks like
output. This page collects the specific divergences that matter for an
SM120-to-gfx1151 port, so that you know what to look for before a model quality
regression tells you.

Section numbers of the form `ISA §x.y` refer to AMD's *"RDNA3.5" Instruction Set
Architecture Reference Guide* (23 July 2024), which is not vendored here.

Facts are labelled **Confirmed** (vendor documentation with a citation, or code
in this tree with a path) or **Hypothesis** (inference not yet measured here).

## Float atomics diverge in three ways

**Confirmed.** Floating-point atomics are available on gfx1151 as LDS, buffer and
flat/global/scratch operations (ISA §13). Three of their behaviours differ from
what a CUDA kernel assumes, and each is easy to mistake for a kernel bug.

### 1. Atomic add flushes input denormals unconditionally

**This differs by address space.** ISA §13.2 tabulates the two separately, and
they do not agree:

| Operation | Cache atomics (buffer, flat, global, scratch) | LDS atomics |
| --- | --- | --- |
| `Add_F32` | Always flush — `MODE.fp_denorm` not used | Follows `MODE` |
| `Min_F32` / `Max_F32` | Follows `MODE` | Follows `MODE` |
| `CmpStore_F32`, `CmpStore_F64` | Follows `MODE` | Follows `MODE` |
| `Min_F64` / `Max_F64` | — | Follows `MODE` |

**Confirmed.** So the unconditional-flush statement applies to **float atomic add
issued to buffer, flat, global or scratch addresses**: that operation is
hardwired to flush input denormals and does not consult `MODE.fp_denorm` (ISA
§13.2). LDS `Add_F32` is *not* in that category — it follows the mode register
like the other LDS float atomics.

**Confirmed.** For memory float atomics generally there is no separate input and
output denormal control: only bit 0 of `sp_denorm` or bit 0 of `dp_denorm` is
considered, and the remaining denormal rules are the same as for LDS (ISA
§13.2).

**Practical consequence.** A reduction that accumulates into LDS and one that
accumulates into global memory can produce different results for the same input,
because only the latter's `Add_F32` forces flushing. A kernel that switches
between the two based on tile size, work-group count or a fallback path inherits
that difference silently — and it will not show up in a comparison that only
exercises one of the paths.

### 2. Atomic rounding is not configurable

**Confirmed.** LDS and memory atomics have the rounding mode for float atomic add
fixed at round-to-nearest-even; the `MODE.round` bits are ignored (ISA §13.1).

Unlike the denormal behaviour above, this one *is* uniform across both address
spaces. Whatever rounding mode the wave has requested, float atomic add will not
honour it.

### 3. Min and max return an unmodified source

**Confirmed.** When denormal flushing is active for comparisons, each input has
its mantissa flushed to zero before the compare if the exponent is zero — but
for min and max the returned value is an *unflushed* copy of the selected input
(ISA §13.2, §13.3). Compare-store flushes the result when input denormal
flushing occurs.

**Confirmed.** Signalling NaNs are converted to quiet NaNs, and LDS raises no
exception or signal on a signalling NaN (ISA §13.3).

### What this means for a port

CUDA's `atomicAdd` on float has its own well-known non-determinism (order of
accumulation), and most kernels are already written to tolerate it. The
gfx1151 additions are:

- The *value* domain differs, not just the order, wherever flushing applies.
- LDS and memory accumulation paths can disagree with each other, because the
  forced flush applies only to the memory-side `Add_F32`.
- Any test that compares an atomic-based reduction bit-for-bit against a
  VALU-based one will fail for reasons that are not bugs.

**Practical consequence.** Where determinism matters, avoid float atomics on both
targets — do a tree reduction in LDS/shared memory and have one wave write the
result. Where you keep atomics, do not write a test that depends on their exact
value.

## NaN ordering in atomic min and max

**Scope first.** These rules are stated in ISA §13.3, which sits inside Chapter
13, *Float Memory Atomics*. They describe the selection behaviour of the float
**atomic** min and max operations issued to LDS, buffer and flat/global/scratch
addresses. They are *not* a statement about the VALU `V_MIN_F32`/`V_MAX_F32`
instructions that ordinary arithmetic — including a softmax row max — lowers to.
Do not carry these rules across to VALU code.

**Confirmed.** For the float atomic operations, quiet NaN is placed at one
extreme of the selection ordering, and the extreme differs between the two
operations (ISA §13.3):

- For **atomic max**, the ordering from smallest to largest starts with QNaN,
  placing it *below* `-inf`.
- For **atomic min**, the ordering from smallest to largest ends with QNaN,
  placing it *above* `+inf`.

Both placements have the same practical effect: **a quiet NaN loses the
selection**. Atomic max returns the non-NaN operand because QNaN sorts lowest;
atomic min returns the non-NaN operand because QNaN sorts highest. A quiet NaN
arriving at either operation is therefore *suppressed*, not propagated — which
is the `minNum`/`maxNum`-style behaviour IEEE-754 describes, not NaN-propagating
behaviour.

**Confirmed.** Signalling NaN is the exception. A signalling NaN input
short-circuits the comparison: the result is a quiet NaN derived from that
source, with the source's bits preserved (ISA §13.3). So sNaN propagates where
qNaN does not.

**Confirmed, and scoped the same way.** ISA §13.3 also specifies float *add*
rules — `-INF + INF` yields a QNaN, `±INF + NaN` copies the NaN input through as
a quiet NaN, `-0 + 0` is `+0`, `INF + (float, ±0)` is `INF` with the sign
preserved, and `NaN + NaN` selects `SRC0`'s NaN converted to quiet. Being in
Chapter 13, these describe the float atomic add path, not VALU addition. The
`-INF + INF` case in particular is ordinary IEEE-754 behaviour that VALU
arithmetic will also exhibit, but do not cite §13.3 as the authority for what a
VALU add does.

**Confirmed.** Compare-swap only swaps when the compare condition is true with
neither source a NaN, treating `+0` and `-0` as equal (ISA §13.3).

### What this actually implies

The consequence is the opposite of the one people expect, and it cuts in a
direction that matters for reductions rather than for softmax.

**A NaN entering a float atomic min or max disappears.** If you implement a
global maximum — an amax pass for a quantisation scale, a running bound, a
reduction across work-groups — using float atomic max, a quiet NaN in the input
leaves no trace in the result. The reduction returns a plausible finite number
and the corrupt input is silently dropped.

That is a *detection* problem, not a propagation problem:

- A NaN-poisoned tensor can pass through an atomic-max amax pass and produce a
  perfectly ordinary-looking scale factor.
- Downstream, that scale is applied to data that still contains the NaN, so the
  failure surfaces somewhere else entirely, with no indication that the
  reduction saw it.
- Signalling NaN behaves differently from quiet NaN here, so whether the
  corruption is visible depends on which kind of NaN upstream code produced.

**Practical consequence.** Do not rely on a float-atomic min/max reduction to
surface NaN contamination — it is specified to hide it. If you need to know that
an input contained NaN, test for it explicitly (an integer-domain check on the
exponent and mantissa bits, or a separate `isnan` reduction) rather than
inferring it from a min/max result.

### What this does *not* imply about masked softmax

The standard masked-softmax idiom seeds masked positions with `-inf` so the
running max ignores them and `exp` drives them to zero. That row max is
ordinary VALU arithmetic; it does not issue float atomics, so ISA §13.3's atomic
selection ordering does not govern it.

If a ported attention kernel produces NaN rows on gfx1151 but not on `sm_121a`,
the atomic NaN ordering is the wrong place to look. The likelier causes, in
rough order of frequency:

- A NaN or uninitialised value entering the tile at all — from an uninitialised
  LDS region, or from a partial out-of-range load returning a mixture of real
  and zero data (see
  [Memory model](memory_model.md#out-of-range-is-the-biggest-silent-failure-source)).
- `-inf` arithmetic producing a genuine NaN. `-INF + INF` is a QNaN under
  IEEE-754, so a masked-position seed of `-inf` reaching an add against `+inf`
  produces NaN regardless of target.
- An entire row masked out, so the row max is `-inf` and the subsequent
  `exp(x - max)` evaluates `-inf - -inf`.
- Denormal-mode differences from the source build changing where underflow
  occurs.

!!! note "Hypothesis — VALU min/max NaN behaviour"
    This page does not state what `V_MIN_F32`/`V_MAX_F32` do with NaN operands,
    because the ordering tables cited above are scoped to Chapter 13's atomic
    operations and the VALU instruction semantics were not established here.
    IEEE-754 leaves `minNum`/`maxNum` NaN behaviour to the implementation and
    PTX's `min.f32`/`max.f32` have their own documented rules, so parity between
    the targets for VALU min/max is unverified. If your kernel can see NaNs in
    production, test that path on both targets rather than assuming either
    parity or a specific ordering.

## Denormal control

**Confirmed.** LDS instructions allow denormals to be passed through or flushed
to zero based on the `MODE.denormal` wave-state register, as with VALU
operations: `denorm_single` affects F32 operations and `denorm_double` affects
F64. LDS instructions use both `FP_DENORM` bits (`allow_input_denormal`,
`allow_output_denormal`) to control flushing of inputs and outputs separately
(ISA §13.2).

**Confirmed.** The 32-bit float adder uses both input and output denormal flush
controls from `MODE`. Float compare, min and max use only the input-denormal
flushing control (ISA §13.2).

**Confirmed.** Memory float atomics have no separate input and output denormal
control: only bit 0 of `sp_denorm` or bit 0 of `dp_denorm` is considered
(ISA §13.2).

**Practical translation.** CUDA exposes flush-to-zero through `-ftz=true` and
per-instruction `.ftz` modifiers, and the default differs by compilation mode.
On gfx1151 the equivalent is `MODE.fp_denorm`, set per wave. The consequences
for a port:

- A kernel compiled with `-ftz=true` on CUDA and ported without setting the
  equivalent denormal mode will produce different results near zero.
- The difference is largest exactly where it is hardest to notice: attention
  scores after masking, softmax tails, and quantisation scale factors near the
  bottom of their range.
- Float atomic add ignores the setting entirely (above), so a kernel mixing
  atomics and VALU arithmetic has two different denormal regimes in one
  reduction.

## Narrow format differences

### FP8: two problems, not one

**Confirmed (in tree).** There are no FP8 tensor cores on gfx1151. WMMA offers
F16, BF16, IU8 and IU4 inputs only (ISA §7.9, Table 33), and
`RocmPlatform.supports_fp8()` in `vllm/platforms/rocm.py` returns true only for
gfx9 and gfx12x, correctly excluding gfx1151.

**Confirmed (in tree).** Where FP8 values are handled at all, gfx1151 uses the OCP
`e4m3fn` encoding, *not* the `e4m3fnuz` ("no infinity, no unsigned zero")
variant used on gfx942. The selection is explicit: `is_fp8_fnuz()` tests for
`gfx94` and only then returns `torch.float8_e4m3fnuz`, otherwise
`torch.float8_e4m3fn`.

The two encodings assign exponent bias and special values differently, so the
same bit pattern denotes a different number.

!!! danger "It fails silently"
    An encoding mismatch does not raise. It produces plausible-looking
    activations that are quietly mis-scaled. Only a reference comparison catches
    it — see [Verification](verification.md).

**Practical consequence for porting from SM120.** SM120 supports FP8 as a tensor
core input type (`float_e4m3_t`, `float_e5m2_t`), so an SM120 kernel may use FP8
end to end. Bringing it to gfx1151 means one of:

1. **Convert at the boundary.** Keep the FP8 checkpoint, dequantise to BF16
   before the matrix operation. Correct, costs bandwidth on the one resource an
   APU has least of.
2. **Re-quantise to INT8.** Use IU8 WMMA with per-channel and per-token scales.
   This fork does exactly this for MoE in
   `vllm/model_executor/layers/quantization/online_int8_moe.py`.
3. **Do not port it.** If the kernel's value was the FP8 tensor core throughput,
   there is nothing to port; the gap is a gap.

Note that option 1 and option 2 have different numerics from the source kernel
and from each other. Both need validation against a reference, not against the
SM120 kernel's output — an SM120 FP8 kernel's output is itself an approximation,
and matching it exactly is neither achievable nor the goal.

### Block scale factors

**Confirmed.** SM120's block-scaled types pair an operand type with a
scale-factor type and a scale-factor vector size — the number of operand
elements sharing one scale factor. The pairings and sizes are tabulated in
[SM120 constraints](sm120.md#block-scaled-and-narrow-precision-types). The key
trap: `nv_float4_t` is not OCP-compliant and uses a different scale factor type
*and* a different vector size (16, not 32) from the `mx_*` family, and sparse
variants double the vector size in every case.

**Confirmed.** gfx1151's WMMA has no block-scaling modifier. See
[Matrix units](matrix_units.md#sm120-block-scaling-has-no-counterpart) for the
substitution options.

**Practical consequence.** When you re-implement block scaling in software, the
group boundary becomes your responsibility. Getting the group size wrong
mis-groups scale factors, which perturbs magnitudes plausibly — the same failure
class as the encoding mismatch above, and equally invisible to a
tolerance-fitted test.

### Integer paths

**Confirmed.** For WMMA and DOT integer forms, `NEG[1:0]` indicates signed (1) or
unsigned (0) per input source rather than meaning negate, and the destination is
signed for the integer types (ISA §7.5, §7.9). Getting the signedness bits wrong
is a silent reinterpretation of the input data.

**Confirmed.** VOP3P provides a `CLMP` (clamp) bit: for float arithmetic it
clamps the result to `[0, 1.0]` with `-0` clamped to `+0`; for signed integer
arithmetic to `[min_int, max_int]`; for unsigned to `[0, max_uint]` (ISA §7.5).

**Practical consequence.** INT8 quantised paths are the most likely destination
for a ported FP8 kernel on this target, which makes signedness and saturation
behaviour a live concern rather than an academic one. Verify that your
quantiser's assumed range (symmetric signed, asymmetric unsigned) matches the
signedness bits the kernel actually sets.

## Exception detection

**Confirmed.** The GPU detects IEEE-754 floating-point exceptions in hardware and
these can be recorded for post-execution analysis (ISA §1.2.1). Trap and
exception registers exist in the wave state (ISA §3.4.9).

**Confirmed.** WMMA generates no ALU exceptions at all (ISA §7.9), and LDS
produces no exception or signal for a signalling NaN (ISA §13.3).

**Practical consequence.** Exception detection is a debugging avenue when chasing
NaN origins, but it will not see anything that happens inside a WMMA. If a
matrix kernel produces NaNs, the exception machinery will point at whatever
consumed the WMMA output, not at the matrix operation itself.

## Validation strategy for numerical ports

The [verification bar](verification.md) applies to all kernel work on both
targets and is not optional. This section is the numerics-specific addition to
it.

### Do not validate against the source kernel

The instinct when porting is to compare the gfx1151 kernel's output against the
`sm_121a` kernel's output. That is the wrong reference for three reasons:

1. The source kernel is itself approximate. Matching it exactly means
   reproducing its errors.
2. If the port changed the numeric format (FP8 to INT8, block-scaled to
   per-channel), the outputs *should* differ. You have no way to tell an
   intended difference from a bug.
3. It gives you no tolerance you can defend. "Close to the CUDA kernel" is not a
   statement about correctness.

Compare against a float64 computation of the same mathematics, or against an
independent implementation, with a tolerance derived from the precision you are
claiming.

### Test the paths that silently degrade

For each of the divergences on this page, there is a test shape that exercises
it:

| Divergence | Test that catches it |
| --- | --- |
| Out-of-range read-zero at tile edges | Shapes that are not multiples of the tile size; assert on the masked region's values, not just the whole-tensor norm |
| FP8 encoding mismatch | Values near the format's max and min normal; round-trip a known bit pattern |
| Denormal flushing in reductions | Inputs whose partial sums land in the denormal range |
| NaN suppressed by atomic min/max | Inject a NaN into a reduction input and assert your explicit NaN check catches it, since the atomic result will not |
| Block-scale grouping | A tensor whose scale factors vary sharply between adjacent groups |
| Integer signedness bits | Inputs spanning the full signed range including negatives |
| Atomic non-determinism | Run twice, assert only what you actually guarantee |

### Selection kernels need rank tests, not closeness tests

For any kernel whose output feeds a selection — top-k, top-p, MoE routing,
speculative-decoding verification, indexer scoring — elementwise closeness is
insufficient, because the failure mode is ranking drift rather than magnitude
error. See
[Verification](verification.md#top-k-style-kernels-elementwise-closeness-is-the-wrong-test)
for the full treatment.

This matters more after a numeric-format port than after a pure layout port,
because re-quantising changes the *ordering* of near-tied scores even when it
preserves their magnitudes to within any tolerance you would pick.

### Run model evals for anything model-affecting

Per this repository's contribution requirements, changes that affect output,
accuracy or serving need model evaluation results in the PR. A numeric-format
port is definitionally model-affecting. Search `tests/evals/` or use
`vllm bench`, and include the results without waiting to be asked.

## Porting checklist

- [ ] Is the FP8 encoding correct for the target, and are scale factors
      re-derived rather than carried over?
- [ ] If block scaling was in the source, what enforces the group boundary now?
- [ ] Are integer signedness bits correct for the quantiser's assumed range?
- [ ] Does the kernel mix float atomics and VALU arithmetic in one reduction? If
      so, is the denormal difference acceptable?
- [ ] Can a NaN reach a float atomic min or max? If so, is there an explicit
      NaN check, given that the atomic will suppress it?
- [ ] Is the reference a float64 computation or an independent implementation —
      not the source kernel's output?
- [ ] For selection kernels, are selected indices compared, not just scores?
- [ ] Have model evals been run and included?

## Related pages

- [Verifying kernel correctness and performance](verification.md)
- [Matrix units: `mma.sync` versus WMMA](matrix_units.md)
- [Memory model, LDS and unified memory](memory_model.md)
- [RDNA3.5 constraints](rdna35.md)
- [SM120/SM121 constraints](sm120.md)

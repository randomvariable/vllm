# Matrix Units: `mma.sync` versus WMMA

The matrix instruction is where an SM120-to-gfx1151 port stops being a rename
exercise. The two targets expose matrix multiply-accumulate through different
instruction families with different operand shapes, different data layouts in
registers, different supported input types and different scheduling hazards.
Almost every other difference in the port — tiling, LDS budgeting, register
pressure — is downstream of this one.

Section numbers of the form `ISA §x.y` refer to AMD's *"RDNA3.5" Instruction Set
Architecture Reference Guide* (23 July 2024), which is not vendored here.
CUTLASS behaviour is from NVIDIA's Blackwell functionality documentation at
<https://docs.nvidia.com/cutlass/latest/media/docs/cpp/blackwell_functionality.html>.

Facts are labelled **Confirmed** (vendor documentation with a citation, or code
in this tree with a path) or **Hypothesis** (inference not yet measured on this
hardware).

## The shape of the problem

| | SM120/SM121 (`sm_121a`) | gfx1151 (RDNA3.5) |
| --- | --- | --- |
| Instruction family | `mma.sync.aligned`, warp-level | `V_WMMA_*`, wave-level (ISA §7.9) |
| Encoding | PTX warp-level MMA | VOP3P (ISA §7.5) |
| Scope | One warp (32 threads) | One wave (32 or 64 work-items) |
| Shapes | A short list per type pair; `128x128x128`, `256x128x128`, `128x128x256` at collective level | Exactly one: 16×16×16 (ISA §7.9, Table 33) |
| Input types | TF32, BF16, FP16, FP8, FP6, FP4, INT8 | F16, BF16, IU8, IU4 (ISA §7.9, Table 33) |
| Accumulator | FP32 (and FP16 paths) | F32, F16, BF16, I32 depending on instruction |
| Operand layout | TN only for narrow precision | Register-resident, replicated across lane halves |
| Block scale factors | Yes (`mxf8f6f4`, `mxf4`, `mxf4nvf4`) | No hardware block scaling |
| Sparsity | Yes (structured sparse variants) | Not in the WMMA table |
| Async data movement | TMA, `cp.async` | None; stage through VGPRs |
| Rounding | Per PTX semantics | Round-to-nearest-even only, for float types (ISA §7.9) |
| FP exceptions | Per PTX semantics | WMMA raises no ALU exceptions (ISA §7.9) |

## The 16-element floor

**Confirmed.** Every WMMA instruction the RDNA3.5 architecture provides has the
same shape: A, B, C and D are all 16×16 matrices, across the F16, BF16, IU8 and
IU4 input types (ISA §7.9, Table 33). There is no narrower matrix instruction to
fall back on.

This is a hard floor, not a performance preference. A matrix operand narrower
than 16 elements has no instruction to lower to, so it fails in the compiler
rather than running slowly.

!!! warning "It fails loudly, and the message does not name your tile"
    A `tl.dot` with any dimension below 16 fails Triton codegen with
    `no matching matrix core intrinsic for wmma version 1`. The error names the
    intrinsic, not the offending config, so it reads like a backend bug.

This fork hit exactly that with an autotune configuration carrying `BV=8`, a
value chosen for MI300X occupancy where gfx9's MFMA unit accepts more shapes.

**Scope the rule correctly.** The constraint is on the **M, N and K dimensions
that are lowered into a WMMA operation** — in Triton terms, the shape of the
`tl.dot` operands. It is not a blanket requirement that every dimension of every
tile or autotune parameter be a multiple of 16. A block size that only controls
how many output elements a work-group is responsible for, a split-K factor, a
number of pipeline stages, or a dimension that is reduced in the VALU rather
than the matrix unit are all unconstrained by ISA §7.9.

What this means when auditing an imported autotune table:

- Identify which parameters feed the matrix operation's operand shape. Those
  must be multiples of 16.
- Parameters that do not reach the matrix unit are free, and forcing them to
  multiples of 16 needlessly narrows the search space.
- The failure is a compile error, so a build sweep over the table will find the
  offending entries — you do not have to reason it out entirely on paper.

A config list that is merely suboptimal on gfx942 can be unbuildable on
gfx1151, so audit before trusting an imported table.

### Translating SM120 tile shapes

**Confirmed.** For SM120 narrow-precision GEMMs the CUTLASS-level MMA tile
shapes are a short list. Across the block-scaled and non-block-scaled tables the
shapes that appear are `128x128x128`, `256x128x128` and `128x128x256`, with
smaller shapes (`64x64x128`, `64x128x128`, `128x64x128`) available for the
non-block-scaled `{float8_t, float6_t, float4_t}` combinations — **TN only** in
every case, with `N` against every other layout combination.

**Confirmed.** Those are collective-level tile shapes, not instruction shapes.
The underlying instruction is `mma.sync.aligned` with the usual warp-level
shapes; the collective tile is what the warp-group cooperates to produce.

**Practical translation.** The right mental model is *not* "SM120 tile 128×128×128
becomes gfx1151 tile 128×128×128". It is:

1. Take the collective tile shape as a statement about how much work one
   work-group does per iteration and how much LDS that requires.
2. Re-derive the LDS footprint under the 64 KB per-work-group cap (ISA §1.2.2.1)
   — not the 100 KB SM120 figure and definitely not SM100's 228 KB.
3. Decompose the resulting tile into 16×16×16 WMMA steps, and check the
   register cost of holding the accumulator fragment.
4. Re-tune. The SM120 shape encodes SM120's shared memory, register file and
   warp-group schedule. None of those transfer.

A `128x128` output tile held in FP32 accumulators is 16384 values per
work-group. Distributed across a 256-work-item group that is 64 VGPRs of
accumulator per work-item before any addressing, staging or index registers.
Against a 256-VGPR ceiling and 24-register allocation blocks
(see [Execution model](execution_model.md#occupancy-is-a-staircase-not-a-slope)),
this is the point where a directly-translated SM120 tile becomes
occupancy-hostile on gfx1151. Expect to shrink tiles and iterate more.

## Data layout inside the matrix instruction

This is the part most likely to produce *plausible but wrong* numbers rather
than a compile error.

**Confirmed.** WMMA instructions run over multiple cycles and internally use the
DOT instructions. To achieve that, data must be arranged so that for the A and B
operands, lanes 0-15 are replicated into lanes 16-31 — and under wave64, also
into lanes 32-47 and 48-63 (ISA §7.9).

**Confirmed.** In the canonical VGPR layout for M = N = K = 16, the A matrix is
column-major while the others are row-major (ISA §7.9).

**Confirmed.** Sources for WMMA must be VGPRs (ISA §7.5, VOP3P field
descriptions: `SRC0`/`SRC1` note "WMMA: must be a VGPR"). Inline constants are
permitted only for the C matrix; for F16 and BF16 an inline value is replicated
into both halves of the DWORD (ISA §7.9).

**Confirmed.** `OPSEL[2]` is used with 16-bit-output WMMA operations to control
whether the C matrix is read from the upper or lower half of the VGPR, and
whether D is stored into the upper or lower half; `OPSEL[0]` and `OPSEL[1]` are
unused for WMMA (ISA §7.5).

**Confirmed.** For the integer forms, the `NEG` field is repurposed: `NEG[1:0]`
indicates signed (1) or unsigned (0) for `SRC0` and `SRC1` rather than meaning
negate, and `NEG[2]`/`NEG_HI` must be zero (ISA §7.5, §7.9). For the float
forms, `NEG[0]` applies to the A matrix and `NEG[1]` to B, with
`{NEG_HI[2], NEG[2]}` acting as `{ABS, NEG}` on C.

### What this means for a port

An `mma.sync` kernel encodes a specific mapping from lane index to matrix
element, usually via a fragment abstraction or hand-written index arithmetic
matching the PTX layout tables. That mapping does not transfer. If you are
writing at the intrinsic level, you must re-derive the layout for WMMA; if you
are using a fragment API (`rocwmma`, or Triton's `tl.dot`), let it do the
mapping and do not hand-place elements.

**Confirmed.** AMD publishes a Matrix Instruction Calculator that generates
element-to-register mappings for the WMMA instructions and reports register
usage and throughput (ISA §7.9 references
<https://github.com/RadeonOpenCompute/amd_matrix_instruction_calculator>). When
you need the exact mapping and the vendor guide is unavailable, that tool is the
authoritative substitute.

!!! danger "Layout errors do not raise"
    A wrong fragment layout produces a matrix product of the wrong thing, not a
    fault. The magnitudes stay plausible, the softmax still normalises, and the
    model still emits text. Only a reference comparison catches it — see
    [Verification](verification.md).

## Scheduling hazards around WMMA

**Confirmed.** Back-to-back dependent WMMA instructions require one `V_NOP` or an
unrelated VALU instruction in between when the first instruction's D matrix
overlaps the second instruction's A or B matrix (ISA §7.9.1). This is required
for *correct function*, not merely for performance.

**Confirmed.** A/B may overlap C as long as C is distinct from D; the typical
case is that C and D are the same (ISA §7.9.1).

**Confirmed.** The same section lists cases that cost only stalls, not
correctness: reusing the previous WMMA's D as the next instruction's C may stall
if the two instructions are not the same WMMA type or if the second uses an
input modifier on `SRC2`; overlapping (rather than identical) VGPRs for that
role may stall; and a VALU instruction reading the previous WMMA's D may stall.

**Practical consequence.** This is a class of bug CUDA does not have. On NVIDIA
the hardware interlocks handle `mma.sync` dependencies. On gfx1151 an
accumulator-chaining loop written in inline assembly, where each WMMA's output
feeds the next iteration's A or B, is *incorrect* without an intervening
instruction — and it will usually appear to work at small K where the loop runs
few iterations.

Compiler-generated code from HIP or Triton handles this. If you are writing
inline asm or patching disassembly, this is the first thing to check when a
matrix kernel gives wrong results under some shapes and right results under
others.

## Narrow precision: the biggest capability gap

### FP8 does not exist as a matrix input on gfx1151

**Confirmed.** WMMA on RDNA3.5 offers F16, BF16, IU8 and IU4 inputs (ISA §7.9,
Table 33). FP8 is not among them.

**Confirmed (in tree).** `RocmPlatform.supports_fp8()` returns true only for
gfx9 and gfx12x, so gfx1151 is correctly excluded — see
`vllm/platforms/rocm.py`. `vllm/_aiter_ops.py` documents the same boundary:
AITER's Triton flash-attention, unified-attention, RMSNorm/RoPE and BF16 MoE
kernels have gfx1151 paths, while FP8/FP4 and MLA remain gated off per-op.

### The FP8 encoding is not the same one

**Confirmed (in tree).** Where FP8 values are handled at all, gfx1151 uses the
OCP `e4m3fn` encoding, not the `e4m3fnuz` variant used on gfx942. The selection
is explicit in `vllm/platforms/rocm.py`: `is_fp8_fnuz()` tests for `gfx94` and
only then returns `torch.float8_e4m3fnuz`, otherwise `torch.float8_e4m3fn`.

The two encodings assign exponent bias and special values differently, so the
same bit pattern denotes a different number. Porting an FP8 kernel from
MI300-class hardware requires re-deriving scale factors and any hand-written
conversion or clamping logic. Widening a capability gate to admit gfx1151
produces numerically wrong output, not slow output.

### SM120 block scaling has no counterpart

**Confirmed.** SM120's narrow-precision `mma` variants carry block scale factors:
scale factors apply along the GEMM-K dimension so that every 16 or 32 elements
of A and B share one scale factor, doubling to 32 or 64 for sparse variants
because sparse GEMM compresses 2× along K.

**Confirmed.** The operand and scale-factor type pairings, and their scale-factor
vector sizes, are tabulated in [SM120 constraints](sm120.md#block-scaled-and-narrow-precision-types).
The `nv_float4_t` type uses a different scale-factor type *and* a different
vector size (16, not 32) from the `mx_*` family, which is the single most
likely thing to be wrong in a port.

**Confirmed.** gfx1151's WMMA has no block-scaling modifier. There is no
instruction-level equivalent.

**Practical translation.** A block-scaled SM120 GEMM does not port to gfx1151 as
a block-scaled GEMM. Your options, in descending order of fidelity:

1. **Re-quantise to a supported type.** BF16 or FP16 WMMA with per-channel or
   per-token scaling applied in the VALU. This fork already does exactly this
   for MoE: `vllm/model_executor/layers/quantization/online_int8_moe.py`
   quantises BF16 checkpoint weights to INT8 at load time with per-channel
   weight scales and per-token activation quantization, gated behind
   `VLLM_ROCM_USE_AITER_ONLINE_INT8_MOE`.
2. **Use IU8/IU4 WMMA with software scaling.** The matrix unit does the integer
   product; scale application moves into the epilogue. This is the closest
   structural analogue to block scaling, with the block boundary enforced by
   your tiling rather than by hardware.
3. **Dequantise to BF16 before the matrix op.** Simplest to get right, worst on
   bandwidth — which on a unified-memory APU is the resource you have least of
   (see [Memory model](memory_model.md)).

!!! note "Hypothesis — emulated block scaling cost"
    Applying per-32-element scale factors in the VALU between WMMA steps adds
    VALU work proportional to K/32 per output tile and additional VGPR pressure
    for the scale fragments. Whether that lands ahead of straightforward BF16
    WMMA on gfx1151 depends on the arithmetic intensity of the kernel and has
    not been measured here. Benchmark both arms before committing.

## What replaces the missing pieces

| SM120 feature | gfx1151 substitute | Fidelity |
| --- | --- | --- |
| `mma.sync` FP8/FP6/FP4 | BF16/FP16 WMMA, or IU8/IU4 WMMA plus software scaling | Different numerics; must re-validate |
| Block scale factors (`mxf8f6f4`, `mxf4`) | Software scaling in the epilogue or between WMMA steps | Structural analogue, extra VALU |
| Structured sparsity | Nothing | Densify, or choose another quantisation |
| TMA / `cp.async` staging | `global_load_*` into VGPRs, then `ds_store_*` into LDS | More instructions, more registers in flight |
| Warp-group cooperative/pingpong schedules | Hand-written wave scheduling over 16×16×16 steps | No equivalent abstraction |
| FP32 accumulate at 2× rate for narrow types | Standard F32 accumulation | Throughput ratios do not transfer |

**Confirmed.** SM120's narrow-precision throughput is characterised relative to
Ada's FP8 tensor cores (1×-4× depending on type and accumulator), not relative to
SM100. Any performance expectation carried from an SM100 number is wrong on both
of this fork's targets.

## Non-matrix paths worth knowing

Not every matrix-shaped problem should use WMMA. RDNA3.5 has VALU features that
sometimes win for small or skinny shapes.

**Confirmed.** Packed math operates on two 16-bit values within a DWORD as if
they were separate threads, via the VOP3P encoding, covering `V_PK_*` add,
multiply, FMA, min/max and shifts for F16/I16/U16 (ISA §7.5). The DOT family
(`V_DOT2_F32_F16`, `V_DOT2_F32_BF16`, `V_DOT4_I32_IU8`, `V_DOT4_U32_U8`,
`V_DOT8_I32_IU4`, `V_DOT8_U32_U4`) provides dot products without the 16-element
matrix floor.

**Confirmed.** `V_FMA_MIX_*` performs a single MAD over a mixture of 16- and
32-bit inputs, using VOP3P encoding but not packed-math semantics (ISA §7.5).

**Confirmed.** Dual-issue VALU (VOPD) packs two operations into one instruction
in wave32 only, subject to hard register-bank rules: there are 4 VGPR banks
indexed by the low bits of the register number, each with its own cache, and
paired operands must occupy different banks (ISA §7.6). Register *numbering*
therefore affects whether dual-issue is legal at all.

**Practical consequence.** For a decode-shaped GEMV — M=1, which is most of
what a bandwidth-bound serving workload does — the 16×16×16 matrix instruction
wastes 15/16ths of its M dimension. A DOT-based or packed-math VALU kernel can
be the better choice, and is what this fork's RDNA3-family W4A16 kernels take
advantage of (`csrc/rocm/q_gemm_rdna3.cu`, built for `gfx1100|gfx1151` per
`CMakeLists.txt`; the separate WMMA prefill translation unit is gfx1100-only).

!!! note "Hypothesis — WMMA crossover point"
    There is some M at which WMMA overtakes a DOT/packed-math path for a given
    N, K and data type on gfx1151. This fork has not established where that
    crossover sits. If you are choosing between paths for a new kernel,
    measure both across the batch sizes you serve rather than assuming the
    matrix unit wins.

## Porting checklist

- [ ] Is every tile dimension a multiple of 16?
- [ ] Have you re-derived the fragment layout for WMMA rather than reusing the
      `mma.sync` mapping?
- [ ] If hand-writing asm: is there an independent VALU instruction between
      dependent WMMA instructions whose D overlaps the next A or B?
- [ ] Is the input type actually supported (F16, BF16, IU8, IU4)? FP8 is not.
- [ ] If the source kernel was block-scaled, what applies the scale factors now,
      and at what granularity?
- [ ] Does the accumulator fragment fit without crossing a VGPR allocation
      block boundary you care about?
- [ ] For decode shapes, have you compared against a non-WMMA path?
- [ ] Have you validated against a float64 or independent reference, not against
      a tolerance fitted to the output? See [Verification](verification.md).

## Related pages

- [Execution model: warps, waves and launch geometry](execution_model.md)
- [Memory model, LDS and unified memory](memory_model.md)
- [Numerics, atomics and validation](numerics.md)
- [SM120/SM121 constraints](sm120.md)
- [RDNA3.5 constraints](rdna35.md)

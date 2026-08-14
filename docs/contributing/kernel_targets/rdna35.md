# RDNA3.5 (gfx1151) Kernel Constraints

Constraints you will hit writing or porting kernels for AMD Strix Halo /
Radeon 8060S. Section numbers refer to AMD's *"RDNA3.5" Instruction Set
Architecture Reference Guide* (23 July 2024) so you can check the primary
source; the guide is not redistributable and is not vendored in this repo.

The recurring theme: RDNA3.5 is not a small CDNA part. Its matrix unit, its
FP8 encoding, and its register economics all differ from gfx942, and porting
kernels across that boundary is not a matter of relaxing a capability gate.

## Matrix operands must be multiples of 16

RDNA3.5's matrix unit is WMMA (Wave Matrix Multiply Accumulate), described in
ISA §7.9. Every WMMA instruction the architecture provides is a 16×16×16
shape: the A, B, C and D operands are all 16×16 matrices, across the F16,
BF16, IU8 and IU4 input types (ISA §7.9, Table 33). There is no narrower
matrix instruction to fall back on.

The practical consequence is a hard floor, not a performance preference. A
matrix operand narrower than 16 elements has no instruction to lower to, so it
fails in the compiler rather than running slowly.

!!! warning "This fails loudly, and the message does not name your tile"
    A `tl.dot` with any dimension below 16 fails Triton codegen with
    `no matching matrix core intrinsic for wmma version 1`. The error names
    the intrinsic, not the offending config, so it reads like a backend bug.

We hit exactly this with an autotune configuration carrying `BV=8`. That value
had been chosen for MI300X occupancy, where gfx9's MFMA unit accepts such
shapes. WMMA does not. **Keep every tile dimension a multiple of 16
throughout**, and when importing an autotune table from a CDNA part, audit it
for sub-16 dimensions before trusting it -- a config list that is merely
suboptimal on gfx942 is unbuildable on gfx1151.

Two further WMMA properties affect hand-written kernels:

- Data must be arranged so that lanes 0-15 are replicated into lanes 16-31
  (and, for wave64, into lanes 32-47 and 48-63) for the A and B operands
  (ISA §7.9).
- Back-to-back dependent WMMA instructions need an independent VALU
  instruction or a `V_NOP` between them when the first instruction's D operand
  overlaps the second's A or B (ISA §7.9.1). Getting this wrong is a
  correctness bug, not just a stall. The same section lists further overlap
  cases that cost only performance.

WMMA supports round-to-nearest-even only for float types, and raises no ALU
exceptions (ISA §7.9).

## Wave32 versus wave64

Every operation works under either wave size, but the size is fixed at compile
time: a shader runs as the size it was built for, whatever the number of
work-items actually active in a given wave (ISA §2.1).

The distinction matters because wave64 is not simply "wave32 with more lanes".
A wave64 wave issues most VALU and memory instructions *twice*, once per
32-lane half, while scalar ALU, scalar memory, branches and messages issue
only once (ISA §2.1). Both halves see the wave state as it was before the
instruction began, so the low half's results do not feed the high half within
one instruction.

Several things shift between the two halves, and these are where hand-written
wave64 assembly goes wrong: carry-in, carry-out, `v_cmp` results and
`div_scale` operands move to the *next* SGPR for the second pass, and `v_cmpx`
writes `EXEC_HI` rather than `EXEC_LO` (ISA §2.1). Under wave32 only the low 32
bits of `EXEC` and `VCC` carry meaning — the high halves are disregarded, and
`VCCZ`/`EXECZ` summarise just the low half.

Consequences for kernel authors:

- Do not assume a 64-wide cross-lane reduction is one hardware step.
- SGPR alignment rules differ by wave size. A carry-in/carry-out value is 32
  bits and arbitrarily alignable under wave32, but is a 64-bit value requiring
  even-SGPR alignment under wave64 (ISA §3.3.1.3). Misaligned multi-DWORD SGPR
  operands are explicitly illegal with unpredictable results -- the hardware
  forces alignment by ignoring low index bits rather than faulting.

## VGPR allocation granularity and occupancy

This is the mechanism behind the most counter-intuitive performance result in
this fork, and it is worth understanding rather than rediscovering.

VGPRs are not allocated per-register. They are allocated in **blocks**, up to a
maximum of 256 VGPRs per shader. The block size depends on the wave size and on
the size of the register file: the baseline is 16 registers per block for
wave32 (8 for wave64). On a SIMD whose register file holds 1536 VGPRs, however,
the wave32 block is **24 registers** wide (12 under wave64) (ISA §3.3.2.1).
Occupancy -- how many waves
a SIMD can host concurrently -- follows from how many blocks each wave holds
and how many fit in the register file.

So VGPR pressure does not degrade performance smoothly. It is a staircase.
Adding registers is free until an allocation block boundary is crossed, at
which point the wave's block count increments, fewer waves fit per SIMD, and
latency hiding drops discontinuously.

!!! example "Worked example: 216 to 224 VGPRs costs a wave"
    Adding one predicated branch to a kernel raised VGPR usage from 216 to 224.
    Work it through with 24-register blocks and a 1536-register file: 216 needs
    9 blocks (216 registers), and 1536/216 gives **7 waves per SIMD**. 224 needs
    10 blocks (240 registers), and 1536/240 gives **6 waves per SIMD**. Eight
    registers bought one fewer wave in flight -- a 1/7 reduction in
    latency-hiding capacity.

    A regression of roughly that order was observed locally on a saturated grid
    when this change was made, and the same change was invisible at low
    occupancy where the SIMD was not wave-limited in the first place. That
    observation is recorded here as motivation for the arithmetic, not as a
    reproducible benchmark result: the kernel, shapes, driver and ROCm version
    behind it are not captured, so treat the percentage as anecdote. The
    block arithmetic above is the part that generalises.

Two lessons generalise from this:

1. **Compute the distance to the next boundary, don't guess.** Check the
   compiled VGPR count (`-Rpass-analysis=kernel-resource-usage`, or the
   `.vgpr_count` metadata in the compiled object), round it up to the
   allocation block size, and divide the register file by the result. That
   quotient -- not the raw register count -- is what changed.

    Worked through with 24-register blocks and a 1536-register file: 192
    registers is exactly 8 blocks (192 allocated), so 1536/192 gives **8 waves
    per SIMD**. Anything from 193 to 216 rounds up to 9 blocks (216 allocated),
    so 1536/216 gives **7 waves**. Anything from 217 to 240 rounds up to 10
    blocks (240 allocated), so 1536/240 gives **6 waves**.

    Growth *within* a block is therefore free -- 200 to 216 costs nothing, both
    allocate 216 -- while one register crossing a boundary costs a wave: 192 to
    193 drops 8 waves to 7, and 216 to 217 drops 7 to 6.
2. **Benchmark at saturation.** Occupancy regressions are invisible on a small
   grid because there were never enough waves to be limited by wave slots.
   A microbenchmark that does not saturate the GPU will report parity for a
   change that costs double digits in production.

A wave may voluntarily release all its VGPRs via `S_SENDMSG` once it only has
stores outstanding, letting a new wave start earlier (ISA §3.3.2.1); after
doing so, terminating is its only legal action.

For SGPRs, each wave gets 106 normal SGPRs plus `VCC_LO`/`VCC_HI` and 16
trap-temporary registers (ISA §3.3.1.1).

## LDS capacity bounds your tiling

Each work-group processor (WGP) has **128 kB** of LDS, but **a single
work-group may allocate at most 64 kB** (ISA §1.2.2.1, §12.1). Size tiles
against the 64 kB per-work-group limit, not the 128 kB physical figure.

LDS is built from 64 DWORD-wide banks, each a 512×32 two-port RAM, subdivided
into two sets of 32 banks — each set affiliated with one pair of SIMD32s
(ISA §12.1, §1.2.2.1). Concurrent accesses to the same bank serialise as a
bank conflict, cutting effective bandwidth (ISA §12.1), so the usual
padding/swizzling advice applies: stride your LDS tiles so that the lanes of a
wave hit distinct banks.

Allocation also depends on dispatch mode (ISA §2.3, §12.1.2):

- **CU mode** splits LDS into upper and lower halves, each serving two
  SIMD32s. Waves can only access their own half. Faster, but a work-group
  cannot share data across the halves.
- **WGP mode** presents LDS as one contiguous region visible to all four
  SIMD32s, at the cost of the parallelism CU mode gets. `LDS_PARAM_LOAD` and
  `LDS_DIRECT_LOAD` are unsupported in WGP mode.

A work-group may use up to 1024 work-items, and a WGP supports up to 32
work-groups; single-wave work-groups do not count against that limit and
allocate no barrier resource, with barrier operations treated as no-ops
(ISA §2.3).

## FP8: the encoding differs from gfx942

Two separate facts, both of which block a naive port from MI300-class hardware.

**There are no FP8 tensor cores.** WMMA on RDNA3.5 offers F16, BF16, IU8 and
IU4 inputs (ISA §7.9, Table 33). FP8 is not among them. In this tree,
`RocmPlatform.supports_fp8()` returns true only for gfx9 and gfx12x, so
gfx1151 is correctly excluded — see `vllm/platforms/rocm.py`.

**The FP8 encoding is not the same one.** Where FP8 values are handled,
gfx1151 uses the OCP `e4m3fn` encoding, *not* the `e4m3fnuz` ("no infinity, no
unsigned zero") variant used on gfx942. In this tree the selection is explicit:
`is_fp8_fnuz()` tests for `gfx94` and only then returns
`torch.float8_e4m3fnuz`, otherwise `torch.float8_e4m3fn`.

The two encodings assign exponent bias and special values differently, so the
same bit pattern denotes different numbers. Porting an FP8 kernel from
MI300-class hardware therefore requires re-deriving scale factors and any
hand-written conversion or clamping logic. Widening a capability gate to admit
gfx1151 produces numerically wrong output, not slow output.

!!! danger "This fails silently"
    An encoding mismatch does not raise. It produces plausible-looking
    activations that are quietly mis-scaled. Only a reference comparison
    catches it — see [verification](verification.md).

## Memory: unified, and quietly forgiving

On an APU the GPU has no private device memory; it accesses a region of system
memory (ISA §1.2). This is why a *fraction* of total device memory is a poor
allocation control on this hardware, and why this fork provides an absolute
budget — see [conserving memory](../../configuration/conserving_memory.md#gpu-memory-budget).

### Out-of-range behaviour is the biggest silent-failure source

Buffer addresses are range-checked against the buffer size, and the failure mode
is not a fault: out-of-range buffer **loads return zero**, while out-of-range
**stores and atomics are discarded** (ISA §9.4.1).

Range checking is per-component for non-formatted access wider than one DWORD:
a `B64`/`B96`/`B128` access is bounds-checked per DWORD, so a partially
out-of-range vector load returns a *mix* of real and zero data (ISA §9.4.1).
Raw buffers bound-check in bytes and handle multi-DWORD and unaligned cases
exactly; structured buffers check in units of stride.

Reading zeros instead of faulting is exactly the behaviour that turns an
indexing bug into a plausible-looking numerical result. A masked load that is
subtly wrong at a tile edge produces zeros that flow through a softmax without
complaint. **Do not rely on hardware bounds checking to catch indexing errors**
— it is designed to make them survivable, not visible.

Register out-of-range behaviour is similarly quiet (ISA §3.3.2.2, §3.3.3): an
out-of-range destination VGPR turns the instruction into a no-op; an
out-of-range source VGPR reads VGPR0; instructions with multiple destinations
write nothing at all if any destination is out of range. For memory and LDS
reads, an out-of-range source GPR yields undefined data, while an out-of-range
destination nullifies the operation as though `EXEC` were zero.

### Choose the right addressing mode

Three related instruction families (ISA §11.1):

- **Global** — transfers between VGPRs and global memory, uses no LDS
  bandwidth, and increments only `VMcnt`/`VScnt`. If a global instruction does
  touch LDS it returns `MEMVIOL`. Prefer this when you know the access is
  global.
- **Flat** — can resolve to global *or* LDS, decided per-address at runtime by
  an aperture check. Because it may be either, it increments *both*
  `VMcnt`/`VScnt` and `LGKMcnt`, and is not complete until both drain.
- **Scratch** — per-thread private, swizzled space; supports multi-DWORD and
  misaligned access (misaligned is slower), and performs no aperture check.

Flat has a consequence worth internalising: since its data may come from either
LDS or the texture cache, which have different latencies, **the only sensible
`S_WAITCNT` value after a flat instruction is zero** (ISA §11.1.1). Flat
instructions may also complete out of order with respect to each other, and if
two flat loads return to the same VGPR the result is undefined. Using `global`
where you know the space is global therefore both avoids LDS bandwidth
accounting and gives you finer-grained waits.

The aperture check happens when VGPRs are read, *before* `inst_offset` is
added, so behaviour is undefined if adding the immediate offset pushes an
address into a different aperture (ISA §11.1.1). Flat atomics that land in
scratch support 4-byte operations; 8-byte atomics return `MEMVIOL`.

### LDS alignment

Misaligned LDS atomics report a memory violation. Misaligned indexed reads and
writes are only handled when the configured alignment mode is `UNALIGNED`;
otherwise low address bits are ignored to force alignment, silently accessing a
different location than intended, and no violation is reported
(ISA §12.7). Atomics must always be aligned.

## Float semantics that will surprise you

Three behaviours from ISA §13 that affect numerical reproducibility, all of
which are easy to mistake for a kernel bug:

- **Float atomic add flushes input denormals unconditionally.** Unlike min,
  max and compare-swap, which honour the `MODE.fp_denorm` bits, `add` is
  hardwired to flush (ISA §13.2). A reduction implemented with float atomic
  add therefore has different denormal behaviour from the same reduction
  written with ordinary VALU adds.
- **Float atomic add rounding is not configurable.** It always rounds to
  nearest-even regardless of what the wave's rounding-mode state requests
  (ISA §13.1).
- **Min/max return an unmodified source.** When denormal flushing is active,
  flushing applies to the *comparison* only; the returned value is an
  unflushed copy of the selected input (ISA §13.2, §13.3). Signalling NaNs are
  converted to quiet NaNs, and LDS raises no exception on a signalling NaN.

NaN ordering is defined for min and max (ISA §13.3): for max, the ordering
from smallest to largest places QNaN *below* `-inf`; for min, QNaN sorts
*above* `+inf`. In both cases a signalling NaN input short-circuits to a quiet
NaN. If you are implementing a masked softmax by seeding masked lanes with
`-inf`, know that a NaN reaching a max behaves differently from a very negative
number.

Float atomics are available as LDS, buffer and flat/global/scratch operations
(ISA §13).

## Other things worth knowing

- **Instruction prefetch runs past the end of your shader.** Shaders must be
  padded with 64 extra DWORDs (256 bytes) beyond their end, ideally using
  `S_CODE_END`, so the prefetcher cannot reach unmapped pages
  (ISA §2.4). Prefetch distance is configurable (1-3 cache lines) via a
  wave-launch register or `S_SET_INST_PREFETCH_DISTANCE`. This is normally the
  toolchain's job, but matters if you emit or patch machine code.
- **Dual-issue VALU has register-bank constraints.** There are 4 VGPR banks
  indexed by the low bits of the register number, each with its own cache, and
  the paired operands of a dual-issued instruction must occupy different banks
  (ISA §7.6). Register *numbering* therefore affects whether dual-issue is
  legal at all — a consideration when hand-allocating registers.
- **Data dependency resolution is partly software's job.** RDNA requires
  explicit `S_WAITCNT` for memory dependencies rather than resolving them in
  hardware (ISA §5.6), and `S_DELAY_ALU` is used to schedule dependent ALU ops
  (ISA §5.7).
- **GDS is small and global.** A 4 kB global data share is available to all
  WGPs with 2 integer atomic units (ISA §1.2.2.2) — occasionally useful for
  cross-work-group coordination, but far too small for tiling.
- **Hardware can detect IEEE-754 exceptions** and record them for
  post-execution analysis (ISA §1.2.1), which is a debugging avenue when
  chasing NaN origins.

## Profiling on gfx1151

Tooling on this target is thinner than on CDNA, and in this fork's development
environment one of the gaps hung the session rather than erroring.

```bash
# Works: kernel-level tracing.
rocprofv3 --kernel-trace -- <your command>
```

!!! danger "Counter collection hung in this fork's environment"
    Counter collection via `rocprofv3 -i <file>` with a `pmc:` line hung
    indefinitely and could not be interrupted, requiring the session to be
    killed. `rocprof`, `rocprofv2` and `rocprofiler-compute` were not installed.

    This is an observation from a single environment whose ROCm and `rocprofv3`
    versions were not recorded -- not an established property of gfx1151 or of
    ROCm generally. Try it if you like, but start something you are willing to
    kill, and record the versions if it works for you.

Where counters are unavailable, lean on kernel-trace timings plus compiled
resource usage (VGPR/LDS counts from the compiler) to reason about occupancy
from the allocation rules above, rather than measuring it directly.

## Related pages

This page is the target-specific reference for gfx1151. For direct contrasts
against `sm_121a`, and for the porting procedure, see:

- [Execution model: warps, waves and launch geometry](execution_model.md)
- [Matrix units: `mma.sync` versus WMMA](matrix_units.md)
- [Memory model: LDS, caches and unified memory](memory_model.md)
- [Numerics: atomics, NaN handling and narrow formats](numerics.md)
- [Toolchain, target suffixes and dispatch gating](toolchain.md)
- [Porting workflow: SM120 to gfx1151](porting_sm120_to_gfx1151.md)

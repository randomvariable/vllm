# Execution Model: Warps, Waves and Launch Geometry

How work is *shaped* differs between NVIDIA GB10 (`sm_121a`) and AMD Strix Halo
(`gfx1151`) in ways that survive translation only if you are deliberate about
them. A kernel ported by mechanically renaming `threadIdx` to `hipThreadIdx`
will compile and run; whether it is still correct, and whether it is anywhere
near the performance you started with, depends on the differences below.

Section numbers of the form `ISA §x.y` refer to AMD's *"RDNA3.5" Instruction Set
Architecture Reference Guide* (23 July 2024). That guide is not redistributable
and is not vendored in this repo; obtain your own copy from AMD if you want to
follow along. Everything stated here is written to stand on its own if you
cannot.

Facts are labelled. Anything marked **Confirmed** is either sourced from vendor
documentation (with a section citation) or from code in this tree (with a
path). Anything marked **Hypothesis** is inference that has not been measured
on this hardware and must be verified before you rely on it.

## Terminology map

The single most common source of confusion in a port is that the same word
means different things on the two sides. This table is the translation key used
throughout these pages.

| CUDA / SM120 | ROCm / gfx1151 | Notes on the mismatch |
| --- | --- | --- |
| Thread | Work-item | Direct equivalent. |
| Warp (32 threads, fixed) | Wave (32 *or* 64 work-items, fixed at compile time) | Not a rename. See [Wave size is a build decision](#wave-size-is-a-build-decision). |
| Thread block / CTA | Work-group | Both cap at 1024 threads on these parts. |
| Cluster (fixed `1x1x1` in CUTLASS SM120 kernels) | No equivalent construct | Nothing to port; nothing to tune. |
| SM | WGP (work-group processor), containing 4×SIMD32 | An SM is *not* a WGP. Occupancy arithmetic differs (see below). |
| Shared memory | LDS (local data share) | Capacities and allocation rules differ; see [Memory model](memory_model.md). |
| Register file (per SM, 64K 32-bit) | VGPRs per SIMD, allocated in blocks | Allocation is quantised on AMD; see [Occupancy](#occupancy-is-a-staircase-not-a-slope). |
| `__syncthreads()` | `s_barrier` plus counter waits (via `__syncthreads()` in HIP) | Execution rendezvous and memory visibility are separate mechanisms; see below. |
| `__shfl_sync` / `__shfl_xor_sync` | `ds_bpermute_b32`, `ds_permute_b32`, DPP16/DPP8 | Lane-range restrictions differ (ISA §7.7, §12.5.2). |
| `__ballot_sync` | `v_cmp_*` writing an SGPR pair | Mask width follows wave size. |
| `__syncwarp()` | No lane-reconvergence equivalent | Reconvergence is implicit, but memory ordering is not; see below. |
| Hardware scoreboarding of memory ops | `s_waitcnt` counters, software-managed | ISA §5.6. This is the big one for hand-written asm. |
| `cp.async` / TMA | No equivalent | Stage through VGPRs; see [Matrix units](matrix_units.md). |
| Tensor Core (`mma.sync`) | Matrix unit (WMMA) | One shape only on RDNA3.5; see [Matrix units](matrix_units.md). |

## Wave size is a build decision

**Confirmed.** Every RDNA3.5 operation works under either wave size, but a
shader is compiled for one specific size and runs at that size regardless of
how many work-items happen to be active (ISA §2.1). There is no runtime
selection, and no equivalent of a warp that silently narrows.

**Confirmed.** Wave64 is not "wave32 with twice the lanes". A wave64 issues most
VALU and vector-memory instructions twice — once for work-items 0-31 and again
for 32-63 — while scalar ALU, scalar memory, branches, messages and exports
issue once (ISA §2.1). Both halves observe the wave state as it stood before
the instruction began, so results computed by the low half are not visible to
the high half within the same instruction.

**Confirmed.** Several operands shift between the two passes, and this is where
hand-written wave64 code goes wrong (ISA §2.1):

- Carry-in, `div_fmas` and `v_cndmask` read the *next* SGPR on the second pass.
- Carry-out, `div_scale` and `v_cmp` write the *next* SGPR on the second pass.
- `v_cmpx` writes `EXEC_HI` rather than `EXEC_LO`.

Under wave32 the upper 32 bits of `EXEC` and `VCC` carry no meaning, and
`VCCZ`/`EXECZ` summarise only the low half (ISA §2.1).

**Confirmed.** SGPR alignment rules follow from wave size. A carry value is 32
bits under wave32 and may sit at any SGPR index; under wave64 it is a 64-bit
value that must start on an even SGPR (ISA §3.3.1.3). Misaligned multi-DWORD
SGPR operands are illegal and give unpredictable results — the hardware forces
alignment by discarding low index bits rather than faulting, so the failure is
silent.

**Confirmed.** The dual-issue VALU encoding (VOPD) is legal only in wave32 and is
skipped by wave64 (ISA §7.6). A wave64 kernel therefore forfeits dual-issue
entirely.

### What this means for a port

A warp-32 CUDA kernel maps most naturally onto wave32, and on this target
wave32 is also where dual-issue lives. Prefer it unless you have a specific
reason not to.

Two habits from CUDA break under wave64 and are worth auditing explicitly:

1. **Assuming a cross-lane reduction is one hardware step.** A 64-wide
   reduction is not a single pass on wave64, and `ds_permute`/`ds_bpermute`
   operate across 32 lanes at a time even in wave64 — each half behaves as an
   independent wave32, and index values reference lanes within the same half
   (ISA §12.5.2). A `__shfl_xor_sync`-style butterfly that crosses the 32-lane
   boundary has no direct translation.
2. **Assuming lane index equals a global identity.** Under wave64 the second
   issue pass sees a different EXEC half and different scalar operands; code
   that derives addresses from lane identity must be written so that both
   passes agree.

!!! note "Hypothesis — wave64 for bandwidth-bound decode"
    Wave64 halves the number of wave slots consumed per unit of work and can
    reduce instruction-issue pressure on memory-bound kernels, at the cost of
    dual-issue and of doubled VALU issue. Whether that trade favours wave64 for
    any of this fork's decode-path kernels has not been measured here. Treat any
    wave64 claim as unproven until benchmarked at saturation
    (see [Verification](verification.md)).

## No independent thread scheduling

**Confirmed (NVIDIA side).** Since Volta, CUDA warps have independent thread
scheduling, which is why `__syncwarp()` exists and why warp-synchronous
programming without explicit sync is unsafe.

**Confirmed (AMD side).** RDNA3.5 waves execute one instruction stream under an
EXEC mask; divergence is handled by masking and branching, not by independently
scheduled lanes (ISA §2, §3.2.2). There is no `__syncwarp()` analogue because
there is no divergent-reconvergence state to repair.

**Practical consequence for porting.** This does *not* mean `__syncwarp()` calls
can be deleted on sight. `__syncwarp()` does two things: it reconverges the
named lanes, and it orders memory accesses across them so that writes issued
before the call are visible to reads issued after it. RDNA3.5's execution model
removes the need for the first. It does not supply the second — memory
dependencies on this architecture are resolved by `s_waitcnt`, in software,
rather than by hardware interlocks (ISA §5.6).

So a `__syncwarp()` in a source kernel is a signal that lanes are communicating,
and the port has to answer what that communication needs:

1. **What is being communicated, and through what?** Registers via shuffle, LDS,
   or global memory. Shuffle-based exchange carries its own ordering; LDS and
   global exchange do not.
2. **Does the destination need the access to have completed?** If lanes write
   LDS and then read each other's writes, the reads must not be allowed to pass
   the writes. On RDNA3.5 that is a `LGKMcnt` wait, which the compiler emits for
   ordinary HIP code and which you must supply in inline assembly.
3. **What scope does it need?** Cross-work-group visibility additionally depends
   on cache scope bits (ISA §4.1.1), not on any lane-level construct.

The safe procedure is to replace each `__syncwarp()` with the ordering primitive
appropriate to what it was protecting — typically leaving the compiler to emit
the waits by writing ordinary HIP, or an explicit fence where the source used
one — and to delete it outright only where you have established that the lanes
were merely reconverging and exchanging nothing through memory.

The converse direction is also worth noting: code ported *from* AMD to NVIDIA
that relied on implicit lockstep needs syncs added.

## Work-groups, barriers and early exit

**Confirmed.** A work-group is up to 1024 work-items — 32 wave32 waves or 16
wave64 waves — and all of its waves are resident on one WGP and can share LDS
(ISA §2.3, §5.5).

**Confirmed.** A WGP hosts up to 32 work-groups, and work-groups consisting of a
single wave do not count against that limit (ISA §2.3). This is a *scheduling*
property: single-wave groups are cheap to co-resident, so single-wave dispatches
can exceed the nominal group count.

**Confirmed.** Separately, a single-wave work-group allocates no barrier resource
and its barrier instructions are treated as no-ops (ISA §2.3, §5.5). This is a
*synchronization* property, and it is consistent rather than dangerous: a
barrier makes the waves of a group wait for each other, and a group with one
wave has no other wave to wait for.

Keep the two apart. The first is about how many groups fit; the second is about
what a barrier does. Neither says synchronization is unreliable.

**Confirmed.** If a wave reaches a barrier before all waves of the group have
been created, it waits until the group is complete. A wave that terminates
early with `s_endpgm` is treated as having satisfied the barrier; the barrier
completes when the remaining live waves arrive (ISA §5.5).

### What this means for a port

The early-exit rule is more forgiving than CUDA, where `__syncthreads()` in
divergent code is undefined behaviour. Do not read that as licence: a kernel
that depends on AMD's early-exit semantics is not portable back to CUDA, and
the fork serves both targets. Write barriers so they are reached uniformly.

The single-wave elision is worth understanding, but it is not a trap in itself.
Inter-wave synchronization is vacuous when there is one wave, so eliding it
changes nothing about correctness.

What *does* need attention is that `s_barrier` synchronizes execution between
waves; it is not by itself a statement about memory visibility. A kernel that
uses a barrier to publish LDS data between waves depends on both the execution
rendezvous and on the LDS accesses having completed, and the latter comes from
`s_waitcnt` on `LGKMcnt` (ISA §5.6), which the compiler emits around the
barrier. When you write inline assembly or hand-schedule around a barrier, the
wait is your responsibility; the barrier alone does not drain outstanding LDS
traffic.

So when auditing a ported kernel, check what each barrier is actually being
asked to do:

- **Inter-wave execution rendezvous.** Handled by `s_barrier`, and correctly
  vacuous at one wave.
- **Making LDS writes visible to other waves.** Needs the accesses to have
  completed, not just the rendezvous to have happened.
- **Ordering against global memory.** Needs the relevant counter waits and, for
  cross-work-group visibility, appropriate cache scope (ISA §4.1.1).

A CUDA kernel gets the first two together from `__syncthreads()`. Splitting them
apart is the part of the port that needs thought -- not the wave-count
bookkeeping.

## LDS visibility depends on dispatch mode

**Confirmed.** Waves are dispatched in one of two modes, chosen per dispatch at
wave-create time (ISA §2.3, §12.1.2):

- **CU mode** splits LDS into an upper and a lower half, each serving two
  SIMD32s. A wave may access only its own half. Both halves run in parallel, so
  this mode can be faster, but a work-group cannot share data across the split.
- **WGP mode** presents LDS as one contiguous region visible to all four
  SIMD32s, giving up some of the parallelism CU mode has.
  `LDS_PARAM_LOAD` and `LDS_DIRECT_LOAD` are unsupported in WGP mode.

**Confirmed.** In CU mode a wave's LDS allocation lives in the same side of LDS
as the wave; in WGP mode an allocation may sit in either side or straddle the
boundary, unrelated to which CU the wave runs on (ISA §3.3.4).

For a compute kernel written in HIP or Triton this is normally the toolchain's
decision, not yours. It matters when you are reasoning about why a
cross-wave-sharing tile scheme behaves differently than the equivalent CUDA
shared-memory scheme did, and when you are reading disassembly and want to know
whether the LDS addresses you see are group-global.

## Occupancy is a staircase, not a slope

**Confirmed (NVIDIA side).** SM120/SM121 allows 48 resident warps, 24 resident
blocks and 1536 resident threads per SM, with 64K 32-bit registers per SM and a
255-register cap per thread. SM100 allows 64 warps, 32 blocks and 2048 threads.
See [SM120 constraints](sm120.md#resource-limits) for the full table.

**Confirmed (AMD side).** VGPRs are not allocated per register. They are
allocated in blocks, up to 256 VGPRs per shader. The baseline block size is 16
registers for wave32 and 8 for wave64, but on a SIMD whose register file holds
1536 VGPRs the wave32 block is 24 registers (12 for wave64) (ISA §3.3.2.1). A
wave may not be created with zero VGPRs.

**Confirmed.** Occupancy follows from block counts, not raw register counts. The
number of waves a SIMD can host is the register file size divided by the
rounded-up per-wave allocation.

This is why VGPR pressure does not degrade smoothly. Adding registers is free
until an allocation-block boundary is crossed, at which point the wave's block
count increments, fewer waves fit, and latency hiding drops discontinuously.
The worked example measured in this fork — 216 → 224 VGPRs costing 4-17% on a
saturated grid — is in
[RDNA3.5 constraints](rdna35.md#vgpr-allocation-granularity-and-occupancy).

**Confirmed.** Each wave also gets 106 normal SGPRs plus `VCC_LO`/`VCC_HI` and 16
trap-temporary registers (ISA §3.3.1.1). A wave may voluntarily release all its
VGPRs with `s_sendmsg` once only stores remain outstanding, after which
terminating is its only legal action (ISA §3.3.2.1).

### Translating an occupancy target

An SM120 kernel usually carries an implicit occupancy target expressed as
"registers per thread" and validated against `cudaOccupancyMaxActiveBlocks`.
That number does not transfer. To re-derive the equivalent on gfx1151:

1. Compile and read the actual VGPR count
   (`-Rpass-analysis=kernel-resource-usage`, or the `.vgpr_count` metadata in
   the compiled object).
2. Round it up to the allocation block size (24 for wave32 on a 1536-VGPR
   SIMD).
3. Divide the register file size by that rounded figure. The quotient is waves
   per SIMD.
4. Compare that against the waves per SIMD your tiling actually needs to hide
   memory latency at your shapes.

Only step 3's quotient changes performance. Under 24-register blocks, moving
from 192 to 216 registers is free (both are 9 blocks); 216 to 224 costs a wave.

!!! danger "Occupancy regressions are invisible on small grids"
    A microbenchmark that does not saturate the GPU will report parity for a
    change that costs double digits in production, because there were never
    enough waves in flight to be limited by wave slots. Benchmark at the grid
    sizes you actually serve.

## Launch geometry translation

There is no formula that maps a CUDA launch configuration to a good HIP one,
but there is a checklist that avoids the common mistakes.

| Decision | On SM120 | On gfx1151 | Porting note |
| --- | --- | --- | --- |
| Block size | Multiple of 32; 128 or 256 typical | Multiple of wave size (32 or 64) | A 96-thread block is 3 waves of 32 but wastes half a wave64. |
| Max block | 1024 threads | 1024 work-items (ISA §2.3) | Same ceiling; different occupancy consequences. |
| Blocks resident | 24 per SM | 32 work-groups per WGP; single-wave groups exempt (ISA §2.3) | Many-small-groups schemes behave differently. |
| Shared/LDS per group | 99 KB per block, opt-in above 48 KB | 64 KB per work-group, from 128 KB per WGP (ISA §1.2.2.1, §12.1) | Re-tile; see [Memory model](memory_model.md). |
| Grid-level cooperation | Clusters exist in the ISA but CUTLASS SM120 kernels fix `1x1x1` | No cluster construct | Any cluster-sweeping autotune space is meaningless on both. |
| Cross-lane reduction width | 32 | 32 within a permute; 64 only via two halves | Rewrite butterflies wider than 32 lanes. |

## Software-managed dependencies

**Confirmed.** RDNA requires the program to resolve memory dependencies with
`s_waitcnt` rather than resolving them in hardware (ISA §5.6). Four counters
track outstanding work, and instructions of different types complete out of
order with respect to each other:

- `VMcnt` — vector-memory loads and atomics that return a value.
- `VScnt` — vector-memory stores and atomics that do not return.
- `LGKMcnt` — LDS indexed operations, scalar memory, GDS and GWS.
- `EXPcnt` — LDS parameter/direct loads and exports.

`FLAT` instructions increment both `LGKMcnt` and one of `VMcnt`/`VScnt`, and are
not complete until both have drained (ISA §5.6, §11.1.1).

**Confirmed.** `s_delay_alu` exists to schedule dependent ALU operations and is
optional — it affects stalls, not correctness (ISA §5.7).

For kernels written in HIP or Triton the compiler emits these. The reason to
know they exist is diagnostic: when a ported kernel produces intermittently
wrong results only under high occupancy, a missing or over-relaxed wait is a
plausible cause in hand-written assembly or inline asm, and there is no hardware
interlock that will save you. It is also why reading disassembly on this target
is more informative than on NVIDIA — the waits tell you what the compiler
believed about your memory ordering.

## Open questions

!!! note "Hypothesis — grid-wide synchronisation"
    RDNA3.5 exposes Global Wave Sync and ordered-count operations for
    GPU-wide coordination (ISA §13.4), with strict programming rules around
    outstanding GDS operations. Whether HIP exposes anything equivalent to
    CUDA cooperative-groups grid sync on this target, and whether it is usable
    from Triton, has not been established here. Do not port a
    grid-synchronising kernel on the assumption that it will work.

!!! note "Hypothesis — CU vs WGP mode selection"
    Whether the ROCm toolchain's choice of CU or WGP mode is influenced by LDS
    request size or by launch bounds on this target has not been verified.
    If a tiling scheme depends on all four SIMD32s seeing one LDS region, check
    the generated code rather than assuming.

## Related pages

- [Matrix units: `mma.sync` versus WMMA](matrix_units.md)
- [Memory model, LDS and unified memory](memory_model.md)
- [Numerics, atomics and validation](numerics.md)
- [Porting workflow: SM120 to gfx1151](porting_sm120_to_gfx1151.md)

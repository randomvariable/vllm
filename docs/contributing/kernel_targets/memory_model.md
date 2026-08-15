# Memory Model: LDS, Caches and Unified Memory

A kernel's tiling is a statement about a memory hierarchy. Port it to a
different hierarchy without re-deriving the numbers and you get one of three
outcomes: a launch failure, a silent fallback, or a kernel that runs at a
fraction of its NVIDIA throughput for reasons no profiler on this target will
show you.

Section numbers of the form `ISA §x.y` refer to AMD's *"RDNA3.5" Instruction Set
Architecture Reference Guide* (23 July 2024), which is not vendored here. SM120
figures come from NVIDIA's CUDA Programming Guide compute-capabilities appendix
and CUTLASS Blackwell functionality documentation; see
[SM120 constraints](sm120.md) for the tables.

Facts are labelled **Confirmed** (vendor documentation with a citation, or code
in this tree with a path) or **Hypothesis** (inference not yet measured here).

## Capacity comparison

| | SM120/SM121 (`sm_121a`) | gfx1151 (RDNA3.5) |
| --- | --- | --- |
| Scratchpad name | Shared memory | LDS (local data share) |
| Physical per SM / per WGP | 100 KB max shared per SM | 128 KB per WGP (ISA §1.2.2.1) |
| Per thread block / work-group | 99 KB | **64 KB** (ISA §1.2.2.1, §12.1) |
| Opt-in threshold | Above 48 KB requires dynamic shared memory and explicit opt-in | Allocated in 1024-byte blocks, 0-64 KB (ISA §3.3.4) |
| Legal carve-outs | 0/8/16/32/64/100 KB | Not carved out the same way; see dispatch mode below |
| Banks | 32 banks (standard for CUDA) | 64 DWORD-wide banks, split into two sets of 32 (ISA §1.2.2.1, §12.1) |
| Bank RAM | — | 512×32 two-port, 1R/1W per clock (ISA §12.1) |
| Unified data cache | 128 KB | L0 per WGP, L1, L2, optional MALL (ISA §1.2.3, §4.1.1) |
| Registers | 64K 32-bit per SM, 255 per thread | 256 VGPRs per shader, block-allocated (ISA §3.3.2.1) |
| Device memory | Dedicated + unified addressing | **No private device memory**; region of system memory (ISA §1.2) |
| Small global scratch | — | 4 KB GDS, all WGPs, 2 integer atomic units (ISA §1.2.2.2) |

The number that ends most direct ports is 64 KB. **Confirmed:** each WGP has
128 kB of LDS, but a single work-group may allocate at most 64 kB (ISA
§1.2.2.1, §12.1). Size tiles against the 64 KB per-work-group limit, not the
128 KB physical figure.

An SM120 kernel tiled against 99 KB per block has to be re-tiled. An SM100
kernel tiled against 227 KB has to be re-tiled twice — once for SM120 and again
for gfx1151 — and the SM120 step is a launch failure if skipped, not a slow
path.

## LDS banking and swizzle translation

**Confirmed.** LDS is built from 64 DWORD-wide banks, each a 512×32 two-port RAM,
subdivided into two sets of 32 banks with each set affiliated to one pair of
SIMD32s (ISA §1.2.2.1, §12.1). DWORDs are placed in banks serially, and all
banks can service a load or store simultaneously.

**Confirmed.** Concurrent accesses to the same bank serialise as a bank conflict
for indexed and atomic operations — the hardware turns them into serial
accesses, cutting effective LDS bandwidth (ISA §12.1). Avoiding conflicts
requires knowing the request scheduling and address mapping.

### What transfers and what does not

A CUDA swizzle exists to make the 32 lanes of a warp hit 32 distinct banks. The
*purpose* transfers directly; the *arithmetic* does not, because the bank count
and the grouping differ.

| Aspect | CUDA | gfx1151 | Porting consequence |
| --- | --- | --- | --- |
| Bank count | 32 | 64, in two sets of 32 (ISA §12.1) | A modulo-32 swizzle is not automatically wrong, but it is not automatically right either. |
| Bank width | 4 bytes | 4 bytes (DWORD) (ISA §12.1) | Element-to-bank arithmetic has the same units. |
| Access granularity | Warp of 32 | Wave of 32 or 64 | Under wave64 the second issue pass presents another 32 addresses. |
| Padding trick | `+1` element per row | Same idea, different modulus | Recompute against 64 banks and your actual access stride. |
| XOR swizzle | `row ^ col` patterns | Same idea, different modulus | Re-derive; do not copy constants. |

**Practical method.** Rather than porting a swizzle expression, port the
constraint it satisfies:

1. Write down the set of LDS byte addresses one wave touches in a single
   instruction.
2. Divide by 4 to get DWORD indices, then take modulo 64 to get banks.
3. Check for duplicate banks across the 32 (or 64) lanes.
4. Adjust padding or XOR terms until duplicates disappear.

Do this for both the store side (writing the staged tile) and the load side
(feeding the matrix unit). A swizzle that fixes one commonly breaks the other,
which is the reason the CUDA constants you started with exist at all.

!!! note "Hypothesis — the two-set structure"
    The affiliation of each 32-bank set with one pair of SIMD32s (ISA §12.1)
    suggests conflict behaviour may differ between CU mode, where a wave uses
    only its own half, and WGP mode, where allocations may straddle the
    boundary (ISA §3.3.4, §12.1.2). Whether a swizzle tuned in one mode remains
    conflict-free in the other has not been verified here.

## Addressing modes: pick the narrow one

**Confirmed.** Three related vector-memory instruction families exist for the
flat address space (ISA §11.1):

- **Global** — transfers between VGPRs and global memory, uses no LDS
  bandwidth, and increments only `VMcnt`/`VScnt`. If a global instruction does
  attempt to touch LDS it returns `MEMVIOL`.
- **Flat** — may resolve to global *or* LDS *or* scratch, decided per address at
  runtime by an aperture check. Because it may be either, it increments *both*
  `VMcnt`/`VScnt` and `LGKMcnt`, and is not complete until both drain.
- **Scratch** — per-thread private, swizzled space; supports multi-DWORD and
  misaligned access (misaligned is slower), performs no aperture check, and
  uses no LDS bandwidth.

**Confirmed.** Because flat data can come from either LDS or the texture cache,
which have different latencies, **the only sensible `s_waitcnt` value after a
flat instruction is zero** (ISA §11.1.1). Flat instructions also complete out of
order with respect to each other, so if two flat loads return to the same VGPR
the result is undefined.

**Confirmed.** The aperture check happens when VGPRs are read, *before*
`inst_offset` is added, so behaviour is undefined if adding the immediate offset
pushes an address into a different aperture (ISA §11.1.1). Flat atomics that
land in scratch support 4-byte operations; 8-byte atomics return `MEMVIOL`.

**Practical consequence for a port.** CUDA's generic pointers are ubiquitous and
essentially free. On gfx1151 a generic pointer that lowers to `FLAT` costs you
both counter granularity and the ability to overlap waits. Where you know the
address space, say so: `__global__`-qualified pointers, HIP's address-space
attributes, or Triton's pointer types that lower to `global_load_*`. Reading the
disassembly for `flat_load_*` versus `global_load_*` is a cheap and high-value
check on a ported kernel.

## Cache control and non-temporal hints

**Confirmed.** Scalar and vector memory instructions carry bits that control
cache behaviour (ISA §4.1.1):

- `GLC` is a scope bit for loads: 0 means work-group (CU) scope, 1 means device
  scope. Typically loads use `GLC=0` except for load-acquire. `GLC=1` forces a
  miss in the first-level cache and reads from L2, invalidating any matching L0
  line.
- `SLC` is a temporal hint for the graphics client caches: 0 regular, 1 stream
  (non-temporal).
- `DLC` is a temporal hint for the memory-attached last-level cache (MALL) if
  present, and is ignored otherwise.
- All stores and atomics are device scope.
- For atomics, `GLC` selects whether the pre-operation value is returned to a
  VGPR (1) or nothing is returned (0).

**Practical translation.** CUDA's `__ldg`, `ld.global.nc`, `.cs`/`.lu`/`.cg`
cache-operator suffixes and `cuda::memcpy_async` staging hints all express
intent about reuse. On gfx1151 the equivalent lever is the `SLC`/`DLC` pair
plus the choice of instruction family. A streaming pass over KV cache that
should not evict weights is a `SLC=1` case; a small hot table that should stay
resident is not.

!!! note "Hypothesis — non-temporal hints on an APU"
    On a discrete GPU, marking a streaming load non-temporal protects a
    dedicated device-memory cache hierarchy. On Strix Halo the same caches front
    system memory shared with the CPU, so the value of the hint depends on
    contention that a GPU-only benchmark will not reproduce. Whether `SLC=1` on
    KV-cache traffic helps, hurts or is neutral on this part has not been
    measured here.

## Unified memory changes the problem

**Confirmed.** On an APU the GPU has no private device memory; it accesses a
region of system memory (ISA §1.2). This is why a *fraction* of total device
memory is poor allocation control on this hardware, and why this fork provides
an absolute budget — see
[conserving memory](../../configuration/conserving_memory.md#gpu-memory-budget).

**Confirmed.** The device memory path runs through multiple channels of L2
feeding read-only L1 and then per-WGP L0 caches, with specific cache-less load
instructions available to force retrieval from device memory during a load
clause (ISA §1.2.3).

### What changes in a port

An SM120 kernel is written against an assumption that never appears in the
source: that device memory bandwidth is large relative to host-transfer
bandwidth, and that anything already on the device is cheap to re-read.

On a unified-memory APU:

1. **Host-device copies are not the bottleneck they were.** Strategies that
   exist purely to avoid transfers may be dead weight.
2. **Bandwidth is shared with the CPU.** A benchmark run on an otherwise idle
   machine overstates what you get while a server is also handling tokenisation,
   sampling and request handling.
3. **Arithmetic intensity matters more, not less.** With no private HBM to hide
   behind, a kernel that re-reads a tile because it did not fit in LDS pays
   system-memory latency, not device-memory latency.
4. **Memory capacity is not free.** The budget you take is taken from the host.

**Practical consequence.** When porting, re-examine any decision in the source
kernel that traded memory traffic for compute. On a bandwidth-rich datacentre
part that trade often favoured recompute; here it more often favours keeping
data resident. This is the same reasoning behind the online INT8 MoE path in
`vllm/model_executor/layers/quantization/online_int8_moe.py`, which trades
weight precision for halved weight traffic on a bandwidth-bound APU.

## Coalescing and alignment

**Confirmed.** Formatted buffer operations have explicit alignment requirements:
1-byte formats need 1-byte alignment, 2-byte formats need 2-byte alignment, and
4-byte and larger formats need 4-byte alignment. Atomics must be aligned to the
data size or trigger `MEMVIOL` (ISA §9.5).

**Confirmed.** Alignment enforcement for non-formatted operations is controlled
by a configuration register (`SH_MEM_CONFIG.alignment_mode`), affecting LDS and
flat/scratch/global operations alike. In `DWORD` mode, alignment is automatic to
a multiple of the smaller of element size or DWORD; in `UNALIGNED` mode there
are no alignment requirements (ISA §3.3.3).

**Confirmed.** Misaligned LDS atomics report a memory violation. Misaligned
indexed reads and writes are handled only when the configured alignment mode is
`UNALIGNED`; otherwise low address bits are ignored to force alignment, silently
accessing a different location than intended, and no violation is reported (ISA
§12.7, §3.3.4.1). Native LDS alignment is byte for B8, 2-byte for B16/D16,
4-byte for B32, 8-byte for B64, and 16-byte for B96/B128.

!!! danger "Silent address rewriting"
    Under the default alignment mode a misaligned LDS access does not fault. The
    hardware zeroes the low address bits and reads or writes somewhere else.
    This is the LDS analogue of the out-of-range read-zero behaviour described
    below, and it has the same consequence: an indexing bug becomes a plausible
    number.

**Practical translation.** CUDA's coalescing rules are about aligned, contiguous
128-byte transactions per warp. The intent transfers. The specifics to check on
gfx1151:

- Vector width choices (`B64`, `B96`, `B128`) carry alignment requirements that
  a `float4` load in CUDA satisfied implicitly. Keep the alignment invariant
  when you change the element type — an FP8-to-BF16 conversion doubles element
  size and can break a stride that was aligned before.
- Under wave64 the addresses of the second issue pass matter as much as the
  first.
- Scratch supports misaligned access but it is slower (ISA §11.1.3). If a port
  ends up in scratch because of register spilling, misalignment compounds an
  already bad situation.

## Out-of-range is the biggest silent-failure source

**Confirmed.** Buffer addresses are range-checked against the buffer size, and
the failure mode is not a fault: out-of-range **loads return zero**, while
out-of-range **stores and atomics are discarded** (ISA §9.4.1).

**Confirmed.** Range checking is per-component for non-formatted access wider
than one DWORD: `B64`/`B96`/`B128` accesses are bounds-checked per DWORD, so a
partially out-of-range vector load returns a *mix* of real and zero data (ISA
§9.4.1). Raw buffers bound-check in bytes and handle multi-DWORD and unaligned
cases exactly; structured buffers check in units of stride.

**Confirmed.** LDS follows the same philosophy: out-of-range LDS writes are
discarded and reads return zero, and for a multi-DWORD read, if any part of the
address is out of range the whole instruction returns zero (ISA §3.3.4.1).

**Confirmed.** Register out-of-range is similarly quiet (ISA §3.3.2.2, §3.3.3).
An out-of-range destination VGPR turns the instruction into a no-op; an
out-of-range source VGPR reads VGPR0; instructions with multiple destinations
write nothing at all if any destination is out of range. For memory and LDS
reads, an out-of-range source GPR yields undefined data, while an out-of-range
destination nullifies the operation as though `EXEC` were zero.

### Why this matters more than it sounds

Reading zeros instead of faulting is exactly the behaviour that turns an
indexing bug into a plausible-looking numerical result. A masked load that is
subtly wrong at a tile edge produces zeros that flow through a softmax without
complaint. **Do not rely on hardware bounds checking to catch indexing errors**
— it is designed to make them survivable, not visible.

This is a genuine behavioural difference from CUDA, where an out-of-bounds
global access typically produces a fault that `compute-sanitizer` will name.
There is no equivalent safety net here, and (see below) the profiling tools that
would help are partly unavailable.

**Practical consequence.** Boundary conditions must be tested explicitly. For
every ported kernel, include shapes that are *not* multiples of the tile size,
and verify the masked region rather than assuming the mask worked. A
tolerance-fitted `allclose` will pass on a kernel that zeroes its last partial
tile.

## Staging: what replaces `cp.async` and TMA

**Confirmed.** SM120's data movement toolkit includes TMA and, in CUTLASS
kernels, the warp-specialised producer/consumer schedules built on it. gfx1151
has no equivalent asynchronous bulk-copy engine.

**Confirmed.** LDS is filled either by transferring from VGPRs with DS
instructions, or by loading from memory — where the data goes to VGPRs first,
or, **for some load types, is loaded directly into LDS from memory** (ISA
§12.1). The direct forms exist; what does not exist is an asynchronous bulk-copy
engine with its own descriptors and completion tracking.

**Confirmed.** The store direction has no such shortcut: to move data from LDS to
global memory it is read from LDS into the work-item's VGPRs and then written
out (ISA §12.1).

**Be precise about what is missing.** The gap is TMA's *programming model* —
descriptor-driven, asynchronous, multi-dimensional bulk transfer that a
producer warp can issue and a consumer warp group can wait on — not the ability
to get memory into LDS without a register round trip. Two consequences follow,
and they are different:

- **You may not need to route every tile through VGPRs.** Where a direct-to-LDS
  load form applies, prefer it: it avoids the register residency that otherwise
  dominates the pipeline's cost. Check what the compiler actually emits before
  assuming a VGPR round trip is mandatory.
- **You do need to build the pipeline yourself.** There is no descriptor object,
  no asynchronous completion group, and no producer/consumer warp
  specialisation to inherit. Sequencing, double buffering and the waits are
  yours to write.

!!! note "Hypothesis — which direct-to-LDS forms apply, and when"
    ISA §12.1 establishes that direct memory-to-LDS loads exist for some load
    types, but this page does not enumerate which forms are available on
    gfx1151, what their alignment and width constraints are, or whether the
    ROCm toolchain selects them for a given HIP or Triton construct. That was
    not established here. Read the generated assembly for your kernel rather
    than assuming either that the direct path is taken or that it is not.

**Practical consequence.** A CUTLASS-style pipelined mainloop with a TMA
producer warp and an MMA consumer warp group does not translate structurally.
Where a VGPR-staged pipeline is what you end up with, it looks like:

1. Issue `global_load_*` for tile *i+1* into VGPRs.
2. Do WMMA work on tile *i*, already in LDS or registers.
3. `s_waitcnt` on the loads for *i+1*.
4. `ds_store_*` tile *i+1* into LDS.
5. Barrier and swap.

The cost relative to TMA is register pressure: the in-flight tile occupies VGPRs
for the whole of step 2, and VGPR pressure is exactly what the
[occupancy staircase](execution_model.md#occupancy-is-a-staircase-not-a-slope)
punishes. Deeper pipelines buy latency hiding and cost waves per SIMD. Where
that trade lands is per-kernel and must be measured.

!!! note "Hypothesis — optimal pipeline depth"
    The right number of pipeline stages on gfx1151 depends on the ratio of
    system-memory latency to per-tile WMMA work, and on where the resulting VGPR
    count falls relative to the 24-register allocation blocks. No general figure
    has been established for this fork's kernels. Compute the VGPR count for
    each candidate depth before benchmarking, so you know which depths are on
    the same step of the staircase.

## GDS: small, global, and rarely what you want

**Confirmed.** A 4 kB global data share is available to all WGPs, with 2 integer
atomic units and 128 bytes per cycle of access, providing full access to any
location from any processor (ISA §1.2.2.2).

Occasionally useful for cross-work-group coordination — a global counter, a
work-stealing index — but far too small for tiling. If a port reaches for GDS,
check first whether the algorithm actually needs cross-group communication or
whether it inherited that structure from a cluster-based SM100 design that
should have been flattened.

## Instruction prefetch padding

**Confirmed.** Instruction prefetch runs past the end of a shader. Shaders must
be padded with 64 extra DWORDs (256 bytes) beyond the end, ideally using
`S_CODE_END`, so the prefetcher cannot reach unmapped pages (ISA §2.4). Prefetch
distance is configurable (1-3 cache lines) via a wave-launch register or
`S_SET_INST_PREFETCH_DISTANCE`.

This is normally the toolchain's job. It matters if you emit or patch machine
code directly.

## Profiling on this target

**Confirmed (in this fork's environment).** Kernel-level tracing works:

```bash
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

**Practical consequence, where counters are unavailable.** You cannot measure LDS
bank conflicts, cache hit rates or memory stall cycles directly, which is a
substantial handicap for exactly the analysis this page describes. The workable
substitutes:

- **Compiled resource usage.** VGPR and LDS counts from the compiler
  (`-Rpass-analysis=kernel-resource-usage`, or `.vgpr_count` metadata) let you
  compute occupancy analytically from the allocation rules.
- **Kernel-trace timings.** Wall time per kernel, interleaved between arms —
  see [Verification](verification.md#performance-interleave-the-arms).
- **Differential experiments.** Change one thing (a padding constant, a vector
  width, an addressing mode) and measure. Slower than a counter read, but it is
  what is available.
- **Disassembly reading.** `flat_` versus `global_`, `s_waitcnt` placement, and
  spill traffic to scratch are all visible statically.

## Porting checklist

- [ ] Does the tile fit in 64 KB per work-group, not 99 KB or 128 KB?
- [ ] Has the swizzle been re-derived against 64 banks, for both store and load
      sides?
- [ ] Do generic pointers lower to `global_*` where the space is known, rather
      than `flat_*`?
- [ ] Are vector loads still aligned after any element-type change?
- [ ] Are edge shapes — non-multiples of the tile — tested, with the masked
      region verified rather than assumed?
- [ ] Have you checked for scratch spill traffic in the disassembly?
- [ ] Does the staging pipeline's VGPR count sit on the step of the occupancy
      staircase you intended?
- [ ] Have you re-examined recompute-versus-reload decisions inherited from a
      dedicated-memory part?

## Related pages

- [Execution model: warps, waves and launch geometry](execution_model.md)
- [Matrix units: `mma.sync` versus WMMA](matrix_units.md)
- [Numerics, atomics and validation](numerics.md)
- [RDNA3.5 constraints](rdna35.md)
- [Conserving memory](../../configuration/conserving_memory.md#gpu-memory-budget)

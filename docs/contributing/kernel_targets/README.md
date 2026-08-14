# Writing Kernels for Homelab Targets

This fork serves two GPU targets whose constraints differ sharply from the
datacentre parts most vLLM kernels are written and tuned against:

| Target | Arch | Notes |
| --- | --- | --- |
| AMD Strix Halo / Radeon 8060S | ROCm, `gfx1151`, RDNA3.5 | Unified memory, WMMA matrix cores, no FP8 tensor cores |
| NVIDIA GB10 / DGX Spark | CUDA, `sm_121a`, Blackwell consumer-class | Consumer-class Blackwell, not SM100 datacentre Blackwell |

A kernel that runs on MI300X or on an H100/B200 will frequently *build* for
these targets and then either fail at codegen, fall back to a slow generic
path, or -- worst -- silently return wrong numbers. These pages cover what
actually bites.

## How to use these pages

**Working on one target.** Read the page for that target, then the verification
page. The verification bar applies to both targets and is not optional.

- [RDNA3.5 (gfx1151) kernel constraints](rdna35.md)
- [Blackwell consumer-class (SM120/SM121) kernel constraints](sm120.md)
- [Verifying kernel correctness and performance](verification.md)

**Porting a kernel between targets.** Start with the workflow page, which
sequences the work and links out to the detail pages as you need them.

- [Porting workflow: SM120 to gfx1151](porting_sm120_to_gfx1151.md)

**Understanding a specific subsystem.** These four pages are the cross-target
reference. Each contrasts the two architectures directly rather than describing
one of them.

- [Execution model: warps, waves and launch geometry](execution_model.md) --
  wave32/wave64, occupancy arithmetic, barriers, launch geometry, terminology
  translation table.
- [Matrix units: `mma.sync` versus WMMA](matrix_units.md) --
  the 16-element floor, fragment layouts, scheduling hazards, narrow-precision
  gaps and what replaces them.
- [Memory model: LDS, caches and unified memory](memory_model.md) --
  capacities, bank conflicts and swizzle translation, addressing modes,
  alignment, silent out-of-range behaviour, staging without TMA.
- [Numerics: atomics, NaN handling and narrow formats](numerics.md) --
  atomic denormal and rounding rules, NaN ordering, FP8 encodings, validation
  strategy for numeric-format ports.

**Build and dispatch problems.** When a kernel is correct but absent:

- [Toolchain, target suffixes and dispatch gating](toolchain.md)

## Why "runs" is not the bar

Per this fork's standing requirements, both targets must run every attention
type on the most optimized available path. A model that serves only by falling
back to a slower generic backend is a gap to close, not support. When you find
a kernel dispatching to a generic Triton fallback on one of the targets,
that is a finding -- not the resolution.

## Confirmed facts versus hypotheses

The cross-target pages label their claims. **Confirmed** means the statement is
sourced either from vendor documentation, with a section citation, or from code
in this tree, with a path you can check. **Hypothesis** means it is reasoned
inference that has *not* been measured on this hardware.

This distinction is load-bearing. Profiling on gfx1151 is
[substantially limited](memory_model.md#profiling-on-this-target) -- hardware
counter collection hangs uninterruptibly -- so a great deal of what would
normally be measured has to be reasoned about instead. Treat every hypothesis
as work to be done, not as guidance to follow.

If you measure one, replace it with the number and cite the benchmark.

## Reference material

The hardware facts in these pages were derived from vendor documentation:

- AMD's *"RDNA3.5" Instruction Set Architecture Reference Guide* (23 July
  2024). Section numbers are cited so you can follow along in your own copy.
  AMD distributes this under a specification agreement that forbids
  redistribution, so it is not vendored here and no text from it is reproduced
  -- obtain it from AMD.
- NVIDIA's CUTLASS Blackwell functionality documentation, at
  <https://docs.nvidia.com/cutlass/latest/media/docs/cpp/blackwell_functionality.html>.
- NVIDIA's CUDA Programming Guide, [compute capabilities
  appendix](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html),
  for per-device limits and compiler target semantics.

These pages are written to be usable **without** those sources. Where a fact
matters for a porting decision, it is stated in full here in original prose,
with the citation present so you can verify it if you have access rather than
so you can look it up because you must. If the upstream documents move or
disappear, the guidance above still stands on its own.

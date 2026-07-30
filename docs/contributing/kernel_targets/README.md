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

Read the page for your target, then read the verification page. The
verification bar applies to both targets and is not optional.

- [RDNA3.5 (gfx1151) kernel constraints](rdna35.md)
- [Blackwell consumer-class (SM120/SM121) kernel constraints](sm120.md)
- [Verifying kernel correctness and performance](verification.md)

## Why "runs" is not the bar

Per this fork's standing requirements, both targets must run every attention
type on the most optimized available path. A model that serves only by falling
back to a slower generic backend is a gap to close, not support. When you find
your kernel dispatching to a generic Triton fallback on one of these targets,
that is the finding -- not the resolution.

## Reference material

The hardware facts in these pages were derived from vendor documentation:

- AMD's *"RDNA3.5" Instruction Set Architecture Reference Guide* (23 July
  2024). Section numbers are cited so you can follow along in your own copy.
  AMD distributes this under a specification agreement that forbids
  redistribution, so it is not vendored here and no text from it is reproduced
  -- obtain it from AMD.
- NVIDIA's CUTLASS Blackwell functionality documentation, at
  <https://docs.nvidia.com/cutlass/latest/media/docs/cpp/blackwell_functionality.html>.

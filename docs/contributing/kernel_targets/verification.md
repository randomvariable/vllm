# Verifying Kernel Correctness and Performance

Applies to both homelab targets. A kernel is not done when it runs, and it is
not fast because a benchmark said so once. This page is the bar.

## Correctness: compare against an independent reference

**Do not choose a tolerance after seeing the result.** Fitting `atol`/`rtol` to
whatever the kernel produced is not a test; it is a record of the kernel's
current output, and it will happily accept a systematic error.

Compare against one of:

1. **A float64 computation of the same mathematics.** Compute the reference in
   `torch.float64` (on CPU if necessary), then compare your kernel's output to
   it. The tolerance is then justified by the precision you are claiming, not
   by the answer you got.
2. **An independent implementation.** An existing, trusted backend path, or a
   naive PyTorch expression of the same operation. "Independent" means it does
   not share the code under test — a reference that calls the same helper as
   your kernel validates nothing about that helper.

State the tolerance's derivation. If you accept `1e-2` for a BF16 kernel,
that number should follow from BF16's mantissa and your accumulation order, not
from experiment.

Both targets have failure modes that produce *plausible* wrong numbers rather
than obvious garbage, and each of these is invisible to a
tolerance-fitted test:

- gfx1151 buffer loads that go out of range **return zero** rather than
  faulting, so an off-by-one at a tile edge silently mixes real data with
  zeros.
- FP8 encoding differences between gfx942 (`e4m3fnuz`) and gfx1151
  (`e4m3fn`) mis-scale values without erroring.
- Block-scaled narrow-precision kernels on SM120 mis-group scale factors if the
  vector size is wrong, which perturbs magnitudes plausibly.

### Top-k style kernels: elementwise closeness is the wrong test

For any kernel whose output feeds a selection — top-k, top-p, MoE routing,
speculative-decoding verification, indexer scoring — **elementwise closeness is
insufficient**, because the failure mode is not magnitude error, it is
**ranking drift**.

Two score vectors can agree to within a tight tolerance at every position and
still select different experts, or different tokens, wherever two scores were
nearly tied. The consequence surfaces far downstream as degraded output
quality, not as a failed allclose.

Test the thing that matters:

- Compare the **selected indices**, not just the scores. Exact set equality for
  the top-k, per row.
- Where ties are legitimately possible, assert that any index disagreement is
  confined to positions whose scores are genuinely tied, and that the
  *multiset of selected scores* matches.
- For routing, check the selection is stable across the batch shapes and
  sequence lengths you intend to serve, not just one shape.

A kernel that is elementwise-accurate and rank-unstable is broken.

## Performance: interleave the arms

Both of these GPUs ramp from an idle clock floor. That single fact invalidates
the obvious benchmarking method.

!!! danger "A sequential sweep charges the ramp to the second arm"
    Running all iterations of arm A, then all iterations of arm B, means A pays
    the clock ramp from idle while B runs at settled clocks. The measured
    difference includes the ramp, attributed entirely to whichever arm ran
    first. On a unified-memory APU or a Spark that has been idle, this can
    dwarf the effect you are trying to measure — and it is systematic, so
    repeating the sweep reproduces the same wrong answer.

**Interleave instead.** Alternate arms within a single measurement loop
(`A B A B A B ...`), so both see the same clock trajectory, and report per-arm
distributions rather than a single mean. Warm up before the measured region,
and keep the warm-up representative of the shape you are measuring.

### Include control configurations

Add arms whose result you can predict, and check the predictions:

- **A parity control.** Two arms that *should* be identical — for example the
  same configuration submitted twice under different labels, or a flag that
  should be a no-op for the shape under test. If your harness reports a
  difference between arms that must be equal, the harness is measuring noise or
  ramp, and every other number it produced is suspect. This is a bias check on
  the measurement, not on the kernel.
- **A known-slow arm.** Something you expect to lose (a generic fallback, or a
  deliberately poor tile). If it does not lose, you are probably not exercising
  the path you think you are.

Report the parity control's result alongside your findings. A speedup claim
without one is not yet evidence — it is a measurement whose bias is unknown.

### Measure at saturation

Occupancy effects only appear when the machine is wave-limited. A local
observation on gfx1151: a register-pressure change that cost a wave per SIMD
showed a clear regression on a saturated grid and was **invisible** at low
occupancy, because there were never enough waves in flight to be limited by
wave slots. The environment behind that observation was not captured, so treat
it as motivation rather than as a number to compare against. Benchmark at the
grid sizes you will actually serve.

## Profiling

### ROCm / gfx1151

```bash
# Works: kernel-level tracing.
rocprofv3 --kernel-trace -- <your command>
```

!!! danger "Counter collection hung in this fork's environment"
    Counter collection via `rocprofv3 -i <file>` with a `pmc:` line **hung
    indefinitely and uninterruptibly** in this fork's development environment on
    gfx1151, requiring the session to be killed. `rocprof`, `rocprofv2` and
    `rocprofiler-compute` were not installed there.

    This is an observation from one environment, not an established property of
    gfx1151 or of ROCm, and the exact ROCm and `rocprofv3` versions were not
    recorded. Try it if you like -- but start something you are willing to kill.
    If counter collection works in your environment, record the versions and
    say so, because the guidance below assumes it does not.

Where hardware counters are unavailable, combine kernel-trace timings with
compiled resource usage (VGPR and LDS counts from the compiler) and reason about
occupancy analytically from the allocation granularity -- see
[RDNA3.5 constraints](rdna35.md#vgpr-allocation-granularity-and-occupancy).

### CUDA / sm_121a

Standard tooling applies; see [profiling](../profiling.md) for vLLM's
PyTorch-profiler and Nsight Systems workflows.

## Before you claim it works

- [ ] Correctness checked against float64 or an independent implementation,
      with a tolerance justified before the comparison.
- [ ] For selection kernels: selected indices compared, not just scores.
- [ ] Benchmark arms interleaved, not swept sequentially.
- [ ] A parity control included, and it showed parity.
- [ ] Measured at a grid size that saturates the GPU.
- [ ] Checked which backend actually ran — not the one you intended, the one
      that dispatched.

That last point deserves emphasis on these targets. Both of them silently fall
back: gated features select generic paths, and a kernel you believe you are
benchmarking may not be the kernel that ran. Confirm dispatch before
interpreting any number.

## Related pages

- [Numerics: atomics, NaN handling and narrow formats](numerics.md) — the
  specific numerical divergences worth writing targeted tests against.
- [Toolchain, target suffixes and dispatch gating](toolchain.md) — how to
  confirm which backend actually ran.
- [Porting workflow: SM120 to gfx1151](porting_sm120_to_gfx1151.md)

---
name: reset-stack-tuning
description: Tune the reasoning-control stack (reasoning_effort, thinking_token_budget, ReSET entropy-adaptive temperature, phase temperature) on a live vLLM reasoning deployment to reduce reasoning-token burn at fixed accuracy. Use when asked to tune or calibrate ReSET parameters (entropy_threshold, temperature_low/high, reset_window), reduce reasoning token burn, or run estonia/LAVD-style benchmark arms against a reasoning model.
---

# Reasoning-Stack Tuning

Tune five layered mechanisms that jointly control reasoning cost and quality.
The layers interact non-linearly under serving noise, so a naive joint grid is
both unaffordable and unresolvable. This skill encodes the causal hierarchy
that reduces the problem to a 2D frontier, plus the measurement discipline
that keeps the evidence valid.

## 0. The stack and its exact API surface

All knobs are per-request fields. Exact names — wrong names silently 400:

| Layer | Request field(s) | Granularity | Failure mode |
|---|---|---|---|
| Effort (prompt-level) | `reasoning_effort` | whole request | soft; shifts the entropy distribution downstream layers see |
| Repetition penalty | `repetition_penalty` | per token | **lock at 1.0** — static penalties distort H_t and make entropy thresholding unpredictable |
| Thinking budget | `thinking_token_budget` (B) | per request | hard: forces the end-of-thinking marker; truncates mid-thought |
| ReSET entropy temperature | `entropy_threshold` (tau_0), `temperature_low` (T_low), `temperature_high` (T_high), `reset_window` (w) | per token | over-sharpen → premature commitment (plausible garbage); under-sharpen → exploration spiral |
| Phase temperature | `reasoning_answer_temperature` | answer phase only | cannot reduce thinking burn; mis-set degrades final-answer quality. Requires `thinking_token_budget` to be set |

ReSET decision rule (per token): T_low if H_t < tau_t else T_high, where tau_t
rises to the within-step entropy H_step when the step runs hotter than the
global mean. Paper invariants: **T_high = 1.0, w = 32**. T_high > ~1.3 is
self-destructive (one hot sample poisons context → sustained garbage cascade).

Baseline arm = **omit all ReSET fields entirely** (inert by construction); do
not "disable" via T_low=1.0.

## 1. Measurement rules (non-negotiable)

1. **A/A control first.** Baseline vs itself, same prompts, n≥8. Token-identity
   comparison is invalid on this class of serving stack (batch-composition +
   speculative acceptance nondeterminism even at temperature 0). Robust
   metrics only: accuracy, median/p90/total completion tokens. If the A/A arm
   shows accuracy flips, your n is too small for any conclusion.
2. **Interleave or warm both arms.** Prefix-cache warmth confounds token rates
   and latency; run arms interleaved or pre-warm identically.
3. **Durable connections.** A dropped stream is indistinguishable from a model
   failure in the results. Use a stable endpoint path; if port-forwarding,
   raise the lifetime beyond the longest arm (a 1h cap once silently poisoned
   an arm as zero-token "failures").
4. **Sample sizes.** n=8 is directional. n≥16 to ship a config. Near the
   sharpening cliff (accuracy flips), n≥30 to separate 95% from 85%.
5. **Accuracy must come from a scorer**, not vibes. Use the benchmark's own
   scorer (pass/exact-with-tolerance). Read traces only to *classify* failures
   (see taxonomy), never to pass/fail.

## 2. Causal dimensionality reduction

Seven parameters collapse to a 2D frontier per effort level:

```
LOCKED:     repetition_penalty=1.0, T_high=1.0, w=32        (prune 3)
Layer 1:    reasoning_effort e                                (discrete, outer)
Layer 2:    tau_0(e) = P80(H_t | e)  — passive calibration    (prune 1)
Layer 3:    joint 2D frontier in (B, T_low)                   (the real search)
Layer 4:    reasoning_answer_temperature — last, answer-phase (decoupled)
```

Why this order is load-bearing:

- **effort × tau_0 coupling**: effort changes P(H_t), so a tau_0 calibrated at
  one effort level is invalid at another. Calibrate per effort, never reuse.
- **B × T_low coupling**: T_low sets the median burn; B caps the tail
  (spirals). Tune T_low at fixed B; only revisit B when truncation rate moves.
- **phase temperature is orthogonal to burn**: it only shapes post-thinking
  tokens, so it is tuned last on the chosen config.

## 3. Entropy calibration (Layer 2)

API logprobs on this path are **pre-temper** (raw temp-1.0 distribution), so
H_t is measurable off the wire. With top-k logprobs over vocab V, correct for
the truncated tail (uniform-tail completion):

```
r = max(0, 1 - Σ p_i)                # residual mass outside top-k
H ≈ -Σ p_i ln p_i + r · ln((V-k)/r)  # nats
```

The naive top-k-only sum underestimates high-entropy positions and biases P80
low; the correction assumes a uniform tail (maximum-entropy completion), so
true H lies between the two. Collect H_t from baseline runs of the target
workload at the chosen effort; take the 80th percentile.

## 4. The protocol

**Phase 0 — A/A control.** Baseline vs baseline, n≥8. Records the noise floor.
**Phase 1 — Anchor.** Baseline (no ReSET fields) at effort e, n≥8: accuracy,
median, P90 of *reasoning* tokens.
**Phase 2 — Cap.** B = 1.2 × baseline P90 reasoning tokens. Caps spirals
without truncating legitimate long reasoning.
**Phase 3 — Coordinate sweep.** Fix B and e; step T_low ∈ {0.4, 0.3, 0.2,
0.1}, n≥8 per point, tau_0 from Layer 2. This traces the (B, T_low) frontier
along the T_low axis.
**Phase 4 — Polish.** On the chosen config, tune
`reasoning_answer_temperature` for answer-phase quality only.

### Decision rules (one knob per signal — never move two)

- **Accuracy drops or garbage-class failures appear** → sharpening cliff
  crossed. Revert T_low +0.1 and stop the sweep.
- **Budget-hit rate > 15% with accuracy intact** → B too tight *for this
  T_low*. Raise B (+25%) and re-run this point; do not also change T_low.
- **P90 well under B, accuracy high** → headroom. Step T_low −0.1.
- **`finish_reason=length`** → max_tokens (visible-output cap) exhausted; a
  harness/config issue, not a stack signal. The thinking channel is not
  bounded by max_tokens — watch reasoning-token counts, not just finish
  reason.

## 5. Failure taxonomy (for trace classification)

| Signature | Cause | Response |
|---|---|---|
| Short run, plausible wrong answer | premature commitment (T_low too low) | cliff — revert |
| Very long run, no termination | exploration spiral (T_low too high / no cap) | lower T_low or confirm B engaged |
| reasoning_tokens ≈ B exactly | budget truncation | raise B if accuracy suffered |
| Multilingual bleed / token soup | T_high too high (cascade) | T_high back to 1.0 |
| Zero tokens / transport error | infrastructure | discard run, fix path, rerun arm |

## 6. Harness

`stack_optimizer.py` (alongside this file) runs calibration + trials and emits
per-trial dossiers (JSON + Markdown) for judge review. It needs a scorer to be
useful for accuracy — wire the benchmark's scorer via `--scorer
module:function` (signature: `fn(trace: str) -> bool | None`). Without one it
reports cost metrics only and leaves pass/fail to trace review.

```bash
python stack_optimizer.py --model <served-model-id> --effort high \
    --t-low 0.3 --budget 36000 --trial-id t005 \
    --scorer my_bench:score --prompts my_prompts.json
```

Omit `--tau-0` to profile P80 live at the given effort before the trial.

## 7. Pitfalls (learned from production runs)

- tau_0 copied across effort levels is silently wrong — always recalibrate.
- ReSET's quality gains can *raise* total tokens on hard workloads while
  lowering cost-per-correct-answer — judge on the normalized metric.
- A config that only wins the median but fattens P90 has moved cost into the
  tail; check B engagement before celebrating.
- Reasoning models may answer entirely in the thinking channel (empty visible
  content). Empty content with `finish_reason=stop` is not a failure — score
  the reasoning trace if the scorer supports it.

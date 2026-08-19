# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Differential evidence that ReSET is applied on-device and steers output.

Measurement artifact (not a product gate). The ReSET *decision logic* is proven
bit-exact against the reference oracle in ``test_reset_policy.py``; this checks
the orthogonal fact that the resolved temperature is actually applied by the
live sampler, on whichever model runner is active.

The contract is controllable rather than tautological. A ReSET request carries
``temperature = 1.0`` and would sample at 1.0 if ReSET were a no-op. By setting
``temperature_low == temperature_high == T_FORCED`` every token's ReSET decision
resolves to ``T_FORCED`` regardless of entropy, so a correctly-wired ReSET must
reproduce a *static* ``temperature = T_FORCED`` run and diverge sharply from the
static ``temperature = 1.0`` baseline. A no-op would instead match the 1.0
baseline. Sampled-token Shannon entropy across seeds is the yardstick:

  H(pos) = -sum_c p_c * log2(p_c),  p_c = count(token==c, pos) / N_runs

Low temperature concentrates the distribution (low entropy); temperature 1.0
spreads it (high entropy).
"""

import math
import statistics
from collections import Counter

import pytest

from vllm import LLM, SamplingParams

MODEL = "Qwen/Qwen3-0.6B"
N_SEEDS = 30
MAX_TOKENS = 40
T_FORCED = 0.2

PROMPTS = [
    "The history of the Roman Empire:",
    "The theory of evolution explains that",
    "In a distant future, humanity discovered",
    "The Industrial Revolution transformed society because",
    "The biology of the human brain reveals that",
    "Climate change poses a serious threat to",
]


@pytest.fixture(scope="module")
def llm():
    return LLM(
        model=MODEL,
        gpu_memory_utilization=0.7,
        enforce_eager=True,
        max_model_len=512,
    )


def _pos_metrics(token_lists):
    """token_lists: list of token-id lists (per run). Returns per-position
    (entropy, distinct), computed only over alive runs."""
    max_len = max(len(t) for t in token_lists)
    entropy, distinct = [], []
    for pos in range(max_len):
        tokens = [t[pos] for t in token_lists if pos < len(t)]
        if not tokens:
            entropy.append(float("nan"))
            distinct.append(0)
            continue
        counts = Counter(tokens)
        n = len(tokens)
        ent = -sum((c / n) * math.log2(c / n) for c in counts.values())
        entropy.append(ent)
        distinct.append(len(counts))
    return entropy, distinct


def _run_batch(llm, prompts, params):
    """Run every prompt x seed under params. Returns {prompt: [token_ids per seed]}."""
    results: dict[str, list[list[int]]] = {p: [] for p in prompts}
    for p in prompts:
        for seed in range(N_SEEDS):
            sp = SamplingParams(**params, seed=seed)
            out = llm.generate([p], sp)[0].outputs[0]
            results[p].append(out.token_ids)
    return results


def _mean_entropy(prompt_results):
    """Mean per-position sampled-token entropy across prompts and positions."""
    vals: list[float] = []
    for p, runs in prompt_results.items():
        ent, _ = _pos_metrics(runs)
        vals.extend(e for e in ent[:MAX_TOKENS] if not math.isnan(e))
    return statistics.fmean(vals) if vals else float("nan")


def test_reset_is_applied_and_steers(llm):
    # Baselines: static temperature 1.0 (what a no-op ReSET would sample at)
    # and static temperature T_FORCED (what a correctly-applied ReSET must
    # match, since t_low == t_high pins every token to T_FORCED).
    hi = _mean_entropy(
        _run_batch(llm, PROMPTS, dict(temperature=1.0, max_tokens=MAX_TOKENS))
    )
    lo = _mean_entropy(
        _run_batch(llm, PROMPTS, dict(temperature=T_FORCED, max_tokens=MAX_TOKENS))
    )
    reset = _mean_entropy(
        _run_batch(
            llm,
            PROMPTS,
            dict(
                temperature=1.0,
                temperature_low=T_FORCED,
                temperature_high=T_FORCED,
                entropy_threshold=0.6,
                reset_window=32,
                max_tokens=MAX_TOKENS,
            ),
        )
    )

    print("\n=== ReSET application evidence ===")
    print(
        f"model={MODEL} prompts={len(PROMPTS)} seeds={N_SEEDS} max_tokens={MAX_TOKENS}"
    )
    print(f"static temp=1.0        mean entropy = {hi:.4f} bit")
    print(f"static temp={T_FORCED}        mean entropy = {lo:.4f} bit")
    print(f"reset  t_low=t_high={T_FORCED} mean entropy = {reset:.4f} bit")

    span = hi - lo
    assert span > 0.2, (
        f"baselines are not separated enough to be discriminating: "
        f"temp=1.0 ({hi:.4f}) vs temp={T_FORCED} ({lo:.4f})"
    )
    # ReSET must land on the low-temperature baseline, proving its resolved
    # temperature (not the request's temperature=1.0) reached the sampler.
    assert abs(reset - lo) <= 0.25 * span, (
        f"ReSET entropy ({reset:.4f}) must match static temp={T_FORCED} "
        f"({lo:.4f}), not the temperature=1.0 baseline ({hi:.4f}); a no-op "
        f"ReSET would sit at {hi:.4f}"
    )
    assert hi - reset > 0.5 * span, (
        f"ReSET entropy ({reset:.4f}) must sit well below the temperature=1.0 "
        f"baseline ({hi:.4f})"
    )

    print(
        f"\nVERDICT: PASS — ReSET is applied on-device: forcing t_low=t_high="
        f"{T_FORCED} reproduces static temp={T_FORCED} ({reset:.4f}~{lo:.4f} bit) "
        f"and diverges from temp=1.0 ({hi:.4f} bit); a no-op would match 1.0."
    )

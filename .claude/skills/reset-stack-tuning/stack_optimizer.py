#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Stack optimizer harness for the reset-stack-tuning skill.

Runs tau_0 entropy calibration and multi-parameter trials against an
OpenAI-compatible vLLM endpoint, emitting per-trial dossiers for judge review.

Correct request field names (verified against the serving stack):
  reasoning_effort, thinking_token_budget,
  temperature_low, temperature_high, entropy_threshold, reset_window,
  reasoning_answer_temperature, repetition_penalty.

Baseline = omit all ReSET fields entirely (--baseline), NOT temperature_low=1.0.
"""

import argparse
import importlib
import json
import math
import os
import time

from openai import OpenAI

# Paper invariants — do not sweep these.
T_HIGH = 1.0
RESET_WINDOW = 32


def shannon_entropy(top_logprobs, vocab_size: int) -> float:
    """Entropy of the pre-temper distribution from top-k logprobs (nats).

    Uniform-tail correction: residual mass r spread over the V-k unobserved
    tokens contributes r * ln((V-k)/r). The uncorrected top-k sum
    underestimates high-entropy positions and biases percentile calibration
    low; the corrected value assumes a maximum-entropy tail. True H lies
    between the two.
    """
    k = len(top_logprobs)
    head = 0.0
    mass = 0.0
    for entry in top_logprobs:
        p = math.exp(entry.logprob)
        head -= p * entry.logprob
        mass += p
    r = max(0.0, 1.0 - mass)
    tail = r * math.log((vocab_size - k) / r) if r > 0 and vocab_size > k else 0.0
    return head + tail


def iter_logprob_entries(resp):
    logprobs = resp.choices[0].logprobs
    if logprobs is None:
        return
    for token_data in logprobs.content or []:
        if token_data.top_logprobs:
            yield token_data.top_logprobs


def profile_tau_zero(client, model, effort, prompts, vocab_size, percentile=80):
    """Derive tau_0 as the entropy percentile conditional on effort level.

    Baseline sampling: no ReSET fields, repetition penalty off.
    """
    entropies = []
    for prompt in prompts:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=1.0,
            logprobs=True,
            top_logprobs=20,
            extra_body={
                "reasoning_effort": effort,
                "repetition_penalty": 1.0,
            },
        )
        for entries in iter_logprob_entries(resp):
            entropies.append(shannon_entropy(entries, vocab_size))
    if not entropies:
        raise RuntimeError("no logprobs returned; cannot profile tau_0")
    entropies.sort()
    rank = int(math.ceil(percentile / 100 * len(entropies))) - 1
    tau_0 = entropies[min(len(entropies) - 1, rank)]
    print(
        f"[profile] effort={effort} tokens={len(entropies)} "
        f"P{percentile} tau_0={tau_0:.4f} nats"
    )
    return tau_0


def load_scorer(spec: str | None):
    """--scorer module:function, signature fn(trace: str) -> bool | None."""
    if not spec:
        return None
    module_name, func_name = spec.rsplit(":", 1)
    return getattr(importlib.import_module(module_name), func_name)


def build_extra_body(config, baseline: bool) -> dict:
    body = {
        "reasoning_effort": config["effort"],
        "thinking_token_budget": config["budget"],
        "repetition_penalty": 1.0,
    }
    if not baseline:
        body["entropy_threshold"] = config["tau_0"]
        body["temperature_low"] = config["t_low"]
        body["temperature_high"] = T_HIGH
        body["reset_window"] = RESET_WINDOW
    if config.get("phase_temp"):
        # Answer-phase temperature requires the budget guard to be set.
        body["reasoning_answer_temperature"] = config["phase_temp"]
    return body


def run_trial(client, model, config, prompts, scorer, baseline=False):
    extra_body = build_extra_body(config, baseline)
    results = []
    for idx, prompt in enumerate(prompts):
        start = time.time()
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            extra_body=extra_body,
        )
        latency = time.time() - start
        choice = resp.choices[0]
        usage = resp.usage
        details = getattr(usage, "completion_tokens_details", None)
        reasoning_tokens = getattr(details, "reasoning_tokens", None)
        trace = choice.message.content or ""
        results.append(
            {
                "prompt_id": idx,
                "finish_reason": choice.finish_reason,
                "completion_tokens": usage.completion_tokens,
                "reasoning_tokens": reasoning_tokens,
                "latency_sec": round(latency, 2),
                "score": scorer(trace) if scorer else None,
                "trace": trace,
            }
        )
    return results


def percentile(sorted_vals, pct):
    if not sorted_vals:
        return None
    rank = int(math.ceil(pct / 100 * len(sorted_vals))) - 1
    return sorted_vals[min(len(sorted_vals) - 1, rank)]


def analyze_trial(config, raw_results, baseline=False):
    tokens = sorted(r["completion_tokens"] for r in raw_results)
    n = len(raw_results)
    scored = [r["score"] for r in raw_results if r["score"] is not None]
    budget = config.get("budget")
    budget_hits = sum(
        1
        for r in raw_results
        if budget
        and r["reasoning_tokens"] is not None
        and r["reasoning_tokens"] >= budget
    )
    return {
        "config": config,
        "baseline": baseline,
        "metrics": {
            "runs": n,
            "accuracy": round(sum(scored) / len(scored), 3) if scored else None,
            "scored_runs": len(scored),
            "median_tokens": percentile(tokens, 50),
            "p90_tokens": percentile(tokens, 90),
            "mean_tokens": round(sum(tokens) / n) if n else None,
            "total_tokens": sum(tokens),
            # length = visible-output cap hit: a harness/config issue, not a
            # stack signal. budget_hit = thinking-channel truncation.
            "length_exhaustion_rate": round(
                sum(1 for r in raw_results if r["finish_reason"] == "length") / n, 3
            ),
            "budget_hit_rate": round(budget_hits / n, 3) if budget else None,
        },
        "raw_results": raw_results,
    }


def save_dossier(summary, trial_id, out_dir="trials"):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"trial_{trial_id}.json"), "w") as f:
        json.dump(summary, f, indent=2)
    md_path = os.path.join(out_dir, f"trial_{trial_id}.md")
    with open(md_path, "w") as f:
        f.write(
            f"# Trial {trial_id}\n\n```json\n"
            f"{json.dumps(summary['config'], indent=2)}\n```\n\n"
        )
        f.write(
            f"## Metrics\n\n```json\n"
            f"{json.dumps(summary['metrics'], indent=2)}\n```\n\n"
        )
        f.write(
            "## Traces (failure classification only — scores come from the scorer)\n\n"
        )
        for res in summary["raw_results"]:
            f.write(
                f"### Prompt {res['prompt_id']} | "
                f"completion={res['completion_tokens']} "
                f"reasoning={res['reasoning_tokens']} "
                f"finish={res['finish_reason']} score={res['score']}\n\n"
                f"```text\n{res['trace']}\n```\n\n---\n\n"
            )
    print(f"[done] {md_path}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, help="served model id")
    p.add_argument(
        "--prompts", required=True, help="JSON file with a list of prompt strings"
    )
    p.add_argument("--effort", default="high")
    p.add_argument(
        "--budget",
        type=int,
        required=True,
        help="thinking_token_budget (B); anchor at 1.2x baseline P90",
    )
    p.add_argument("--t-low", type=float, default=0.3)
    p.add_argument(
        "--tau-0",
        type=float,
        default=None,
        help="omit to profile P80 entropy live at --effort",
    )
    p.add_argument(
        "--phase-temp",
        type=float,
        default=None,
        help="reasoning_answer_temperature; tune last",
    )
    p.add_argument(
        "--vocab-size",
        type=int,
        default=163840,
        help="for the entropy tail correction; check model config",
    )
    p.add_argument(
        "--baseline",
        action="store_true",
        help="omit all ReSET fields (the A/A and anchor arm)",
    )
    p.add_argument("--scorer", default=None, help="module:function trace scorer")
    p.add_argument("--trial-id", default="001")
    args = p.parse_args()

    with open(args.prompts) as f:
        prompts = json.load(f)

    client = OpenAI()  # OPENAI_API_KEY / OPENAI_BASE_URL from env
    scorer = load_scorer(args.scorer)

    tau_0 = args.tau_0
    if tau_0 is None and not args.baseline:
        tau_0 = profile_tau_zero(
            client, args.model, args.effort, prompts, args.vocab_size
        )

    config = {
        "effort": args.effort,
        "budget": args.budget,
        "tau_0": tau_0,
        "t_low": None if args.baseline else args.t_low,
        "t_high": None if args.baseline else T_HIGH,
        "reset_window": None if args.baseline else RESET_WINDOW,
        "phase_temp": args.phase_temp,
    }
    raw = run_trial(client, args.model, config, prompts, scorer, baseline=args.baseline)
    save_dossier(analyze_trial(config, raw, baseline=args.baseline), args.trial_id)


if __name__ == "__main__":
    main()

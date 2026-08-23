#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Sentinel driver for nvfp4 KV-cache recovery via reasoning controls.

Runs a builtin bench profile sequentially against a served endpoint and reports
correctness, wall time and completion-token burn. Grading is delegated to
llm-inference-bench so results are directly comparable to the fp8 baselines.

Objectives, in priority order:
  1. always correct   -- correct_frac must be 1.0
  2. minimise time    -- seconds_total
  3. minimise tokens  -- completion_tokens_total

`score` encodes that lexicographic order so a sweep can minimise one number:
correct runs score on time, incorrect runs are pushed above any correct run.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
import urllib.error
import urllib.request

BENCH_DIR = "/home/naadir/go/src/github.com/local-inference-lab/llm-inference-bench"
INCORRECT_FLOOR = 1e9


def _load_bench():
    if BENCH_DIR not in sys.path:
        sys.path.insert(0, BENCH_DIR)
    import llm_decode_bench

    return llm_decode_bench


def build_request(args, prompt: str, index: int = 0) -> dict:
    body: dict = {
        "model": args.model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "seed": args.seed_base + index,
    }
    if not args.omit_temperature:
        body["temperature"] = args.temperature
    if args.max_tokens > 0:
        body["max_completion_tokens"] = args.max_tokens
    if args.top_p is not None:
        body["top_p"] = args.top_p
    if args.top_k is not None:
        body["top_k"] = args.top_k
    if args.effort:
        # Top-level field: build_chat_params merges it as the override side of
        # merge_kwargs(chat_template_kwargs, extra_kwargs), so placing it in
        # chat_template_kwargs would silently lose to the server default.
        body["reasoning_effort"] = args.effort
    if args.marker_penalty is not None:
        body["reasoning_marker_penalty"] = args.marker_penalty
    if args.monitor:
        body["reasoning_monitor"] = True
    if args.extra:
        # Arbitrary fork controls: entropy_threshold, reset_window,
        # temperature_high/low, reasoning_answer_temperature,
        # repetition_penalty, repetition_detection, ...
        body.update(json.loads(args.extra))
    return body


def post(url: str, body: dict, timeout: float, auth: str = "") -> dict:
    payload = json.dumps(body).encode()
    headers = {"Content-Type": "application/json"}
    if auth:
        headers["Authorization"] = f"Bearer {auth}"
    req = urllib.request.Request(url, data=payload, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def run_trial(args, prompt: str, profile: dict, bench, index: int) -> dict:
    url = args.endpoint.rstrip("/") + "/v1/chat/completions"
    started = time.monotonic()
    try:
        raw = post(url, build_request(args, prompt, index), args.timeout, args.auth)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")[:300]
        return {
            "index": index,
            "ok": False,
            "error": f"HTTP {exc.code}: {detail}",
            "seconds": time.monotonic() - started,
            "tokens": 0,
            "correct": False,
        }
    except Exception as exc:  # noqa: BLE001 - report and continue the sweep
        return {
            "index": index,
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
            "seconds": time.monotonic() - started,
            "tokens": 0,
            "correct": False,
        }
    seconds = time.monotonic() - started

    choice = (raw.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    content = message.get("content") or ""
    reasoning = message.get("reasoning_content") or ""
    usage = raw.get("usage") or {}
    verdict = bench.score_completion_profile(
        profile=profile,
        final_answer=content,
        content_text=content,
        output_text=reasoning + content,
        regex=profile.get("correct_regex", ""),
        source=profile.get("score_source", "content"),
        finish_reason=choice.get("finish_reason", ""),
    )
    return {
        "index": index,
        "ok": True,
        "correct": bool(verdict.get("correct")),
        "seconds": seconds,
        "tokens": int(usage.get("completion_tokens") or 0),
        "prompt_tokens": int(usage.get("prompt_tokens") or 0),
        "reasoning_chars": len(reasoning),
        "answer_chars": len(content),
        "finish_reason": choice.get("finish_reason", ""),
        "verdict": {k: v for k, v in verdict.items() if k != "detail"},
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    env = os.environ.get
    ap.add_argument("--endpoint", default=env("ENDPOINT", "http://127.0.0.1:8888"))
    ap.add_argument("--model", default=env("MODEL", "deepseek-v4-flash-0731"))
    ap.add_argument("--profile", default=env("PROFILE", "lavd-test"))
    ap.add_argument("--runs", type=int, default=int(env("RUNS", "4")))
    ap.add_argument("--timeout", type=float, default=float(env("TIMEOUT", "3600")))
    ap.add_argument(
        "--temperature", type=float, default=float(env("TEMPERATURE", "1.0"))
    )
    ap.add_argument(
        "--omit-temperature",
        action="store_true",
        default=env("OMIT_TEMPERATURE") == "1",
        help="Send no temperature field, exercising the server/proxy default path",
    )
    ap.add_argument("--auth", default=env("AUTH", ""), help="Bearer token")
    ap.add_argument("--top-p", type=float, default=_optional_float("TOP_P"))
    ap.add_argument("--top-k", type=int, default=_optional_int("TOP_K"))
    ap.add_argument("--max-tokens", type=int, default=int(env("MAX_TOKENS", "60000")))
    ap.add_argument("--seed-base", type=int, default=int(env("SEED_BASE", "0")))
    ap.add_argument("--effort", default=env("EFFORT", ""))
    ap.add_argument(
        "--marker-penalty", type=float, default=_optional_float("MARKER_PENALTY")
    )
    ap.add_argument("--monitor", action="store_true", default=env("MONITOR") == "1")
    ap.add_argument("--extra", default=env("EXTRA", ""), help="JSON of fork controls")
    ap.add_argument("--label", default=os.environ.get("LABEL", "run"))
    ap.add_argument("--jsonl", default=os.environ.get("JSONL", ""))
    args = ap.parse_args()

    bench = _load_bench()
    name = bench.normalize_builtin_test_profile_name(args.profile)
    prompt, _encoding, profile = bench.decode_builtin_test_profile_prompt(name)

    print(
        f"# label={args.label} profile={name} runs={args.runs} "
        f"effort={args.effort or '(server default)'} temp={args.temperature} "
        f"top_p={args.top_p} marker_penalty={args.marker_penalty} "
        f"monitor={int(args.monitor)} max_tokens={args.max_tokens}",
        flush=True,
    )

    results = []
    for i in range(args.runs):
        row = run_trial(args, prompt, profile, bench, i)
        results.append(row)
        if row["ok"]:
            print(
                f"=== run {i} correct={int(row['correct'])} "
                f"seconds={row['seconds']:.1f} tokens={row['tokens']} "
                f"prompt_tokens={row['prompt_tokens']} "
                f"answer_chars={row['answer_chars']} finish={row['finish_reason']}",
                flush=True,
            )
        else:
            print(f"=== run {i} FAILED {row['error']}", flush=True)
        if args.jsonl:
            with open(args.jsonl, "a", encoding="utf-8") as handle:
                handle.write(json.dumps({"label": args.label, **row}) + "\n")

    completed = [r for r in results if r["ok"]]
    correct = [r for r in completed if r["correct"]]
    correct_frac = len(correct) / len(results) if results else 0.0
    seconds_total = sum(r["seconds"] for r in results)
    tokens_total = sum(r["tokens"] for r in results)

    if correct_frac >= 1.0 and completed:
        score = seconds_total
    else:
        score = INCORRECT_FLOOR + (1.0 - correct_frac) * 1e6

    print(f"METRIC label={args.label}")
    print(f"METRIC correct_frac={correct_frac:.4f}")
    print(f"METRIC correct_count={len(correct)}/{len(results)}")
    print(f"METRIC seconds_total={seconds_total:.2f}")
    print(f"METRIC completion_tokens_total={tokens_total}")
    if completed:
        print(
            "METRIC seconds_median="
            f"{statistics.median(r['seconds'] for r in completed):.2f}"
        )
        print(
            "METRIC completion_tokens_median="
            f"{int(statistics.median(r['tokens'] for r in completed))}"
        )
    print(f"METRIC score={score:.2f}")
    return 0


def _optional_float(env: str) -> float | None:
    raw = os.environ.get(env, "")
    return float(raw) if raw else None


def _optional_int(env: str) -> int | None:
    raw = os.environ.get(env, "")
    return int(raw) if raw else None


if __name__ == "__main__":
    raise SystemExit(main())

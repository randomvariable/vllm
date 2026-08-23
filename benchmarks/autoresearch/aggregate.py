#!/usr/bin/env python3
"""Aggregate sentinel trial JSONL shards into autoresearch METRIC lines.

Objective ordering is fixed by the campaign: correctness first, then wall
time, then token burn. That is encoded lexicographically in ``score``:

* any incorrect or errored trial pushes ``score`` above ``INCORRECT_FLOOR``,
  scaled by how many trials were wrong, so no time/token win can ever buy
  its way past a correctness regression;
* an all-correct arm scores its total wall seconds, with total completion
  tokens folded in at 1e-6 weight as a tiebreak only.
"""

from __future__ import annotations

import json
import statistics
import sys

INCORRECT_FLOOR = 1e9


def main(paths: list[str]) -> int:
    rows: list[dict] = []
    for path in paths:
        try:
            with open(path, encoding="utf-8") as handle:
                rows.extend(json.loads(line) for line in handle if line.strip())
        except FileNotFoundError:
            pass

    if not rows:
        print("ERROR: no trial rows collected", file=sys.stderr)
        return 1

    errored = [r for r in rows if not r.get("ok")]
    correct = [r for r in rows if r.get("correct")]
    correct_frac = len(correct) / len(rows)
    seconds_total = sum(float(r.get("seconds") or 0.0) for r in rows)
    tokens_total = sum(int(r.get("tokens") or 0) for r in rows)

    if correct_frac >= 1.0:
        score = seconds_total + 1e-6 * tokens_total
    else:
        score = INCORRECT_FLOOR + (1.0 - correct_frac) * 1e6

    print(f"METRIC score={score:.4f}")
    print(f"METRIC correct_frac={correct_frac:.4f}")
    print(f"METRIC correct_count={len(correct)}")
    print(f"METRIC trials={len(rows)}")
    print(f"METRIC errored={len(errored)}")
    print(f"METRIC seconds_total={seconds_total:.2f}")
    print(f"METRIC completion_tokens_total={tokens_total}")
    print(
        "METRIC seconds_median="
        f"{statistics.median(float(r['seconds']) for r in rows):.2f}"
    )
    print(
        "METRIC completion_tokens_median="
        f"{int(statistics.median(int(r.get('tokens') or 0) for r in rows))}"
    )

    for row in sorted(rows, key=lambda r: (r.get("label", ""), r.get("index", 0))):
        if row.get("ok"):
            print(
                f"ASI trial_{row.get('label')}_{row.get('index')}="
                f"correct:{int(bool(row.get('correct')))}"
                f",s:{float(row.get('seconds') or 0):.0f}"
                f",tok:{int(row.get('tokens') or 0)}"
                f",finish:{row.get('finish_reason', '')}"
            )
        else:
            print(
                f"ASI trial_{row.get('label')}_{row.get('index')}="
                f"error:{str(row.get('error'))[:80]}"
            )

    return 1 if errored else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

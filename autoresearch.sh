#!/usr/bin/env bash
# ReSET tuning harness for the live deepseek-v4-flash deployment.
#
# Workload: the `hotel-lights` sentinel (175-token prompt, single-integer
# answer graded by the bench `numeric_exact` scorer). It is the established
# failing sentinel for this campaign: correct answers need a ~13.4K reasoning
# token floor, and every uniform reasoning-shortening control found so far
# buys speed by dropping below that floor and answering confidently wrong.
#
# Tunable surface: benchmarks/autoresearch/arm.json — a JSON object of
# request-level fork controls merged into every trial body. ReSET lives here
# (entropy_threshold, reset_window, temperature_high, temperature_low), along
# with reasoning_answer_temperature, reasoning_marker_penalty,
# repetition_penalty and friends. All are per-request sampling params exposed
# on the chat protocol, so an arm change takes effect with no rebuild.
#
# Determinism: fixed profile, fixed trial count, fixed per-trial seeds
# (0..N-1), no network beyond the two local tunnels.
#
# Emits `METRIC score=` as the primary objective. score is lexicographic:
# correctness dominates, then wall seconds, with completion tokens as a
# 1e-6-weighted tiebreak.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

HARNESS=benchmarks/autoresearch
ARM_FILE=${ARM_FILE:-$HARNESS/arm.json}
PROFILE=${PROFILE:-hotel-lights}
MODEL=${MODEL:-deepseek-v4-flash-0731}
MAX_TOKENS=${MAX_TOKENS:-60000}
TEMPERATURE=${TEMPERATURE:-1.0}
RUNS_PER_ENDPOINT=${RUNS_PER_ENDPOINT:-2}
TIMEOUT=${TIMEOUT:-1800}
ENDPOINT_A=${ENDPOINT_A:-http://127.0.0.1:8898}
ENDPOINT_B=${ENDPOINT_B:-http://127.0.0.1:8897}
PYTHON=${PYTHON:-$PWD/.venv/bin/python}

ARM=$(tr -d '\n' <"$ARM_FILE")
if ! printf '%s' "$ARM" | $PYTHON -c 'import json,sys; json.loads(sys.stdin.read())'; then
    echo "FATAL: $ARM_FILE is not valid JSON" >&2
    exit 1
fi

# Both TP pairs must be live: arms are split across them so pod identity never
# correlates with the treatment (the confound that bit the earlier sweeps).
for ep in "$ENDPOINT_A" "$ENDPOINT_B"; do
    if ! curl -sf -m 10 "$ep/v1/models" >/dev/null; then
        echo "FATAL: endpoint $ep is not serving" >&2
        exit 1
    fi
done

WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT

echo "# arm=$ARM profile=$PROFILE runs=$((RUNS_PER_ENDPOINT * 2)) max_tokens=$MAX_TOKENS"

run_shard() {
    local endpoint=$1 label=$2 seed_base=$3 out=$4
    ENDPOINT="$endpoint" \
    MODEL="$MODEL" \
    PROFILE="$PROFILE" \
    RUNS="$RUNS_PER_ENDPOINT" \
    SEED_BASE="$seed_base" \
    MAX_TOKENS="$MAX_TOKENS" \
    TEMPERATURE="$TEMPERATURE" \
    TIMEOUT="$TIMEOUT" \
    EXTRA="$ARM" \
    LABEL="$label" \
    JSONL="$out" \
        $PYTHON "$HARNESS/sentinel.py" 2>&1 |
        grep -v '^METRIC ' | sed "s/^/[$label] /"
}

run_shard "$ENDPOINT_A" a 0 "$WORK/a.jsonl" &
pid_a=$!
run_shard "$ENDPOINT_B" b "$RUNS_PER_ENDPOINT" "$WORK/b.jsonl" &
pid_b=$!

rc=0
wait "$pid_a" || rc=1
wait "$pid_b" || rc=1
if [[ $rc -ne 0 ]]; then
    echo "FATAL: a sentinel shard exited non-zero" >&2
    exit 1
fi

$PYTHON "$HARNESS/aggregate.py" "$WORK/a.jsonl" "$WORK/b.jsonl"

#!/usr/bin/env bash
# Model-check the shm_broadcast specs with TLC (exhaustive) and Apalache
# (bounded symbolic). Usage: check.sh [tlc|apalache|all] [jobs]
#
# Checks run in parallel (default 12 jobs; each is a JVM of up to 4 GiB), the
# slowest first; the whole suite takes about 2-3 minutes on 16 cores.
# Apalache bounds: a configuration expected to fail gets the length of its
# TLC counterexample, so Apalache must find a violation of the same depth; a
# configuration expected to hold is checked for every behavior of up to 10
# steps (6 for inference.qnt, whose formula grows fastest). TLC covers the
# full state space of every configuration. "-" skips Apalache where the
# counterexample is too deep for it within the suite's time (hwm1: 3-5
# minutes; 20 or more steps: over an hour); hwm1OneReader is the
# Apalache-sized copy of hwm1.
set -uo pipefail
cd "$(dirname "$0")"
backend=${1:-all}
max_jobs=${2:-12}
mkdir -p .runs .tmp
export TMPDIR=$PWD/.tmp JAVA_TOOL_OPTIONS="-Djava.io.tmpdir=$PWD/.tmp"

# spec module invariants expected(ok|violation) apalache-steps
checks=(
  "paths pathsFixed inOrder,noLoss,progress ok 10"
  "message_queue futexFenced noLostWakeup,ringSafe ok 10"
  "message_queue futexUnfenced noLostWakeup,ringSafe violation 9"
  "paths pathsCancel inOrder,noLoss,progress violation 12"
  "message_queue skipIdleFenced noLostWakeup,ringSafe ok 10"
  "message_queue hwm1NoThrottle noLostWakeup,ringSafe ok 10"
  "message_queue unlimitedHandshake noLostWakeup,ringSafe ok 10"
  "inference fixedDecode noLostWakeup,ringSafe,progress ok 6"
  "inference fixedAllReply noLostWakeup,ringSafe,progress ok 6"
  "inference backpressureAtBound noLostWakeup,ringSafe,progress ok 6"
  "message_queue hwm1OneReader noLostWakeup,ringSafe violation 14"
  "message_queue hwm1 noLostWakeup,ringSafe violation -"
  "message_queue hwm4Slow noLostWakeup,ringSafe violation -"
  "message_queue hwm4Burst noLostWakeup,ringSafe violation -"
  "message_queue hwm2Slow noLostWakeup,ringSafe violation -"
  "message_queue pr58917 noLostWakeup,ringSafe violation 5"
  "message_queue unlimitedNoHandshake noLostWakeup,ringSafe violation 5"
  "message_queue skipIdleUnfenced noLostWakeup,ringSafe violation 9"
  "message_queue skipIdleFlaglessPark noLostWakeup,ringSafe violation 8"
  "inference hwm1Decode noLostWakeup,ringSafe,progress violation 4"
  "inference raceDecode noLostWakeup,ringSafe,progress violation 4"
  "inference backpressureOver noLostWakeup,ringSafe,progress violation -"
  "paths pathsCurrent inOrder,noLoss,progress violation 3"
  "paths pathsNoDrop inOrder,noLoss,progress violation 4"
  "fanin faninTwo noStuckWait ok 10"
  "fanin faninFour noStuckWait ok 10"
  "fanin faninBroken noStuckWait violation 8"
)

# run <id> <backend> <spec> <module> <invariants> <expected> <steps>
run() {
  local id=$1 be=$2 spec=$3 main=$4 invs=$5 want=$6 steps=$7 out got text
  local -a inv
  IFS=, read -ra inv <<<"$invs"
  out=$PWD/.runs/$be-$main.txt
  local start=$SECONDS
  # Every quint verify starts its own Apalache server (the TLC backend uses
  # it to compile), so each check needs a private port, and a private working
  # and temp directory: TLC unpacks its standard modules into java.io.tmpdir.
  local dir=$PWD/.runs/work-$id
  rm -rf "$dir" && mkdir -p "$dir/tmp"
  local -a cmd=(quint verify "../../$spec.qnt" --main "$main"
    --server-endpoint "localhost:$((8900 + id))" --invariants "${inv[@]}")
  if [[ $be == tlc ]]; then
    cmd+=(--backend tlc)
  else
    cmd+=(--max-steps "$steps")
  fi
  (cd "$dir" && TMPDIR=$dir/tmp JAVA_TOOL_OPTIONS="-Djava.io.tmpdir=$dir/tmp" \
    "${cmd[@]}") >"$out" 2>&1
  rm -rf "$dir"
  # Classify by the reported outcome, not the exit code: quint also exits 1
  # on tool errors.
  text=$(<"$out")
  if [[ $text == *"[ok]"* ]]; then
    got=ok
  elif [[ $text == *"[violation]"* ]]; then
    got=violation
  else
    got=error
  fi
  printf '%-9s %-22s %-10s want %-10s %5ss %s\n' "$be" "$main" "$got" "$want" \
    $((SECONDS - start)) "$([[ $got == "$want" ]] && echo PASS || echo FAIL)" \
    >".runs/$id.result"
}

rm -f .runs/*.result
id=0
for c in "${checks[@]}"; do
  read -r spec main invs want steps <<<"$c"
  for be in tlc apalache; do
    [[ $backend == "$be" || $backend == all ]] || continue
    [[ $be == apalache && $steps == - ]] && continue
    id=$((id + 1))
    while (($(jobs -rp | wc -l) >= max_jobs)); do
      wait -n
    done
    run "$id" "$be" "$spec" "$main" "$invs" "$want" "$steps" &
  done
done
wait

status=0
for ((i = 1; i <= id; i++)); do
  cat ".runs/$i.result"
  [[ $(<".runs/$i.result") == *PASS ]] || status=1
done
exit $status

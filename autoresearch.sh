#!/usr/bin/env bash
# Autoresearch benchmark: shm_broadcast MessageQueue latency vs CPU.
# Runs benchmarks/benchmark_shm_broadcast.py (executor -> N workers -> replies,
# lock-step) with the spinloop extension built, so the production path
# (parked-only pings, native fence) is measured.
#
# Primary metric: rtt_us = geometric mean of round-trip p50 over the paced
# scenarios (1 ms and 10 ms step gaps, 1 and 4 workers) - the latency an idle
# executor loop sees per step.
# Secondary metrics give the CPU side of the Pareto front and guard the
# back-to-back path.
set -euo pipefail
cd "$(dirname "$0")"
ROOT=$PWD
PY=$ROOT/.venv/bin/python
OUT=$HOME/.cache/sb
mkdir -p "$OUT"
rm -f "$OUT/results.jsonl"

INC=$("$PY" -c 'import sysconfig; print(sysconfig.get_paths()["include"])')
# -mmwaitx is an x86 ISA flag (enables the MONITORX/MWAITX intrinsics); it is
# meaningless and rejected by aarch64 g++, where the extension uses WFET
# (runtime-gated on HWCAP2) instead. Select the arch flag explicitly.
case "$(uname -m)" in
  x86_64 | i386 | i686) ARCH_FLAGS=(-mmwaitx) ;;
  *) ARCH_FLAGS=() ;;
esac
g++ -O2 -shared -fPIC "${ARCH_FLAGS[@]}" -DPy_LIMITED_API=0x030B0000 -I"$INC" \
  csrc/spinloop.cpp -o vllm/spinloop.abi3.so
trap 'rm -f "$ROOT/vllm/spinloop.abi3.so"' EXIT


# Announce pre-wake knobs. The executor's collective_rpc calls
# MessageQueue.announce() before its pre-send work; it is a no-op unless
# VLLM_SHM_BROADCAST_ANNOUNCE_MAX_MS is set, so 0/0 measures the plain path.
# Defaults are the paired-validated win on the benchmark host (rtt -7%,
# 4-worker 10ms cell -16%, worker CPU -29%): a 0.5 ms expectant cap and a
# 200 us lead that spans the method-pickle + enqueue serialization. Override
# both to 0 to run without the hint.
: "${VLLM_SHM_BROADCAST_ANNOUNCE_MAX_MS:=0.5}"
: "${ANNOUNCE_LEAD_US:=200}"
export VLLM_SHM_BROADCAST_ANNOUNCE_MAX_MS
echo "[ar] announce: cap=${VLLM_SHM_BROADCAST_ANNOUNCE_MAX_MS}ms lead=${ANNOUNCE_LEAD_US}us"

TMPDIR=$OUT PYTHONPATH=$ROOT VLLM_TARGET_DEVICE=cpu \
  VLLM_SHM_BROADCAST_ANNOUNCE_MAX_MS="$VLLM_SHM_BROADCAST_ANNOUNCE_MAX_MS" "$PY" \
  benchmarks/benchmark_shm_broadcast.py --variant ar --repeat 1 \
  --readers 1,4 --gaps 0,1,10 --idle-seconds 3 --target-wall 3 \
  --min-steps 300 --max-steps 300000 --json-out "$OUT/results.jsonl" \
  --announce-lead-us "$ANNOUNCE_LEAD_US" \
  >"$OUT/bench.log" 2>&1 || { tail -40 "$OUT/bench.log"; exit 1; }

"$PY" - "$OUT/results.jsonl" <<'EOF'
import json, math, sys

recs = [json.loads(line) for line in open(sys.argv[1])]
by = {(r["readers"], r["scenario"]): r for r in recs}
assert all(r["full_fence"] for r in recs), "spinloop extension not loaded"


def cores(r, key):
    return r[key] / r["window_s"]


paced = [by[(n, s)]["rtt"]["p50"] for n in (1, 4) for s in ("gap_1ms", "gap_10ms")]
print(f"METRIC rtt_us={math.exp(sum(map(math.log, paced)) / len(paced)):.2f}")
for n in (1, 4):
    for s in ("gap_1ms", "gap_10ms"):
        r = by[(n, s)]
        tag = f"r{n}_{s.removeprefix('gap_')}"
        print(f"METRIC {tag}_rtt_p50_us={r['rtt']['p50']:.1f}")
        print(f"METRIC {tag}_rtt_p99_us={r['rtt']['p99']:.1f}")
        print(f"METRIC {tag}_worker_cores={cores(r, 'workers_cpu_s'):.4f}")
        print(f"METRIC {tag}_stall_max_us={r['rtt']['max']:.0f}")
    b2b = by[(n, "gap_0ms")]
    print(f"METRIC r{n}_b2b_steps_per_s={b2b['steps_per_s']:.0f}")
    print(f"METRIC r{n}_b2b_rtt_p50_us={b2b['rtt']['p50']:.1f}")
    idle = by[(n, "idle_post")]
    print(f"METRIC r{n}_idle_worker_cores={cores(idle, 'workers_cpu_s'):.4f}")
# Paced CPU summary for the Pareto front: mean worker cores over paced cells.
pc = [cores(by[(n, s)], "workers_cpu_s") / n for n in (1, 4)
      for s in ("gap_1ms", "gap_10ms")]
print(f"METRIC paced_cores_per_worker={sum(pc) / len(pc):.4f}")
EOF

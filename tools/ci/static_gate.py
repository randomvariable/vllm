#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pre-build static gate: catch drift that would only show up as a crashed pod.

Both image pipelines run this before the (multi-hour) build so that a broken
call site or an undeclared environment variable fails in seconds instead of
after a full build plus a rollout.

Two checks, matching the two ways this fork has actually broken at runtime:

``envs``
    ``vllm/envs.py`` defines a module-level ``__getattr__``, so a typo'd or
    never-declared ``envs.SOMETHING`` type-checks clean and raises
    ``AttributeError`` at runtime. ``check_envs_refs.py`` reads the declared
    names out of ``envs.py`` itself and fails closed.

``drift``
    Cross-module signature and attribute drift, via the repo's own mypy hook.
    Only ``call-arg`` and ``attr-defined`` are gated: those are the codes that
    raise ``TypeError``/``AttributeError`` at import or engine init. Other codes
    in these trees are pre-existing debt, so they are reported but do not fail
    the gate, which keeps it actionable instead of permanently red.

The mypy scope is the fork-owned trees, i.e. the code a rebase churns.
Upstream-owned trees are excluded: they use vLLM's dynamic-attribute idioms
(``Parameter.weight_loader`` and friends) that mypy flags by design.
"""

from __future__ import annotations

import glob
import pathlib
import subprocess
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]

GATED_CODES = ("[call-arg]", "[attr-defined]")

FORK_OWNED = (
    "vllm/v1/worker/gpu/**/*.py",
    "vllm/v1/core/**/*.py",
    "vllm/model_executor/layers/fused_moe/**/*.py",
    "vllm/model_executor/kernels/**/*.py",
    "vllm/models/qwen4_exp/**/*.py",
)


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args, cwd=REPO_ROOT, capture_output=True, text=True, check=False
    )


def check_envs() -> int:
    files = sorted(str(p) for p in (REPO_ROOT / "vllm").rglob("*.py"))
    print(f"[envs] checking {len(files)} files for undeclared envs references")
    result = _run(sys.executable, "tools/pre_commit/check_envs_refs.py", *files)
    sys.stdout.write(result.stdout)
    sys.stderr.write(result.stderr)
    if result.returncode == 0:
        print("[envs] OK")
    return result.returncode


def check_drift() -> int:
    files: set[str] = set()
    for pattern in FORK_OWNED:
        files.update(glob.glob(pattern, recursive=True, root_dir=REPO_ROOT))
    if not files:
        print("[drift] FAIL: scope matched no files (globs are stale)")
        return 1

    print(f"[drift] type-checking {len(files)} fork-owned files")
    result = _run(sys.executable, "tools/pre_commit/mypy.py", "local", *sorted(files))
    output = result.stdout + result.stderr
    crashers = [line for line in output.splitlines() if line.endswith(GATED_CODES)]
    summary = [line for line in output.splitlines() if line.startswith("Found ")]
    for line in summary:
        print(f"[drift] mypy: {line}")
    if crashers:
        print("[drift] FAIL: signature/attribute drift that crashes at runtime:")
        for line in crashers:
            print(f"  {line}")
        return 1
    print("[drift] OK: no call-arg/attr-defined drift")
    return 0


def main() -> int:
    return check_envs() | check_drift()


if __name__ == "__main__":
    sys.exit(main())

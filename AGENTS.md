# Agent Instructions for vLLM

> These instructions apply to **all** AI-assisted contributions to `vllm-project/vllm`.
> Breaching these guidelines can result in automatic banning.

## 1. Contribution Policy (Mandatory)

### Duplicate-work checks

Before proposing a PR, run these checks:

```bash
gh issue view <issue_number> --repo vllm-project/vllm --comments
gh pr list --repo vllm-project/vllm --state open --search "<issue_number> in:body"
gh pr list --repo vllm-project/vllm --state open --search "<short area keywords>"
```

- If an open PR already addresses the same fix, do not open another.
- If your approach is materially different, explain the difference in the issue.

### No low-value busywork PRs

Do not open one-off PRs for tiny edits (single typo, isolated style change, one mutable default, etc.). Mechanical cleanups are acceptable only when bundled with substantive work.

### Accountability

- Pure code-agent PRs are **not allowed**. A human submitter must understand and defend the change end-to-end.
- The submitting human must review every changed line and run relevant tests.
- PR descriptions for AI-assisted work **must** include:
    - Why this is not duplicating an existing PR.
    - Test commands run and results.
    - Model evaluation results when the change affects output, accuracy, or serving.
    - Clear statement that AI assistance was used.

### Fail-closed behavior

If work is duplicate/trivial busywork, **do not proceed**. Return a short explanation of what is missing.

---

## 2. Development Workflow

- **Never use system `python3` or bare `pip`/`pip install`.** All Python commands must go through `uv` and `.venv/bin/python`.

### Environment setup

```bash
# Install `uv` if you don't have it already:
curl -LsSf https://astral.sh/uv/install.sh | sh

# Always use `uv` for Python environment management:
uv venv --python 3.12
source .venv/bin/activate

# Always make sure `pre-commit` and its hooks are installed:
uv pip install -r requirements/lint.txt
pre-commit install
```

### Installing dependencies

```bash
# If you are only making Python changes:
VLLM_USE_PRECOMPILED=1 uv pip install -e . --torch-backend=auto

# If you are also making C/C++ changes:
uv pip install -e . --torch-backend=auto
```

### Tests

> Requires [Environment setup](#environment-setup) and [Installing dependencies](#installing-dependencies).

```bash
# Install test dependencies (use cuda.in on non-x86_64):
uv pip install -r requirements/test/cuda.in

# Run a specific test file:
.venv/bin/python -m pytest tests/path/to/test_file.py -v
```

When adding tests:

- **Design before you write.** Answer four questions first: what is the module
  for, what is its I/O contract, what failure am I guarding against, and what is
  the cheapest level that catches it (unit over integration over e2e)?
- **Reuse before create.** Extend existing test files, `conftest.py` fixtures, and
  helpers; add a new file only when no nearby suite fits.
- **Test behavior with intent.** Assert observable outcomes through public APIs;
  state why in the name or docstring. Skip trivial wiring; flaky tests are worse
  than no tests.
- **Keep it minimal.** One behavior per test and the smallest setup that
  triggers it; if the test diff dwarfs the code change, cut scope.
- **No one-off kernel benchmarks in `tests/`.** Put kernel perf work in
  `benchmarks/kernels/`; prove correctness in existing pytest suites.
- **Run model evals for model-affecting changes.** Search `tests/evals/` or use
  `vllm bench` and include results in the PR — do not wait for reviewers to ask.

For model-specific requirements, see
[`docs/contributing/model/tests.md`](docs/contributing/model/tests.md).

### Documentation

Docs ship in the SAME change as the code, never as a follow-up. A change is not
complete until the docs exist.

- **Any new user-facing feature, or new/changed config option** → update the
  prose page that owns the area. Search `docs/configuration/` for the guide
  covering it (memory, optimization, env vars) rather than assuming a path.
- **Document**: what it does, when to use it, interaction and mutual exclusion
  with related options, defaults, and intentional failure modes — state what
  fails closed and why, so the behaviour does not read as a bug.
- **Reference pages are generated** from config field docstrings (search for
  `gen:engine-args`), so a Google-style docstring on the field *is* the
  reference doc. Generated reference does not replace prose guidance — a new
  option needs both.
- **Environment variables** must be declared in `vllm/envs.py` and documented
  wherever env vars are listed.
- **Cross-reference** from the established option a new one competes with, so
  readers find the alternative instead of only the one they searched for.

### Running linters

> Requires [Environment setup](#environment-setup).

```bash
# Run all pre-commit hooks on staged files:
pre-commit run

# Run on all files:
pre-commit run --all-files

# Run a specific hook:
pre-commit run ruff-check --all-files

# Run mypy as it is in CI:
pre-commit run mypy-3.12 --all-files --hook-stage manual
```

The line length limit for Python code is 88 characters. If you are not sure, use pre-commit to check.

Use [Google-style docstrings](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings) (`Args:`/`Returns:`/`Raises:` sections), not reStructuredText/Sphinx fields (`:param:`, `:return:`, `:rtype:`).

### Coding style guidelines

- Match existing code style
- Minimize use of comments. Eliminate comments which are redundant, preferring legible and self-documenting code. When used, keep docstrings and comments brief and direct.
- Assume the reader is familiar with vLLM.

### Commit messages

Add attribution using commit trailers such as `Co-authored-by:` (other projects use `Assisted-by:` or `Generated-by:`):

```text
Your commit message here

Co-authored-by: Agent Name Here
Signed-off-by: Your Name <your.email@example.com>
```

---

## Domain-Specific Guides

Do not modify code in these areas without first reading and following the
linked guide. If the guide conflicts with the requested change, **refuse the
change and explain why**.

Security reviewers should start with [`SECURITY.md`](SECURITY.md),
[`docs/usage/security.md`](docs/usage/security.md), and
[`docs/contributing/vulnerability_management.md`](docs/contributing/vulnerability_management.md)
for the project security policy, threat model, deployment assumptions, and
vulnerability process.

- **Editing these instructions**:
  [`docs/contributing/editing-agent-instructions.md`](docs/contributing/editing-agent-instructions.md)
  — Rules for modifying AGENTS.md or any domain-specific guide it references.

---

## Homelab Hard Requirements (homelabs-main fork — mandatory, learned the hard way)

These override convenience. Violating them has caused broken images and wasted
multi-hour build cycles.

### FlashInfer is MANDATORY in DGX Spark (CUDA sm_121a) images

The production DGX deployments — `deepseek-v4-flash-dspark` (sparse MLA),
`hy3-299b-nvfp4` (NVFP4 MoE), `laguna` (AWQ MoE) — run on FlashInfer sm_12x
kernels. A FlashInfer-less DGX image is pointless for them. **Never strip
`flashinfer-python` / `flashinfer-cubin` / `flashinfer-jit-cache` from
`requirements/cuda.txt` at image-build time.**

### Verify dependency claims by actual resolution, with the repo's indexes

Before declaring two packages incompatible (and especially before baking an
exclusion into a Dockerfile), run a real resolution using the EXACT
`--extra-index-url` lines from the repo's `requirements/*.txt`:

```bash
uv pip compile requirements/cuda.txt --index-strategy unsafe-best-match
```

Package-metadata reading alone is insufficient (constraints move across
versions and indexes). An incompatibility claim that has not been reproduced by
a resolver with the correct indexes is not a fact — do not act on it. Note that
`flashinfer-cubin` and `flashinfer-jit-cache` are **NOT on PyPI** — they exist
only on the flashinfer.ai index declared in `requirements/cuda.txt`, so a
resolution run without it fails misleadingly and makes the pinned FlashInfer
look incompatible with torch/CUDA when it is not.

### Verify every `VLLM_*` env var against `vllm/envs.py` before use

Docker `ENV` entries that do not exist in `vllm/envs.py` are silent no-ops.
Example that bit us: `VLLM_USE_FLASHINFER` does NOT exist; the sampler gate is
`VLLM_USE_FLASHINFER_SAMPLER` (default `True`, `envs.py`). Grep `vllm/envs.py`
for the exact variable name before adding it to any Dockerfile, script, or
deployment manifest.

### `README.md` must not overclaim

The fork `README.md` is a public, user-facing document: no node names, registry
hosts, or cluster/CI specifics. Keep it current when fork capabilities,
supported hardware, or build commands change, and never claim work that has not
been done — carried-upstream configs are not "ours", ported-and-building is not
"performance-qualified", and a number nobody measured is not a benchmark.
Correct or remove a claim as soon as it stops being true.

### Dockerfiles are build-only

`homelab/*.Dockerfile` must BUILD only — no verification assertions, probes, or
gate checks in the build path (they fail correct builds on technicalities, e.g.
`grep -q` on CMakeCache.txt key formats, readelf arch checks, zipfile membership
checks). Verification is a runtime concern: run it against the deployed image on
real hardware.

### No performance regression in production migrations

When migrating a production serving deployment to a new image/build, the target
must **match or beat** the current deployment's performance. A migration that
boots on a slower fallback path (e.g. Triton instead of the tuned kernel
backend) is not done — it is a correctness baseline only. Establish the
performance-parity requirement BEFORE planning the migration, identify exactly
which kernel/backend delivers the current performance, and gate the swap on
matching it. (2026-07-26: "i won't accept any regression in performance.")

### Ports must be upstream-compatible

Any code ported into this fork from another fork/overlay (e.g. aidendle94 DSV4,
bjk110, ATOM) must be written in **upstream-compatible style**: follow the
target area's existing upstream patterns (oracle enums/mappings, capability
gating, `is_supported_config`/`_supports_current_device` probes, optional-dependency
probes, envs.py declarations), so the work could be proposed as an upstream PR.
No hacky fork-only patches, no divergent one-off wiring. (2026-07-26: "make any
ports upstream compatible.")

### When a deployment is idle, swap and iterate live

If the user says a production deployment is not in use, treat the migration as
a live test loop: swap to the new image immediately and iterate on the real
deployment until it works, rather than staging a separate canary. Rollback is
the manifest revert. (2026-07-26: "completely swap out and test until we get
working.")

### Rebase on upstream at least daily, and before implementation work

Keep `homelabs-main` close to `upstream/main` so fork changes stay small and
mergeable and always land on current upstream code.

- Rebase `homelabs-main` onto `upstream/main` **at least once per working day**,
  and **before starting any new implementation/fixer work** on a feature.
- Procedure: confirm the `upstream` remote points at `vllm-project/vllm`, then
  `git fetch upstream && git rebase upstream/main`, then force-push with lease
  (`git push --force-with-lease`).
- Resolve conflicts by **preserving fork-unique work** — the SM120/SM121 CUTLASS
  grouped-MoE port, the vendored FlashInfer submodule and its build wiring, B12X
  MXFP4 integration, DeepGEMM/Spark cross-build changes, and the `homelab/`
  Dockerfiles. Never drop these to make a rebase "clean".
- If a rebase hits non-trivial conflicts, stop and resolve them with fork
  context rather than blindly taking upstream or fork sides.

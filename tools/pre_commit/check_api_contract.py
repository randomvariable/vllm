#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Detect call sites whose keyword arguments violate the callee's signature.

This is the fast, compile-free guard against signature drift: when an API
changes its accepted kwargs (e.g. ``AutoWeightsLoader`` dropping
``skip_prefixes``/``skip_substrs`` in #53106), every stale call site fails
here in seconds instead of failing a multi-hour device build, or worse,
crashing model init at serve time on hardware.

How it works, fully static (no package imports, works with a bare
``python3`` and fails closed):

1. The defining module of each KNOWN_CALLABLES entry is located inside this
   repository and parsed with ``ast``; signatures of the listed names
   (functions, or ``__init__`` for classes) are extracted, including
   ``*args``-style VAR_KEYWORD detection.
2. Every ``.py`` file passed on the command line is parsed and each
   ``Call`` node whose callee's final name segment matches a known callable
   is checked: every explicit keyword must be a parameter of the callee
   (unless the callee accepts ``**kwargs``).
3. ``**kwargs`` unpacking at the call site is skipped (mypy's job). If a
   defining module cannot be located or parsed, the check FAILS - never
   silently skips.
"""

import ast
import sys
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# defining module -> callable names to contract-check.
KNOWN_CALLABLES: dict[str, set[str]] = {
    "vllm.model_executor.models.utils": {
        "AutoWeightsLoader",  # __init__ kwargs; skip_* removed in #53106
        "WeightsMapper",
        "make_layers",
        "make_empty_intermediate_tensors_factory",
        "maybe_fuse_shared_experts",
        "maybe_prefix",
    },
    "vllm.model_executor.layers.vocab_parallel_embedding": {
        "VocabParallelEmbedding",
        "ParallelLMHead",
    },
    "vllm.model_executor.layers.linear": {
        "ColumnParallelLinear",
        "RowParallelLinear",
        "MergedColumnParallelLinear",
        "QKVParallelLinear",
    },
    "vllm.model_executor.layers.logits_processor": {"LogitsProcessor"},
    "vllm.v1.worker.gpu.dp_utils": {
        "dispatch_cg_and_sync_dp",
        "sync_cudagraph_and_dp_padding",
    },
}


@dataclass
class Signature:
    params: set[str] = field(default_factory=set)
    has_var_kw: bool = False
    # *args present: positional-count check unreliable
    has_var_pos: bool = False
    # ordered positional-or-keyword parameter names (self/cls stripped)
    pos_params: list[str] = field(default_factory=list)
    # keyword-only parameter names
    kwonly_params: list[str] = field(default_factory=list)
    # parameter names that carry a default value
    defaults: set[str] = field(default_factory=set)


def _signature_of(node: ast.AST) -> Signature | None:
    if isinstance(node, ast.ClassDef):
        if _is_dataclass(node):
            # dataclasses synthesize __init__ from annotated fields; defaults
            # are not tracked (assume all acceptable to keep the check sound).
            fields = [
                a.target.id
                for a in node.body
                if isinstance(a, ast.AnnAssign) and isinstance(a.target, ast.Name)
            ]
            sig = Signature(
                params=set(fields),
                has_var_kw=False,
                pos_params=fields,
                defaults=set(fields),
            )
            return sig
        init = next(
            (
                n
                for n in node.body
                if isinstance(n, ast.FunctionDef) and n.name == "__init__"
            ),
            None,
        )
        if init is None:
            return None
        node = init
    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return None
    args = node.args
    posall = (*args.posonlyargs, *args.args)
    pos_params = [a.arg for a in posall]
    if pos_params and pos_params[0] in ("self", "cls"):
        pos_params = pos_params[1:]
    n_defaults = len(args.defaults)
    tail_posall = posall[-n_defaults:] if n_defaults else ()
    defaults = {a.arg for a in tail_posall}
    defaults.update(
        a.arg for a, d in zip(args.kwonlyargs, args.kw_defaults) if d is not None
    )
    return Signature(
        params={a.arg for a in (*posall, *args.kwonlyargs)},
        has_var_kw=args.kwarg is not None,
        has_var_pos=args.vararg is not None,
        pos_params=pos_params,
        kwonly_params=[a.arg for a in args.kwonlyargs],
        defaults=defaults,
    )


def _is_dataclass(node: ast.ClassDef) -> bool:
    return any(
        (isinstance(d, ast.Name) and d.id == "dataclass")
        or (
            isinstance(d, ast.Call)
            and isinstance(d.func, ast.Name)
            and d.func.id == "dataclass"
        )
        for d in node.decorator_list
    )


def _module_signatures(module: str) -> dict[str, Signature]:
    path = REPO_ROOT.joinpath(*module.split("."))
    for candidate in (path.with_suffix(".py"), path / "__init__.py"):
        if candidate.exists():
            break
    else:
        raise SystemExit(
            f"{Path(__file__).name}: cannot locate defining module "
            f"'{module}' inside the repository - fix KNOWN_CALLABLES"
        )
    try:
        tree = ast.parse(candidate.read_text(encoding="utf-8"), filename=str(candidate))
    except SyntaxError as e:
        raise SystemExit(
            f"{Path(__file__).name}: defining module '{module}' "
            f"({candidate}) does not parse: {e}"
        ) from e
    wanted = KNOWN_CALLABLES[module]
    out: dict[str, Signature] = {}
    for node in tree.body:
        if (
            isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name in wanted
        ):
            out[node.name] = _signature_of(node)
    missing = wanted - out.keys()
    if missing:
        raise SystemExit(
            f"{Path(__file__).name}: '{module}' no longer defines "
            f"{sorted(missing)} - fix KNOWN_CALLABLES"
        )
    return out


def _callee_name(node: ast.Call) -> str | None:
    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def main() -> int:
    signatures: dict[str, Signature] = {}
    for module in KNOWN_CALLABLES:
        signatures.update(_module_signatures(module))

    findings: list[str] = []
    for path in sys.argv[1:]:
        try:
            tree = ast.parse(Path(path).read_text(encoding="utf-8"), filename=path)
        except SyntaxError:
            continue  # syntax handled by the compile/ruff lanes
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if any(kw.arg is None for kw in node.keywords):
                continue  # **kwargs unpacking; delegated to mypy
            name = _callee_name(node)
            if name is None or name not in signatures:
                continue
            sig = signatures[name]
            if sig.has_var_kw:
                continue
            npos = len(node.args)
            kw_args = {kw.arg for kw in node.keywords}
            for kw in node.keywords:
                if kw.arg not in sig.params:
                    findings.append(
                        f"{path}:{node.lineno}: {name}() got an unexpected "
                        f"keyword argument '{kw.arg}'"
                    )
            if (
                npos > len(sig.pos_params)
                and not sig.has_var_kw
                and not sig.has_var_pos
            ):
                findings.append(
                    f"{path}:{node.lineno}: {name}() received {npos} "
                    f"positional arguments but takes at most "
                    f"{len(sig.pos_params)}"
                )
            for i, p in enumerate(sig.pos_params, start=1):
                if i > npos and p not in kw_args and p not in sig.defaults:
                    findings.append(
                        f"{path}:{node.lineno}: {name}() missing required "
                        f"positional argument '{p}'"
                    )
            for p in sig.kwonly_params:
                if p not in kw_args and p not in sig.defaults:
                    findings.append(
                        f"{path}:{node.lineno}: {name}() missing required "
                        f"keyword-only argument '{p}'"
                    )

    for line in findings:
        print(f"\033[91merror:\033[0m {line}", file=sys.stderr)
    if findings:
        print(
            "\nAPI signature drift detected. Update the call sites to the "
            "current signature, or extend KNOWN_CALLABLES in "
            "tools/pre_commit/check_api_contract.py.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

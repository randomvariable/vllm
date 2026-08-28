# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reject references to environment variables that `vllm/envs.py` never declares.

`vllm/envs.py` resolves attributes through a module-level `__getattr__`, so a
typo or an undeclared knob type-checks cleanly and only fails at runtime, when
the reading code path is first executed. mypy cannot see through that
indirection; this check closes exactly that gap and nothing else.

Declared names are read out of `envs.py` itself, so the check needs no
allowlist and stays correct as knobs are added or removed.

Usage:
    python tools/pre_commit/check_envs_refs.py <files...>
"""

import ast
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
ENVS_PATH = REPO_ROOT / "vllm" / "envs.py"

# Guards against silently passing everything if envs.py is restructured and the
# extraction below stops matching.
MIN_EXPECTED_DECLARATIONS = 50


def _declared_names() -> set[str]:
    """Names resolvable as `envs.<NAME>`: lazy-map keys plus module globals."""
    tree = ast.parse(ENVS_PATH.read_text(encoding="utf-8"), filename=str(ENVS_PATH))
    names: set[str] = set()
    for node in tree.body:
        # `environment_variables: dict[str, Callable[[], Any]] = {...}` and any
        # other module-level dict whose keys are the lazily resolved knobs.
        targets = []
        if isinstance(node, ast.AnnAssign) and node.value is not None:
            targets = [node.target]
            value = node.value
        elif isinstance(node, ast.Assign):
            targets = node.targets
            value = node.value
        else:
            value = None
        if value is None:
            continue
        if isinstance(value, ast.Dict):
            for key in value.keys:
                if isinstance(key, ast.Constant) and isinstance(key.value, str):
                    names.add(key.value)
        for target in targets:
            # Eagerly defined module constants are resolvable too.
            if isinstance(target, ast.Name):
                names.add(target.id)
    # TYPE_CHECKING annotations declare the public surface for type checkers.
    for node in ast.walk(tree):
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
    return names


def _is_knob(name: str) -> bool:
    return name.isupper() and not name.startswith("_")


def check_file(path: str, declared: set[str]) -> list[str]:
    try:
        tree = ast.parse(Path(path).read_text(encoding="utf-8"), filename=path)
    except SyntaxError as exc:
        return [f"{path}:{exc.lineno}: syntax error: {exc.msg}"]

    # Track the local aliases that refer to the envs module.
    aliases = {"envs"}
    findings = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "vllm.envs" and alias.asname:
                    aliases.add(alias.asname)
        elif isinstance(node, ast.ImportFrom):
            if node.module == "vllm" and any(a.name == "envs" for a in node.names):
                aliases.update(
                    a.asname or a.name for a in node.names if a.name == "envs"
                )
            elif node.module == "vllm.envs":
                for alias in node.names:
                    if _is_knob(alias.name) and alias.name not in declared:
                        findings.append(
                            f"{path}:{node.lineno}: 'vllm.envs' does not declare "
                            f"'{alias.name}' (add it to envs.py)"
                        )
        elif (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id in aliases
            and _is_knob(node.attr)
            and node.attr not in declared
        ):
            findings.append(
                f"{path}:{node.lineno}: 'vllm.envs' does not declare "
                f"'{node.attr}' (add it to envs.py)"
            )
    return findings


def main() -> int:
    if not ENVS_PATH.is_file():
        print(f"error: {ENVS_PATH} not found", file=sys.stderr)
        return 2
    try:
        declared = _declared_names()
    except SyntaxError as exc:
        print(f"error: cannot parse {ENVS_PATH}: {exc}", file=sys.stderr)
        return 2
    if len(declared) < MIN_EXPECTED_DECLARATIONS:
        print(
            f"error: only {len(declared)} env declarations found in {ENVS_PATH}; "
            "the extraction is stale — fix _declared_names() before relying on "
            "this check",
            file=sys.stderr,
        )
        return 2

    findings = []
    for path in sys.argv[1:]:
        findings.extend(check_file(path, declared))
    for finding in findings:
        print(finding)
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main())

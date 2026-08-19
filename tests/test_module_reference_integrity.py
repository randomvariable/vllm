# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Guard against imports of ``vllm`` modules that do not exist.

Two engine-startup crashes have shipped from the same mistake: an upstream
merge takes a module deletion but leaves a fork-local import of it behind. The
reference resolves at collection time nowhere in the CPU test suite, because
the importing module is only loaded on a GPU worker, so the failure first
appears as a crash-looping serving pod.

The check is static on purpose. Importing every module would need a GPU and a
built extension tree; reading the import graph needs neither, and it is the
part that regressed.
"""

import ast
from pathlib import Path

import pytest

VLLM_ROOT = Path(__file__).resolve().parent.parent / "vllm"

# Modules that exist only after a build: compiled extensions, vendored
# third-party trees installed into the wheel, and optional out-of-tree
# accelerator packages. Absence from a source checkout says nothing.
_BUILD_GENERATED_PREFIXES = (
    "vllm.third_party.",
    "vllm.vllm_flash_attn.",
)

_BUILD_GENERATED_EXACT = {
    "vllm.cumem_allocator",
    "vllm.fs_io_C",
    "vllm.spinloop",
}


def _is_build_generated(module: str) -> bool:
    if module in _BUILD_GENERATED_EXACT:
        return True
    if module.startswith(_BUILD_GENERATED_PREFIXES):
        return True
    # Compiled extensions are conventionally underscore-prefixed: vllm._C,
    # vllm._moe_C_stable_libtorch, vllm._qutlass_C.
    return module.split(".")[-1].startswith("_")


def _guarded_line_ranges(tree: ast.AST) -> list[range]:
    """Line spans of ``try`` bodies, where an absent module is handled."""
    return [
        range(node.body[0].lineno, node.body[-1].end_lineno + 1)
        for node in ast.walk(tree)
        if isinstance(node, ast.Try) and node.body and node.body[-1].end_lineno
    ]


def _imported_vllm_modules(tree: ast.AST) -> list[tuple[int, str]]:
    guarded = _guarded_line_ranges(tree)
    found: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Import | ast.ImportFrom):
            continue
        modules: list[str] = []
        if isinstance(node, ast.ImportFrom):
            # Relative imports resolve against the package, not the tree root.
            if node.level == 0 and node.module and node.module.startswith("vllm."):
                modules.append(node.module)
        else:
            modules += [
                alias.name for alias in node.names if alias.name.startswith("vllm.")
            ]
        if not modules:
            continue
        if any(node.lineno in span for span in guarded):
            # An optional import that handles its own absence is not a stale
            # reference; requiring the module to exist would ban the pattern.
            continue
        found += [(node.lineno, module) for module in modules]
    return found


def test_every_vllm_module_reference_resolves():
    """Every ``vllm.*`` import names a module present in the source tree."""
    repo_root = VLLM_ROOT.parent
    unresolved: list[str] = []

    for path in sorted(VLLM_ROOT.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError as exc:  # pragma: no cover - a syntax error is its own bug
            pytest.fail(f"{path.relative_to(repo_root)}: {exc}")

        for lineno, module in _imported_vllm_modules(tree):
            if _is_build_generated(module):
                continue
            relative = Path(*module.split("."))
            exists = (repo_root / relative).with_suffix(".py").exists() or (
                repo_root / relative / "__init__.py"
            ).exists()
            if not exists:
                unresolved.append(
                    f"{path.relative_to(repo_root)}:{lineno} imports {module}"
                )

    assert not unresolved, "imports name modules that do not exist:\n" + "\n".join(
        unresolved
    )

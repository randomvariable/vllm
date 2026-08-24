# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Guard against references to ``vllm`` modules, symbols, and enum members that
do not exist.

Five engine-startup crashes have shipped from the same family of mistake: an
upstream merge deletes or renames something and leaves a fork-local reference
behind. The reference resolves at collection time nowhere in the CPU test
suite, because the referring module is only loaded on a GPU worker, so the
failure first appears as a crash-looping serving pod. Observed variants:

* a deleted module (``vllm.v1.attention.ops.dcp_utils``);
* a symbol moved between modules (``cp_lse_ag_out_rs``, which the upstream
  consolidation of ``ops/common.py`` moved into ``ops/dcp.py``);
* an enum member split in two (``Mxfp4MoeBackend.B12X`` becoming
  ``B12X_MXFP4_MXFP8`` and ``B12X_MXFP4_BF16``).

The checks are static on purpose. Importing every module would need a GPU and
a built extension tree; reading the source graph needs neither, and it is the
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


def _module_source(repo_root: Path, module: str) -> Path | None:
    relative = Path(*module.split("."))
    candidate = (repo_root / relative).with_suffix(".py")
    if candidate.exists():
        return candidate
    package = repo_root / relative / "__init__.py"
    return package if package.exists() else None


def _bound_names(tree: ast.Module) -> set[str] | None:
    """Names a module binds at import time, or ``None`` when it binds them lazily.

    Two patterns put the export set out of static reach: ``from x import *``
    defers to another module's ``__all__``, and a module-level ``__getattr__``
    resolves attributes on demand (``vllm.utils.humming`` is a facade over an
    optional package). Those modules are skipped rather than guessed at.
    """
    if any(
        isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
        and node.name == "__getattr__"
        for node in tree.body
    ):
        return None
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            names |= {
                target.id for target in node.targets if isinstance(target, ast.Name)
            }
            names |= {
                element.id
                for target in node.targets
                if isinstance(target, ast.Tuple)
                for element in target.elts
                if isinstance(element, ast.Name)
            }
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
        elif isinstance(node, ast.Import | ast.ImportFrom):
            for alias in node.names:
                if alias.name == "*":
                    return None
                names.add(alias.asname or alias.name.split(".")[0])
    return names


def test_every_vllm_symbol_import_resolves():
    """``from vllm.x import Name`` names something ``vllm.x`` actually binds."""
    repo_root = VLLM_ROOT.parent
    exports: dict[str, set[str] | None] = {}
    unresolved: list[str] = []

    for path in sorted(VLLM_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        guarded = _guarded_line_ranges(tree)
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom):
                continue
            if node.level != 0 or not node.module:
                continue
            if not node.module.startswith("vllm.") and node.module != "vllm":
                continue
            if _is_build_generated(node.module):
                continue
            if any(node.lineno in span for span in guarded):
                continue
            if node.module not in exports:
                source = _module_source(repo_root, node.module)
                exports[node.module] = (
                    _bound_names(ast.parse(source.read_text(encoding="utf-8")))
                    if source is not None
                    else None
                )
            bound = exports[node.module]
            if bound is None:
                continue
            for alias in node.names:
                if alias.name == "*" or alias.name in bound:
                    continue
                # A dotted import target may be a submodule rather than a name.
                if _module_source(repo_root, f"{node.module}.{alias.name}") is not None:
                    continue
                unresolved.append(
                    f"{path.relative_to(repo_root)}:{node.lineno} imports "
                    f"{alias.name} from {node.module}"
                )

    assert not unresolved, (
        "imports name symbols their module does not define:\n" + "\n".join(unresolved)
    )


_ENUM_BASES = frozenset({"Enum", "IntEnum", "StrEnum", "IntFlag", "Flag"})


def _enum_members(node: ast.ClassDef) -> set[str]:
    members: set[str] = set()
    for statement in node.body:
        if isinstance(statement, ast.Assign):
            members |= {
                target.id
                for target in statement.targets
                if isinstance(target, ast.Name)
            }
        elif isinstance(statement, ast.AnnAssign) and isinstance(
            statement.target, ast.Name
        ):
            members.add(statement.target.id)
        elif isinstance(statement, ast.FunctionDef | ast.AsyncFunctionDef):
            members.add(statement.name)
    return members


def test_every_enum_member_reference_resolves():
    """``SomeEnum.MEMBER`` names a member that enum declares.

    The enum name is resolved per file — defined there, or imported from a
    ``vllm`` module that defines it — so a reference is never matched against
    a same-named enum the file never sees (``RequestType`` exists both in
    vLLM and in the optional ``lmcache`` package).
    """
    repo_root = VLLM_ROOT.parent
    trees = {
        path: ast.parse(path.read_text(encoding="utf-8"))
        for path in sorted(VLLM_ROOT.rglob("*.py"))
    }

    by_module: dict[str, dict[str, set[str]]] = {}
    for path, tree in trees.items():
        module = ".".join(path.relative_to(repo_root).with_suffix("").parts)
        module = module.removesuffix(".__init__")
        declared: dict[str, set[str]] = {}
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            bases = {
                base.id if isinstance(base, ast.Name) else getattr(base, "attr", "")
                for base in node.bases
            }
            if bases & _ENUM_BASES:
                declared[node.name] = _enum_members(node)
        by_module[module] = declared

    unresolved: list[str] = []
    for path, tree in trees.items():
        module = ".".join(path.relative_to(repo_root).with_suffix("").parts)
        visible = dict(by_module[module.removesuffix(".__init__")])
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom) or node.level != 0:
                continue
            imported = by_module.get(node.module or "")
            if not imported:
                continue
            for alias in node.names:
                if alias.name in imported:
                    visible[alias.asname or alias.name] = imported[alias.name]

        for node in ast.walk(tree):
            if not isinstance(node, ast.Attribute):
                continue
            if not isinstance(node.value, ast.Name):
                continue
            members = visible.get(node.value.id)
            # Member names are upper-case by convention; anything else is a
            # method, a dunder, or an attribute on a member's value.
            if members is None or not node.attr.isupper():
                continue
            if node.attr not in members:
                unresolved.append(
                    f"{path.relative_to(repo_root)}:{node.lineno} references "
                    f"{node.value.id}.{node.attr}"
                )

    assert not unresolved, (
        "references name enum members that do not exist:\n" + "\n".join(unresolved)
    )

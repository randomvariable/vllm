#!/usr/bin/env bash
# Generate compile_commands.json so clangd can resolve csrc/ in the editor.
#
# vLLM does not set CMAKE_EXPORT_COMPILE_COMMANDS for the GPU extensions (only
# cmake/cpu_extension.cmake does), so the file is never produced by a build.
# ninja can emit an equivalent database from an already-configured tree, which
# costs nothing and needs no recompile.
#
# The paths need fixing up afterwards. The HIP extensions are compiled from a
# hipified copy of csrc/ inside the build tree, so most entries name
# /vllm-build/cmake/csrc/*.hip rather than the csrc/*.cu file you actually
# edit. Those are rewritten back to the checkout where the original exists,
# which is what makes go-to-definition work on the real sources.

set -euo pipefail

BUILD_ROOT="${VLLM_DEV_BUILD_ROOT:-/vllm-build}"
SRC="/src/vllm"
OUT="$SRC/compile_commands.json"

if [[ ! -f /etc/profile.d/vllm-hip-toolchain.sh ]]; then
    echo "postattach: not inside the devtools image; skipping clangd setup" >&2
    exit 0
fi
# shellcheck source=/dev/null
. /etc/profile.d/vllm-hip-toolchain.sh

ninja_dir="$(find "$BUILD_ROOT" -name build.ninja -printf '%h\n' 2>/dev/null | head -1 || true)"
if [[ -z "$ninja_dir" ]]; then
    echo "postattach: no configured build tree under $BUILD_ROOT." >&2
    echo "            run 'vllm-hip-build' once, then re-attach, for clangd support." >&2
    exit 0
fi

raw="$(mktemp)"
trap 'rm -f "$raw"' EXIT
( cd "$ninja_dir" && ninja -t compdb ) > "$raw"

BUILD_ROOT="$BUILD_ROOT" SRC="$SRC" RAW="$raw" OUT="$OUT" python3 - <<'PY'
import json
import os
import pathlib

raw = pathlib.Path(os.environ["RAW"])
src = pathlib.Path(os.environ["SRC"])
build_root = pathlib.Path(os.environ["BUILD_ROOT"])
out = pathlib.Path(os.environ["OUT"])

entries = json.loads(raw.read_text())
kept, remapped = [], 0

for entry in entries:
    path = pathlib.Path(entry.get("file", ""))
    # Link steps list the .so as their input; clangd has no use for them.
    if path.suffix in {".so", ""}:
        continue

    if not path.is_absolute() or build_root not in path.parents:
        kept.append(entry)
        continue

    # Map the hipified copy back onto the source it was generated from.
    for anchor in ("cmake", "csrc"):
        try:
            rel = path.relative_to(build_root / anchor)
        except ValueError:
            continue
        for suffix in (path.suffix, ".cu", ".cpp"):
            candidate = (src / rel).with_suffix(suffix)
            if candidate.exists():
                entry["file"] = str(candidate)
                remapped += 1
                break
        break
    kept.append(entry)

out.write_text(json.dumps(kept, indent=1) + "\n")
in_src = sum(1 for e in kept if e["file"].startswith(str(src)))
print(f"postattach: wrote {out} ({len(kept)} entries, {in_src} resolve to the checkout, {remapped} remapped from the hipified tree)")
PY

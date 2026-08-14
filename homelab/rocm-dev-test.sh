#!/usr/bin/env bash
# Run tests against a real gfx1151 (Strix Halo) GPU inside a ROCm container.
#
# Why a container: a host virtualenv can install torch and the ROCm Core SDK
# device package and still fail every kernel launch with hipErrorInvalidImage,
# because the torch build's HIP code objects do not match the host runtime.
# rocminfo reporting the agent correctly is not sufficient. The runtime image
# already contains a matching torch, ROCm and compiled vLLM, so use it.
#
# Usage:
#   homelab/rocm-dev-test.sh tests/kernels/attention/test_prefix_prefill.py -k sliding_window
#   IMAGE=vllm-strix-runtime:local homelab/rocm-dev-test.sh tests/...
#
# Pure-Python files from this checkout are copied over the installed package, so
# this checkout is what runs, committed or not. C++/HIP changes DO need a
# rebuilt image.
#
# The overlay is verified and fails loudly. It has to be: an overlay that
# quietly copies nothing runs the image's own vLLM, and a test suite that
# passes against stale code looks exactly like a test suite that passes.

set -euo pipefail

IMAGE="${IMAGE:-vllm-strix-runtime:local}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [ "$#" -eq 0 ]; then
    echo "usage: ${BASH_SOURCE[0]##*/} <pytest args...>" >&2
    exit 2
fi

echo "warning: ${BASH_SOURCE[0]##*/} is deprecated; use cargo make test -- <pytest args...>" >&2
echo "warning: compatibility wrapper preserves legacy argument semantics" >&2

for dev in /dev/kfd /dev/dri; do
    if [ ! -e "$dev" ]; then
        echo "error: $dev missing; this host has no usable AMD GPU" >&2
        exit 1
    fi
done

if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
    echo "error: image '$IMAGE' not found locally." >&2
    echo "build it with: docker build -f homelab/strix.Dockerfile -t $IMAGE ." >&2
    exit 1
fi

# --security-opt seccomp=unconfined is required for HSA; --ipc=host avoids
# shared-memory limits during multi-process tests.
exec docker run --rm -i \
    --device /dev/kfd --device /dev/dri \
    --group-add video \
    --security-opt seccomp=unconfined \
    --ipc=host \
    -v "$REPO":/work:ro \
    -e VLLM_TEST_CLEAN_GPU_MEMORY=0 \
    -e PYTHONPYCACHEPREFIX=/tmp/pyc \
    --entrypoint bash "$IMAGE" -s -- "$@" <<'CONTAINER'
set -euo pipefail

# vLLM logs warnings to stdout, so take only the last line.
SITE="$(python3 -c 'import vllm, os; print(os.path.dirname(vllm.__file__))' | tail -1)"

# Overlay this checkout's Python sources onto the installed package. Compiled
# extensions in the image are left alone.
#
# Every tracked file is considered, not just modified/untracked ones: once work
# is committed it is still the code under test, and an overlay that skips it
# would silently fall back to whatever the image was built from.
#
# safe.directory is required because the mount is owned by the host user, and
# git otherwise aborts with "detected dubious ownership". That failure must not
# be swallowed -- it previously left the overlay empty with no indication.
if ! tracked=$(git -c safe.directory=/work -C /work ls-files -- 'vllm/*.py' 'vllm/**/*.py'); then
    echo "error: could not list tracked sources in /work" >&2
    exit 1
fi

if [ -z "$tracked" ]; then
    echo "error: no tracked vllm Python sources found in /work" >&2
    exit 1
fi

# Files absent from the image are created, not skipped. Skipping them yields a
# broken hybrid: an overlaid __init__.py importing a sibling module that was
# added after the image was built fails with ModuleNotFoundError.
overlaid=0
added=0
while read -r rel; do
    [ -n "$rel" ] || continue
    [ -f "/work/$rel" ] || continue
    dest="$SITE/${rel#vllm/}"
    if [ -f "$dest" ]; then
        cmp -s "/work/$rel" "$dest" && continue
        overlaid=$((overlaid + 1))
    else
        mkdir -p "$(dirname "$dest")"
        added=$((added + 1))
    fi
    cp "/work/$rel" "$dest"
done <<<"$tracked"

echo "overlay: $overlaid changed, $added added -> $SITE"

# A large count means the image is well behind this checkout. The overlay only
# replaces Python, so compiled extensions still come from the image and may not
# match what these sources expect. Rebuild rather than trust a surprising pass.
if [ "$((overlaid + added))" -gt 100 ]; then
    echo "warning: $((overlaid + added)) Python files differ from the image, so it" >&2
    echo "warning: is substantially older than this checkout. Compiled extensions" >&2
    echo "warning: still come from the image and may not match these sources;" >&2
    echo "warning: rebuild before trusting a surprising result." >&2
fi

python3 -m pytest --version >/dev/null 2>&1 || pip install -q pytest
python3 -c 'import tblib' 2>/dev/null || pip install -q tblib pytest-asyncio

# Resolve test paths against the mount, leaving flags and their values alone.
args=()
for a in "$@"; do
    if [[ "$a" != -* ]] && [ -e "/work/$a" ]; then
        args+=("/work/$a")
    else
        args+=("$a")
    fi
done

# Run from /tmp so pytest does not treat the read-only mount as rootdir.
cd /tmp

# --import-mode=importlib is load-bearing, not a style choice. Under pytest's
# default prepend mode the mount lands on sys.path and `import vllm` resolves to
# /work/vllm, which has no compiled extensions: vllm._C, vllm._C_stable_libtorch
# and vllm._rocm_C all fail to import and the run silently falls back to Triton
# and pure-Python paths, so a test meant to cover a C++/HIP op covers a fallback
# instead. With importlib the installed package is imported, its extensions load,
# and the overlay above is what carries this checkout's Python on top of them.
exec python3 -m pytest --import-mode=importlib -p no:cacheprovider --no-header \
    "${args[@]}"
CONTAINER

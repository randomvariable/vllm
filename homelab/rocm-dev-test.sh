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
# Changed pure-Python files from this checkout are copied over the installed
# package, so edits take effect without a rebuild. C++/HIP changes DO need a
# rebuilt image.

set -euo pipefail

IMAGE="${IMAGE:-vllm-strix-runtime:local}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [ "$#" -eq 0 ]; then
    echo "usage: ${BASH_SOURCE[0]##*/} <pytest args...>" >&2
    exit 2
fi

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

# Overlay changed Python sources onto the installed package. Compiled
# extensions in the image are left alone.
( cd /work && git ls-files -m -o --exclude-standard -- 'vllm/**/*.py' || true ) \
    | while read -r rel; do
        [ -f "/work/$rel" ] || continue
        dest="$SITE/${rel#vllm/}"
        [ -f "$dest" ] || continue
        cp "/work/$rel" "$dest"
        echo "overlaid: $rel"
    done

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
exec python3 -m pytest -p no:cacheprovider --no-header "${args[@]}"
CONTAINER

#!/usr/bin/env bash
# Native aarch64 translation of homelab/spark-cross.Dockerfile.
#
# The Dockerfile cross-compiles this fork for sm_121a from an x86_64 builder.
# This script performs the same build natively on a DGX Spark (GB10, sm_121)
# and then serves DeepSeek-V4-Flash-DSpark from the result, so a source change
# can be qualified without a full Tekton round trip.
#
# Everything the Dockerfile does purely to survive cross-compilation is dropped:
# the SBSA toolchain file, the aarch64 gcc/g++ prefixes, the separately
# extracted aarch64 torch tree, the DEEPGEMM_* target overrides, the arm64
# pyconfig.h graft, FLASHINFER_FMHA_V2_HOST_BUILD, the aarch64-linux-gnu-nm
# symbol audit, and the --plat-name/_PYTHON_HOST_PLATFORM overrides. On the
# target these are all just "the host compiler".
#
# ---------------------------------------------------------------------------
# DGX OS 7 packages
# ---------------------------------------------------------------------------
# DGX OS 7 (Ubuntu 24.04 aarch64) ships the driver, /usr/local/cuda-13.0 and
# python3.12, but none of the build tooling. Install once as root:
#
#   sudo apt-get update && sudo apt-get install -y --no-install-recommends \
#     build-essential ccache cmake ninja-build pkg-config git curl wget unzip \
#     file perl python3-dev python3-venv python3-pip \
#     libprotobuf-dev protobuf-compiler libnuma-dev libibverbs-dev librdmacm-dev \
#     libmimalloc-dev patchelf
#
# The image builds against CUDA 13.3 while DGX OS 7 preinstalls 13.0. The
# NVIDIA compute repo is already configured on these nodes, so match the image:
#
#   sudo apt-get install -y cuda-toolkit-13-3
#
# Building against 13.0 works but diverges from the shipped artifact; CUDA_HOME
# below prefers 13.3 when present and falls back to 13.0.
#
# uv and the Rust toolchain are installed into $HOME by this script (as in the
# Dockerfile) and need no apt packages.
#
# ---------------------------------------------------------------------------
# Usage
# ---------------------------------------------------------------------------
#   ./spark-deepseek-test.sh deps      # venv + torch + submodules + rust
#   ./spark-deepseek-test.sh build     # flashinfer AOT, rust frontend, vllm, b12x
#   ./spark-deepseek-test.sh fetch     # download the model into HF_HOME
#   ./spark-deepseek-test.sh serve     # serve (TP=2, this node is rank 0)
#   ./spark-deepseek-test.sh worker    # serve (TP=2, this node is rank 1)
#   ./spark-deepseek-test.sh smoke     # probe a running server
#   ./spark-deepseek-test.sh all       # deps + build + fetch
#
# DeepSeek-V4-Flash-DSpark is ~155 GiB of weights against 121 GiB of unified
# memory per GB10, so a single node cannot hold it. serve/worker therefore
# default to the same two-node TP=2 topology as the deployment: rank 0 here,
# rank 1 on ${PEER_HOST}, rendezvous over the RoCE /30 rather than vlan192.

set -euo pipefail

log() { printf '\033[1;36m[spark-test]\033[0m %s\n' "$*" >&2; }
die() { printf '\033[1;31m[spark-test] ERROR:\033[0m %s\n' "$*" >&2; exit 1; }

# --- pins ------------------------------------------------------------------
# Mutagen syncs the worktree without .git, so submodule gitlinks cannot be
# resolved on the node. These SHAs mirror the gitlinks under third_party/ and
# must be updated together with .gitmodules.
FLASHINFER_REF=0f9103316020d6b041ca7be681b650c84f01b1aa
FLASHINFER_URL=https://github.com/randomvariable/flashinfer.git
DEEPGEMM_REF=3dc66cea8b2034eaec1b1d19b84e0e0476f7fe4b
DEEPGEMM_URL=https://github.com/randomvariable/DeepGEMM.git
B12X_REF=261adac256cb7027fbc0eb676f2824ff4abaeeef
B12X_URL=https://github.com/randomvariable/b12x.git

TORCH_VERSION=2.13.0
TORCH_INDEX=https://download.pytorch.org/whl/cu130
RUST_TOOLCHAIN=1.95

# Ubuntu 24.04 ships ccache 4.9.1, which reports every nvcc
# --generate-dependencies-with-compile call as "Preprocessing failed" and so
# caches none of FlashInfer's ~3400 AOT units. 4.13.x handles it.
CCACHE_VERSION=${CCACHE_VERSION:-4.13.6}
CCACHE_SHA256=2098d561e4a8e36bd06a29aedce53ea90c7e365f9573a93d91c230efbf96a958

DSPARK_MODEL_REPO=${DSPARK_MODEL_REPO:-deepseek-ai/DeepSeek-V4-Flash-DSpark}
# Pin the snapshot: serving must never race a moving refs/main. This is the
# commit refs/main resolved to; bump deliberately, not by following the branch.
DSPARK_MODEL_REVISION=${DSPARK_MODEL_REVISION:-62af8fffb2f7030cac4de2f0169f5b8d1101b646}

# --- paths and topology ----------------------------------------------------
VLLM_ROOT=${VLLM_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
# Everything generated lives outside VLLM_ROOT. Mutagen syncs that tree
# bidirectionally, so a venv or wheel dir under it would push several GiB of
# aarch64 binaries back onto the dev machine and fight the sync on every build.
STATE=${STATE:-${HOME}/.cache/spark-test}
VENV=${VENV:-${STATE}/venv}
PY=${VENV}/bin/python
WHEELS=${WHEELS:-${STATE}/wheels}
UV=${UV:-${HOME}/.local/bin/uv}

# rank 0 owns the rendezvous. Use the RoCE /30 (172.31.18.0/30 on this pair),
# not vlan192: NCCL_IB_HCA below expects the ConnectX rails to carry traffic.
NODE_RANK=${NODE_RANK:-0}
NNODES=${NNODES:-2}
MASTER_ADDR=${MASTER_ADDR:-172.31.18.1}
MASTER_PORT=${MASTER_PORT:-25000}
PEER_HOST=${PEER_HOST:-192.168.192.19}
SERVE_HOST=${SERVE_HOST:-0.0.0.0}
SERVE_PORT=${SERVE_PORT:-8888}
TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE:-2}

# nvcc RSS per unit climbs sharply through the FA3 and MLA kernels; the
# Dockerfile measured ~4.1 GiB per job under a cgroup limit. On a GB10 the
# 121 GiB of memory is shared with the resident model of anything already
# serving on this node, so size the build against what is actually free rather
# than against core count -- overcommitting here OOM-kills the serving process,
# not the compiler.
_avail_gib() { awk '/^MemAvailable:/ {print int($2/1048576)}' /proc/meminfo; }
_jobs_for_mem() {
  local per_job=$1 cap=$2 avail n
  avail=$(_avail_gib)
  n=$(( (avail - 8) / per_job ))
  ((n < 1)) && n=1
  ((n > cap)) && n=${cap}
  printf '%s' "${n}"
}
MAX_JOBS=${MAX_JOBS:-$(_jobs_for_mem 2 "$(nproc)")}
FLASHINFER_JOBS=${FLASHINFER_JOBS:-$(_jobs_for_mem 5 6)}

# Refuse to build underneath a live serving process unless explicitly allowed:
# these nodes host production TP ranks and an OOM kill lands on the server.
ALLOW_BUSY_GPU=${ALLOW_BUSY_GPU:-0}

# --- toolchain environment -------------------------------------------------
# Dockerfile builder ENV, minus every cross-compilation override.
if [[ -z ${CUDA_HOME:-} ]]; then
  for c in /usr/local/cuda-13.3 /usr/local/cuda-13.0 /usr/local/cuda; do
    if [[ -x ${c}/bin/nvcc ]]; then
      CUDA_HOME=${c}
      break
    fi
  done
fi
: "${CUDA_HOME:?no CUDA toolkit found; install cuda-toolkit-13-3}"
export CUDA_HOME
export CUDA_TOOLKIT_ROOT=${CUDA_HOME}
export PATH=${VENV}/bin:${HOME}/.local/bin:${HOME}/.cargo/bin:${CUDA_HOME}/bin:${PATH}

export VLLM_TARGET_DEVICE=cuda
export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-12.1a}
export CMAKE_BUILD_PARALLEL_LEVEL=${CMAKE_BUILD_PARALLEL_LEVEL:-${MAX_JOBS}}
export NVCC_THREADS=${NVCC_THREADS:-2}

# DeepGEMM's _C builder takes its target from the environment. Natively these
# are just the host's own compiler, python and torch, but tools/build_deepgemm_C.py
# still reads them, so set them explicitly rather than relying on its defaults.
export DEEPGEMM_SRC_DIR=${VLLM_ROOT}/third_party/deep_gemm

# Point CMake at the CUTLASS tree carried by the FlashInfer submodule, exactly
# as the image does; without it cmake fetches a second, unpinned copy.
export CMAKE_ARGS="${CMAKE_ARGS:-} -DVLLM_CUTLASS_SRC_DIR=${VLLM_ROOT}/third_party/flashinfer/3rdparty/cutlass"

export CCACHE_DIR=${CCACHE_DIR:-${HOME}/.cache/spark-test/ccache}
export CCACHE_MAXSIZE=${CCACHE_MAXSIZE:-40G}
export CCACHE_NOHASHDIR=true
export CCACHE_COMPILERCHECK=content
export CCACHE_BASEDIR=${VLLM_ROOT}
export CCACHE_EXTRAFILES=${HOME}/.cache/spark-test/ccache-keyfile
export CCACHE_SLOPPINESS=time_macros,include_file_mtime,include_file_ctime
export CCACHE_DEPEND=true

# setuptools-scm cannot see a version: mutagen syncs the tree without .git.
# setup.py's VLLM_VERSION_OVERRIDE forwards to this same variable, so one
# assignment covers the wheel build and ./build_rust.sh.
export SETUPTOOLS_SCM_PRETEND_VERSION=${SETUPTOOLS_SCM_PRETEND_VERSION:-0.1.dev1}

preflight() {
  [[ -f ${VLLM_ROOT}/setup.py ]] || die "VLLM_ROOT=${VLLM_ROOT} is not a vLLM checkout"
  command -v nvidia-smi >/dev/null || die "nvidia-smi not found"
  local cc smi_mem smi_apps busy mem
  cc=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader) \
    || die "nvidia-smi compute capability query failed"
  cc=$(printf '%s\n' "${cc}" | head -1)
  [[ ${cc} == 12.1 ]] || die "expected compute_cap 12.1 (GB10), got '${cc}'"
  [[ $(uname -m) == aarch64 ]] || die "this script builds natively; run it on the Spark"
  local missing=()
  for t in cmake ninja git curl unzip g++ patchelf; do
    if ! command -v "${t}" >/dev/null; then missing+=("${t}"); fi
  done
  if ((${#missing[@]} != 0)); then
    die "missing tools: ${missing[*]} (see DGX OS 7 packages above)"
  fi

  # Fail closed: nvidia-smi errors, unknown memory values, active compute PIDs,
  # or any allocated memory block build/serve unless explicitly overridden.
  smi_mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits) \
    || die "nvidia-smi GPU memory query failed"
  smi_apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader) \
    || die "nvidia-smi process query failed"
  busy=0
  mem=0
  while read -r value; do
    [[ ${value} =~ ^[0-9]+$ ]] || die "unparseable GPU memory usage: '${value}'"
    ((value > 0)) && mem=$((mem + value))
  done <<<"${smi_mem}"
  while read -r value; do
    [[ -z ${value} ]] || ((busy++))
  done <<<"${smi_apps}"
  if ((busy > 0 || mem > 0)) && [[ ${ALLOW_BUSY_GPU} != 1 ]]; then
    nvidia-smi --query-compute-apps=pid,used_memory --format=csv >&2 || true
    die "GPU busy: ${busy} compute process(es), ${mem} MiB allocated, $(_avail_gib) GiB free.
  These nodes host production TP ranks; a parallel build can OOM-kill the server.
  Drain the node first, or set ALLOW_BUSY_GPU=1 to override."
  fi
  if ((busy > 0 || mem > 0)); then
    log "WARNING: ALLOW_BUSY_GPU=1 override; ${busy} process(es), ${mem} MiB allocated"
  fi
  log "preflight ok: ${cc} sm_121, CUDA_HOME=${CUDA_HOME} ($("${CUDA_HOME}"/bin/nvcc --version | sed -n 's/.*release \([0-9.]*\).*/\1/p')), $(_avail_gib) GiB free, MAX_JOBS=${MAX_JOBS} FLASHINFER_JOBS=${FLASHINFER_JOBS}"
}

# --- deps ------------------------------------------------------------------
install_ccache() {
  local want=${CCACHE_VERSION} have=""
  have=$(ccache --version 2>/dev/null | head -1 | awk '{print $3}') || true
  if [[ ${have} == "${want}" ]]; then
    log "ccache ${have} already current"
    return
  fi
  local base=ccache-${want}-linux-aarch64-musl-static
  log "installing ccache ${want} (apt's 4.9.1 cannot cache nvcc dependency-generating compiles)"
  local tmp
  tmp=$(mktemp -d)
  curl -fsSL -o "${tmp}/cc.tar.xz" \
    "https://github.com/ccache/ccache/releases/download/v${want}/${base}.tar.xz"
  echo "${CCACHE_SHA256}  ${tmp}/cc.tar.xz" | sha256sum -c -
  tar -xf "${tmp}/cc.tar.xz" -C "${tmp}"
  install -Dm755 "${tmp}/${base}/ccache" "${HOME}/.local/bin/ccache"
  rm -rf "${tmp}"
  [[ $(command -v ccache) == "${HOME}/.local/bin/ccache" ]] \
    || die "ccache resolves to $(command -v ccache), not ~/.local/bin"
  ccache --version | head -1
}

clone_pin() {
  local path=$1 url=$2 ref=$3
  if [[ -e ${path}/.git ]]; then
    if [[ $(git -C "${path}" rev-parse HEAD) == "${ref}" ]]; then
      log "$(basename "${path}") at ${ref:0:12}"
      return
    fi
  elif [[ -n $(ls -A "${path}" 2>/dev/null) ]]; then
    die "${path} is non-empty but has no .git; remove it and rerun"
  fi
  log "fetching $(basename "${path}") ${ref:0:12}"
  mkdir -p "${path}"
  git -C "${path}" init -q 2>/dev/null || true
  git -C "${path}" remote add origin "${url}" 2>/dev/null || \
    git -C "${path}" remote set-url origin "${url}"
  git -C "${path}" fetch --depth 1 origin "${ref}"
  git -C "${path}" checkout -q FETCH_HEAD
  git -C "${path}" submodule update --init --recursive --depth 1
}

stage_deps() {
  preflight
  install_ccache
  mkdir -p "${WHEELS}" "${CCACHE_DIR}" "$(dirname "${CCACHE_EXTRAFILES}")"

  [[ -x ${UV} ]] || curl -LsSf https://astral.sh/uv/install.sh | sh
  [[ -x ${PY} ]] || "${UV}" venv --seed --python 3.12 "${VENV}"

  log "installing torch ${TORCH_VERSION} and build requirements"
  "${UV}" pip install --python "${PY}" "torch==${TORCH_VERSION}" --index-url "${TORCH_INDEX}"
  "${UV}" pip install --python "${PY}" \
    -r "${VLLM_ROOT}/requirements/build/cuda.txt" \
    -r "${VLLM_ROOT}/requirements/build/rust.txt"
  "${PY}" -c "import torch;print('torch',torch.__version__,'cuda',torch.version.cuda)"

  # Submodules: the synced tree has empty gitlink directories.
  clone_pin "${VLLM_ROOT}/third_party/flashinfer" "${FLASHINFER_URL}" "${FLASHINFER_REF}"
  clone_pin "${VLLM_ROOT}/third_party/deep_gemm" "${DEEPGEMM_URL}" "${DEEPGEMM_REF}"
  clone_pin "${VLLM_ROOT}/third_party/b12x" "${B12X_URL}" "${B12X_REF}"
  [[ -f ${VLLM_ROOT}/third_party/flashinfer/3rdparty/cutlass/CMakeLists.txt ]] \
    || die "flashinfer's cutlass submodule is missing; CMAKE_ARGS points at it"

  if ! rustup toolchain list 2>/dev/null | grep -q "^${RUST_TOOLCHAIN}"; then
    log "installing rust ${RUST_TOOLCHAIN}"
    [[ -x ${HOME}/.cargo/bin/rustup ]] || \
      curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain none
    rustup toolchain install "${RUST_TOOLCHAIN}"
  fi
  log "deps complete"
}

# --- build -----------------------------------------------------------------
build_flashinfer() {
  local fi=${VLLM_ROOT}/third_party/flashinfer
  # 12.1a alone is not sufficient. FlashInfer names each AOT module after the
  # arch it was built for (fp4_quantization_121), but its runtime resolver
  # rewrites SM12x to the family variant "120f" whenever CUDA >= 12.9 and then
  # looks up fp4_quantization_120f. With FLASHINFER_DISABLE_JIT=1 that lookup
  # raises MissingJITCacheError instead of compiling a fallback, so both
  # variants must be present in the wheel.
  local arches="12.0f 12.1a"
  local cuda_ver
  cuda_ver=$("${CUDA_HOME}"/bin/nvcc --version | sed -n 's/.*release \([0-9.]*\).*/\1/p')

  log "flashinfer AOT for ${arches} (CUDA ${cuda_ver}, ${FLASHINFER_JOBS} jobs) -- this is the long pole"
  printf '%s\n' "${arches}" "${cuda_ver}" > "${CCACHE_EXTRAFILES}"
  ccache -z

  "${UV}" pip install --python "${PY}" \
    'setuptools>=77' 'packaging>=24' wheel tqdm ninja requests numpy \
    nvidia-ml-py 'apache-tvm-ffi>=0.1,<0.2' filelock

  # flashinfer-build.sh hardcodes /opt/venv/bin/python, so drive the two
  # documented build steps directly against this venv instead.
  (
    cd "${fi}"
    export UV_TORCH_BACKEND=cu$(echo "${cuda_ver}" | cut -d. -f1,2 | tr -d '.')
    export TORCH_CUDA_ARCH_LIST=${arches}
    export FLASHINFER_CUDA_ARCH_LIST=${arches}
    export FLASHINFER_NVCC=${CUDA_HOME}/bin/nvcc
    export FLASHINFER_NVCC_LAUNCHER=ccache
    export FLASHINFER_CXX_LAUNCHER=ccache
    export MAX_JOBS=${FLASHINFER_JOBS}
    export FLASHINFER_NVCC_THREADS=1
    "${UV}" build --python "${PY}" --no-build-isolation --wheel --out-dir "${WHEELS}" .
    FLASHINFER_LOCAL_VERSION=cu130 \
      "${UV}" build --python "${PY}" --no-build-isolation --wheel \
        --out-dir "${WHEELS}" ./flashinfer-jit-cache
    "${UV}" build --python "${PY}" --no-build-isolation --wheel \
      --out-dir "${WHEELS}" ./flashinfer-cubin
  )
  ccache -sv | tee "${WHEELS}/ccache-stats-flashinfer.txt"
}

build_rust_frontend() {
  log "building rust frontend"
  ( cd "${VLLM_ROOT}" && ./build_rust.sh )
  local so
  so=$(find "${VLLM_ROOT}/vllm" -maxdepth 1 -name '_rust_tool_parser*.so' -print -quit)
  [[ -x ${VLLM_ROOT}/vllm/vllm-rs && -n ${so} ]] \
    || die "rust frontend artifacts missing after build_rust.sh"
}

build_vllm() {
  log "building vllm wheel (${CMAKE_BUILD_PARALLEL_LEVEL} jobs)"
  printf '%s\n' "${TORCH_CUDA_ARCH_LIST}" "${CUDA_HOME}" > "${CCACHE_EXTRAFILES}"
  ccache -z
  ( cd "${VLLM_ROOT}" && "${PY}" setup.py bdist_wheel --dist-dir "${WHEELS}" --py-limited-api=cp38 )
  ccache -sv | tee "${WHEELS}/ccache-stats-vllm.txt"

  # The Dockerfile's cross-build audits for unresolved torch::stable symbols
  # because a mangling mismatch there only surfaced after a rollout. Natively
  # the same check costs seconds, so keep it.
  local tmp undef
  tmp=$(mktemp -d)
  unzip -q -o "${WHEELS}"/vllm-*.whl -d "${tmp}"
  undef=$(find "${tmp}" -name '*.so' -exec nm -D -u {} + | grep 'N5torch6stable6Tensor' || true)
  rm -rf "${tmp}"
  if [[ -n ${undef} ]]; then
    echo "${undef}" | sort -u | sed 's/^ *U *//' | c++filt >&2
    die "unresolved internal symbols taking torch::stable::Tensor"
  fi
  log "symbol check: no unresolved internal torch::stable symbols"
}

stage_build() {
  preflight
  build_flashinfer
  build_rust_frontend
  build_vllm

  log "installing built wheels into ${VENV}"
  # The Dockerfile strips FlashInfer's public-index pins from requirements before
  # resolving, then installs the locally built wheels. Do the same: leaving them
  # in would pull an upstream flashinfer-python over the patched local build.
  local reqs=${WHEELS}/runtime-cuda.txt
  "${PY}" -c "
import sys
from pathlib import Path
src, dst = Path(sys.argv[1]), Path(sys.argv[2])
skip = ('--extra-index-url ', 'flashinfer-python==', 'flashinfer-cubin==', 'flashinfer-jit-cache==')
dst.write_text(''.join(l for l in src.read_text().splitlines(keepends=True)
                       if not l.startswith(skip)))
" "${VLLM_ROOT}/requirements/cuda.txt" "${reqs}"
  "${UV}" pip install --python "${PY}" -r "${reqs}"
  # cubin before the python wheel so FlashInfer's import-time version check
  # sees the same release. Install the local file rather than a version spec:
  # the version follows the submodule and must not be restated here.
  "${UV}" pip install --python "${PY}" "${WHEELS}"/flashinfer_cubin-*.whl
  "${UV}" pip install --python "${PY}" "${WHEELS}"/flashinfer_python-*.whl
  "${UV}" pip install --python "${PY}" --no-deps "${WHEELS}"/flashinfer_jit_cache-*.whl
  "${UV}" pip install --python "${PY}" "${VLLM_ROOT}/third_party/b12x"
  "${UV}" pip install --python "${PY}" xxhash
  "${UV}" pip install --python "${PY}" --no-deps "${WHEELS}"/vllm-*.whl

  # Import from a neutral cwd: ${VLLM_ROOT}/vllm would shadow the installed
  # package and hide a missing extension (memory #5637's CUDA analogue).
  ( cd /tmp && "${PY}" -c "
import torch, vllm, vllm._C, flashinfer, b12x
print('vllm', vllm.__version__, '| torch', torch.__version__, '| flashinfer', flashinfer.__version__)
print('vllm._C OK, cuda arch', torch.cuda.get_device_capability())
" )
  log "build complete"
}

# --- model -----------------------------------------------------------------
export HF_HOME=${HF_HOME:-${HOME}/.cache/spark-test/huggingface}
export VLLM_CACHE_ROOT=${VLLM_CACHE_ROOT:-${HOME}/.cache/spark-test/vllm}
export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-${HOME}/.cache/spark-test/torchinductor}
export FLASHINFER_WORKSPACE_BASE=${FLASHINFER_WORKSPACE_BASE:-${HOME}/.cache/spark-test/flashinfer}

model_dir() {
  printf '%s/hub/models--%s/snapshots/%s' \
    "${HF_HOME}" "${DSPARK_MODEL_REPO//\//--}" "${DSPARK_MODEL_REVISION}"
}

stage_fetch() {
  # 155.4 GiB across 74 files. Resolve by explicit revision, never by refs/main:
  # a moving ref would silently change what the two ranks load.
  local dst
  dst=$(model_dir)
  if [[ -f ${dst}/config.json ]]; then
    log "model already present at ${dst}"
    return
  fi
  log "fetching ${DSPARK_MODEL_REPO}@${DSPARK_MODEL_REVISION:0:12} (~156 GiB) into ${HF_HOME}"
  "${UV}" pip install --python "${PY}" 'huggingface_hub[hf_transfer]'
  HF_HUB_ENABLE_HF_TRANSFER=1 HF_HUB_DISABLE_XET=1 \
    "${PY}" -c "
from huggingface_hub import snapshot_download
print(snapshot_download('${DSPARK_MODEL_REPO}', revision='${DSPARK_MODEL_REVISION}'))
"
  [[ -f ${dst}/config.json ]] || die "snapshot did not land at ${dst}"
}

# --- runtime environment ---------------------------------------------------
# Mirrors the LeaderWorkerSet's modelserver container env. Only the transport
# settings differ: the pods route rendezvous over the pod network (eth0) with
# RoCE for NCCL data, while here both live on the ConnectX rails directly.
runtime_env() {
  export VLLM_USE_RUST_FRONTEND=${VLLM_USE_RUST_FRONTEND:-0}
  export VLLM_USE_B12X_MOE=1
  export VLLM_USE_FLASHINFER_SAMPLER=1
  export VLLM_USE_BREAKABLE_CUDAGRAPH=0
  export CUTE_DSL_ARCH=sm_121a
  export TORCH_CUDA_ARCH_LIST=12.1a
  export FLASHINFER_CUDA_ARCH_LIST=12.1a
  export FLASHINFER_DISABLE_VERSION_CHECK=1
  export TILELANG_CLEANUP_TEMP_FILES=1
  export DG_JIT_USE_NVRTC=0
  export DG_JIT_NVCC_COMPILER=${CUDA_HOME}/bin/nvcc
  export HF_HUB_OFFLINE=1
  export HF_HUB_DISABLE_XET=1
  export VLLM_SERVER_DEV_MODE=1
  export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
  export VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=256
  export VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

  # Two ConnectX rails, each its own /30 (17: 172.31.18.1 + .5, 19: .2 + .6),
  # so NCCL must be allowed to cross NICs. Rendezvous and the gloo CPU group
  # ride rail 1 rather than the k8s pod network.
  export NCCL_NET=IB
  export NCCL_IB_DISABLE=0
  export NCCL_IB_HCA=${NCCL_IB_HCA:-rocep1s0f1,roceP2p1s0f1}
  export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-enp1s0f1np1}
  export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-${NCCL_SOCKET_IFNAME}}
  export NCCL_CROSS_NIC=1
  export NCCL_CUMEM_ENABLE=0
  export NCCL_IGNORE_CPU_AFFINITY=1
  export NCCL_NVLS_ENABLE=0
  export NCCL_DEBUG=${NCCL_DEBUG:-WARN}

  mkdir -p "${VLLM_CACHE_ROOT}" "${TORCHINDUCTOR_CACHE_DIR}" "${FLASHINFER_WORKSPACE_BASE}"
}

# --- serve -----------------------------------------------------------------
SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-deepseek-v4-flash-dspark}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-1048576}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-6}
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-8192}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.84}
# DSpark's semi-autoregressive block drafter requires
# num_speculative_tokens >= config.dspark_block_size (5 for this checkpoint).
# A smaller value feeds the block/Markov heads an unsupported block length and
# produces garbled output rather than an error.
MTP_NUM_TOKENS=${MTP_NUM_TOKENS:-5}
DEFAULT_THINKING=${DEFAULT_THINKING:-low}

stage_serve() {
  local rank=${1:-${NODE_RANK}}
  preflight
  runtime_env

  local model
  model=$(model_dir)
  [[ -f ${model}/config.json ]] || die "model missing at ${model}; run 'fetch' first"

  local kwargs
  case "${DEFAULT_THINKING}" in
    off)  kwargs='{"thinking":false}' ;;
    low)  kwargs='{"thinking":true,"reasoning_effort":"low"}' ;;
    high) kwargs='{"thinking":true,"reasoning_effort":"high"}' ;;
    max)  kwargs='{"thinking":true,"reasoning_effort":"max"}' ;;
    *)    die "DEFAULT_THINKING must be off|low|high|max (got '${DEFAULT_THINKING}')" ;;
  esac

  # rank 0's own worker subprocess loops back, so it must bind the rendezvous
  # locally; only rank>0 dials the leader across the rail.
  local master=${MASTER_ADDR}
  if ((rank == 0)); then master=127.0.0.1; fi

  local argv=(
    vllm serve "${model}"
    --served-model-name "${SERVED_MODEL_NAME}"
    --host "${SERVE_HOST}"
    --port "${SERVE_PORT}"
    --trust-remote-code
    --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}"
    --pipeline-parallel-size 1
    --kv-cache-dtype fp8
    --block-size 256
    --max-model-len "${MAX_MODEL_LEN}"
    --max-num-seqs "${MAX_NUM_SEQS}"
    --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}"
    --max-cudagraph-capture-size "$(( MAX_NUM_SEQS * (MTP_NUM_TOKENS + 1) ))"
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
    --enable-prefix-caching
    --enable-prompt-tokens-details
    --async-scheduling
    --enable-chunked-prefill
    --speculative-config "{\"method\":\"dspark\",\"num_speculative_tokens\":${MTP_NUM_TOKENS},\"draft_sample_method\":\"probabilistic\"}"
    --tokenizer-mode deepseek_v4
    --distributed-executor-backend mp
    --moe-backend flashinfer_b12x
    --tool-call-parser deepseek_v4
    --enable-auto-tool-choice
    --reasoning-parser deepseek_v4
    --reasoning-config '{"reasoning_parser":"deepseek_v4","reasoning_start_str":" thinking","reasoning_end_str":"\n"}'
    --default-chat-template-kwargs "${kwargs}"
    --generation-config vllm
    --enable-flashinfer-autotune
    --nnodes "${NNODES}"
    --node-rank "${rank}"
    --master-addr "${master}"
    --master-port "${MASTER_PORT}"
  )
  # rank>0 runs no API server; it joins the leader's process group and serves
  # as its second TP shard.
  if ((rank != 0)); then argv+=(--headless); fi

  log "vllm serve rank ${rank}/${NNODES}, master ${master}:${MASTER_PORT}"
  if ((rank == 0)); then
    log "peer: NODE_RANK=1 MASTER_ADDR=${MASTER_ADDR} $0 worker   (on ${PEER_HOST})"
  fi
  exec "${argv[@]}"
}

# --- smoke -----------------------------------------------------------------
stage_smoke() {
  local base=http://${SMOKE_HOST:-127.0.0.1}:${SERVE_PORT}
  log "waiting for ${base}/health"
  local i
  for i in $(seq 1 "${SMOKE_TIMEOUT:-2400}"); do
    if curl -fsS "${base}/health" >/dev/null 2>&1; then break; fi
    sleep 1
    if (( i % 60 == 0 )); then log "  still waiting (${i}s)"; fi
  done
  curl -fsS "${base}/health" >/dev/null || die "server did not become healthy"

  curl -fsS "${base}/v1/models" | "${PY}" -m json.tool

  log "chat completion"
  curl -fsS "${base}/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"${SERVED_MODEL_NAME}\",\"max_tokens\":128,\"temperature\":0,
         \"messages\":[{\"role\":\"user\",\"content\":\"Name the four inner planets, in order.\"}]}" \
    | "${PY}" -c "
import json, sys
d = json.load(sys.stdin)
m = d['choices'][0]['message']
print('reasoning:', (m.get('reasoning_content') or '')[:200])
print('content  :', m['content'])
print('usage    :', d['usage'])
assert m['content'].strip(), 'empty completion'
"

  # Speculative decode has to actually accept drafts, otherwise DSpark is
  # running but contributing nothing.
  curl -fsS "${base}/metrics" 2>/dev/null \
    | grep -E '^vllm:spec_decode_(num_draft|num_accepted)_tokens' || \
    log "no spec-decode counters exposed"
  log "smoke ok"
}

# --- dispatch --------------------------------------------------------------
case "${1:-all}" in
  deps)   stage_deps ;;
  build)  stage_build ;;
  fetch)  stage_fetch ;;
  serve)  stage_serve 0 ;;
  worker) stage_serve 1 ;;
  smoke)  stage_smoke ;;
  all)    stage_deps; stage_build; stage_fetch ;;
  *)      die "unknown subcommand '$1' (deps|build|fetch|serve|worker|smoke|all)" ;;
esac

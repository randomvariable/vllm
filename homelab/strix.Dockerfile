# syntax=docker/dockerfile:1.7
#
# vllm-strix-runtime -- vLLM runtime for AMD Strix Halo (gfx1151 / Ryzen AI Max /
# Radeon 8060S) APUs, built from the same branch as the CUDA DGX Spark runtime.
# Sibling of vllm-spark-runtime; same source-build and two-stage builder->runtime
# contract, retargeted to ROCm.
#
# WHY TWO STAGES (load-bearing, do not collapse):
#   * The BUILDER needs the TheRock *nightly* ROCm SDK (rocm-sdk-devel) to get
#     hipcc / amdclang / HIP CMake config -- the base image ships a runtime-only
#     ROCm that cannot compile HIP extensions.
#   * The nightly HSA runtime (libhsa-runtime64) SEGFAULTS in
#     rocr::AMD::GpuAgent::InitDma() on gfx1151 at torch GPU init. So the RUNTIME
#     stage must NOT carry the nightly SDK: it uses the base image's *release*
#     ROCm 7.14 torch, whose HSA runtime initialises the 8060S cleanly. The wheel
#     built against the nightly SDK imports fine against the release runtime
#     (same 7.14 major; the crash was the nightly HSA lib, not an ABI mismatch).
#
# The base image is TheRock's rocm7.14 gfx1151 PyTorch image (torch 2.12.0 +
# rocm7.14, arch list includes gfx1151, GPU enumerates as "AMD Radeon 8060S").

ARG BASE_IMAGE=rocm/pytorch:rocm7.14_ubuntu24.04_py3.12_pytorch_release_2.12.0

# Coherent TheRock nightly used for the BUILD toolchain (hipcc/cmake) and the
# build-time torch. Runtime keeps the base image's release torch.
ARG ROCM_NIGHTLY=7.14.0a20260612
ARG ROCM_INDEX=https://rocm.nightlies.amd.com/v2-staging/gfx1151/
ARG BUILD_TORCH=2.12.0+rocm7.14.0a20260612
ARG PYTORCH_ROCM_ARCH=gfx1151

# GGUF quantization plugin (out-of-tree; ROCm 7 supported). "super important for
# Strix Halo." Its _C_gguf HIP kernel is compiled in the builder (where hipcc
# lives) and the resulting wheel is installed into the runtime.
ARG GGUF_PLUGIN_REPOSITORY=https://github.com/vllm-project/vllm-gguf-plugin.git
# Pinned SHA (main tip with ROCm 7 support, #85). Unpinned `main` on an
# out-of-tree repo is non-reproducible; bump deliberately.
ARG GGUF_PLUGIN_REF=1df60c43f1f1274681bb957e5bb9b8f5c44d2f4d

# ---------------------------------------------------------------------------
FROM ${BASE_IMAGE} AS builder
ARG ROCM_NIGHTLY
ARG ROCM_INDEX
ARG BUILD_TORCH
ARG PYTORCH_ROCM_ARCH
ARG GGUF_PLUGIN_REPOSITORY
ARG GGUF_PLUGIN_REF

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    VLLM_TARGET_DEVICE=rocm \
    MAX_JOBS=8 \
    CMAKE_BUILD_PARALLEL_LEVEL=8 \
    VERBOSE=1

SHELL ["/bin/bash", "-euo", "pipefail", "-c"]

# --- coherent nightly ROCm SDK (hipcc / amdclang / HIP CMake config) ----------
# The coherence key is the DEVICE-SUFFIXED rocm-sdk-libraries-gfx1151, pulled by
# the rocm[libraries,devel] meta extra at one dated nightly. The base image's
# generic release libraries (.so.1) mismatch the devel nightly cmake targets
# (.so.1.1); the meta extra installs the matching gfx1151 libraries. rocm-sdk
# init links the device libs into the devel tree and materialises the toolchain.
RUN --mount=type=cache,target=/root/.cache/pip \
    /opt/venv/bin/pip install --index-url "${ROCM_INDEX}" \
        "rocm[libraries,devel]==${ROCM_NIGHTLY}" \
 && /opt/venv/bin/rocm-sdk init \
 && ROOT="$(/opt/venv/bin/rocm-sdk path --root)" \
 && test -e "${ROOT}/lib/librocrand.so.1.1" \
 && test -x "${ROOT}/lib/llvm/bin/amdclang++"

# Build-time torch MUST match the SDK nightly date so the wheel's HIP extensions
# link torch's libamdhip64 coherently (kyuz0 "torch first" ordering).
RUN --mount=type=cache,target=/root/.cache/pip \
    /opt/venv/bin/pip install --pre --index-url "${ROCM_INDEX}" \
        "torch==${BUILD_TORCH}" \
 && /opt/venv/bin/pip install \
        "setuptools>=77.0.3,<80.0.0" wheel "setuptools-scm>=8" \
        "setuptools-rust>=1.9.0" "cmake>=3.26.1,<4" pybind11

# --- acquire the source: this Dockerfile lives in the vllm fork (homelab/), ---
# so the build context IS the source -- a straight copy, no clone. The builder
# (an external CI harness) owns which commit is checked out.
# .git is excluded by .dockerignore (common CI pattern) so setuptools-scm
# cannot derive the version. The external CI harness must supply it.
ARG VLLM_SCM_VERSION=0.1.dev1
ENV SETUPTOOLS_SCM_PRETEND_VERSION=${VLLM_SCM_VERSION}
COPY . /src/vllm

# --- Rust vllm-rs frontend ----------------------------------------------------
# Must run before the wheel build: setup.py's optional Rust extensions otherwise
# allow a wheel with no vllm-rs or _rust_tool_parser to build silently.
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    --mount=type=cache,target=/root/.cache/pip \
    --mount=type=cache,target=/root/.rustup,sharing=locked \
    --mount=type=cache,target=/root/.cargo/registry,sharing=locked \
    --mount=type=cache,target=/root/.cargo/git,sharing=locked \
    --mount=type=cache,target=/src/vllm/target,sharing=locked \
    rm -f /etc/apt/apt.conf.d/docker-clean && \
    apt-get update && apt-get install -y --no-install-recommends \
        curl perl make build-essential protobuf-compiler libprotobuf-dev && \
    /opt/venv/bin/pip install "setuptools-rust>=1.9.0" && \
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | \
      sh -s -- -y --default-toolchain none && \
    . "$HOME/.cargo/env" && \
    cd /src/vllm && \
    ./build_rust.sh && \
    test -x /src/vllm/vllm/vllm-rs && \
    ls /src/vllm/vllm/_rust_tool_parser*.so >/dev/null 2>&1

# --- build the gfx1151 wheel --------------------------------------------------
# amdclang as HOST compiler (kyuz0 segfault fix: aligns vLLM-ext ABI with torch;
# also dodges gcc's <mwaitxintrin.h> spinloop failure). ROCM_PATH/CMAKE args come
# from the materialised devel root.
RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=cache,target=/root/.ccache \
    cd /src/vllm; \
    ROOT="$(/opt/venv/bin/rocm-sdk path --root)"; \
    export ROCM_PATH="$ROOT" HIP_PATH="$ROOT" ROCM_HOME="$ROOT"; \
    export CMAKE_PREFIX_PATH="$ROOT/lib/cmake"; \
    export PATH="$ROOT/bin:$ROOT/lib/llvm/bin:$PATH"; \
    export CC="$ROOT/lib/llvm/bin/amdclang" CXX="$ROOT/lib/llvm/bin/amdclang++"; \
    export PYTORCH_ROCM_ARCH="${PYTORCH_ROCM_ARCH}" \
           HIP_ARCHITECTURES="${PYTORCH_ROCM_ARCH}" \
           AMDGPU_TARGETS="${PYTORCH_ROCM_ARCH}"; \
    BC="$(find "$ROOT" -type d -name bitcode -print -quit 2>/dev/null || true)"; \
    [ -n "$BC" ] && export HIP_DEVICE_LIB_PATH="$BC"; \
    export CMAKE_ARGS="-DROCM_PATH=$ROOT -DHIP_PATH=$ROOT -DAMDGPU_TARGETS=${PYTORCH_ROCM_ARCH} -DHIP_ARCHITECTURES=${PYTORCH_ROCM_ARCH}"; \
    /opt/venv/bin/pip wheel -v --no-build-isolation --no-deps --wheel-dir /wheels .; \
    /opt/venv/bin/python -c "import glob, sys, zipfile; whl = glob.glob('/wheels/vllm-*.whl')[0]; names = zipfile.ZipFile(whl).namelist(); missing = [artifact for artifact, present in [('vllm-rs', any(n.endswith('vllm/vllm-rs') for n in names)), ('_rust_tool_parser', any(n.startswith('vllm/_rust_tool_parser') and n.endswith('.so') for n in names))] if not present]; sys.exit(0) if not missing else (print(f'FATAL: {\", \".join(missing)} missing from wheel -- rust build did not bundle', file=sys.stderr), sys.exit(1))"

# --- build the GGUF plugin wheel (compiles _C_gguf HIP kernel for gfx1151) -----
# Built here where hipcc lives; installed into the release runtime later. Kept
# non-fatal so a plugin-side breakage never blocks the core vLLM image.
RUN --mount=type=cache,target=/root/.cache/pip \
    set -eux; \
    ROOT="$(/opt/venv/bin/rocm-sdk path --root)"; \
    export ROCM_PATH="$ROOT" HIP_PATH="$ROOT" ROCM_HOME="$ROOT"; \
    export CMAKE_PREFIX_PATH="$ROOT/lib/cmake"; \
    export PATH="$ROOT/bin:$ROOT/lib/llvm/bin:$PATH"; \
    export CC="$ROOT/lib/llvm/bin/amdclang" CXX="$ROOT/lib/llvm/bin/amdclang++"; \
    export PYTORCH_ROCM_ARCH="${PYTORCH_ROCM_ARCH}"; \
    git init -q /src/gguf-plugin; \
    git -C /src/gguf-plugin remote add origin "${GGUF_PLUGIN_REPOSITORY}"; \
    git -C /src/gguf-plugin fetch --depth 1 origin "${GGUF_PLUGIN_REF}"; \
    git -C /src/gguf-plugin checkout -q FETCH_HEAD; \
    /opt/venv/bin/pip wheel --no-build-isolation --no-deps \
        --wheel-dir /wheels-gguf /src/gguf-plugin \
      || { echo "WARN: GGUF plugin wheel build failed; core image continues"; \
           mkdir -p /wheels-gguf; }

# Snapshot the fork's ROCm runtime requirement files for the runtime stage.
RUN mkdir -p /runtime-requirements \
 && cp /src/vllm/requirements/rocm.txt   /runtime-requirements/rocm.txt \
 && cp /src/vllm/requirements/common.txt /runtime-requirements/common.txt

# ---------------------------------------------------------------------------
# devtools -- incremental C++/HIP kernel development image for gfx1151.
#
# WHY THIS STAGE EXISTS: `builder` throws its cmake tree away on every run, so a
# one-line kernel edit costs a full image rebuild (~40 min). This stage keeps the
# same proven SDK recipe but installs vLLM EDITABLE and puts the cmake tree and
# ccache on volume-backed paths, so ninja and ccache see previous state and a
# single-TU edit rebuilds in seconds. Not a serving image: no ENTRYPOINT vllm.
#
# WHY THE TOOLCHAIN LIVES IN ITS OWN VENV (/opt/rocm-toolchain): the nightly HSA
# runtime segfaults in rocr::AMD::GpuAgent::InitDma() on gfx1151 at torch GPU
# init, and this stage must both COMPILE and RUN on the box. Installing the
# nightly SDK into a separate venv gives amdclang/hipcc + HIP CMake config for
# compiling while /opt/venv keeps the base image's *release* torch and its
# InitDma-safe HSA runtime for running. Same split the builder->runtime pair
# achieves across stages, collapsed into one image via two venvs.
#
# VOLUME CONTRACT (paths chosen so they can be backed by named volumes; without
# them every run is a cold build):
#   /vllm-build  -> cmake build tree (build.ninja + objects) and FetchContent deps
#   /ccache      -> ccache store, picked up by setup.py's CMAKE_*_COMPILER_LAUNCHER
#   /src/vllm    -> bind mount of the host checkout (rw; compiled .so land here)
FROM ${BASE_IMAGE} AS devtools
ARG ROCM_NIGHTLY
ARG ROCM_INDEX
ARG PYTORCH_ROCM_ARCH

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    VLLM_TARGET_DEVICE=rocm \
    PYTORCH_ROCM_ARCH=${PYTORCH_ROCM_ARCH} \
    HIP_ARCHITECTURES=${PYTORCH_ROCM_ARCH} \
    AMDGPU_TARGETS=${PYTORCH_ROCM_ARCH} \
    VLLM_DEV_BUILD_ROOT=/vllm-build \
    CCACHE_DIR=/ccache \
    PYTHONPYCACHEPREFIX=/vllm-build/pycache \
    HIP_VISIBLE_DEVICES=0 \
    VLLM_DISABLE_COMPILE_CACHE=1

# FetchContent must NOT default to <checkout>/.deps: a tree configured on the
# host (or in another container) caches absolute paths and CMake then aborts with
# "current CMakeCache.txt directory ... is different". Keeping it on the build
# volume also shares dependency downloads across runs.
ENV FETCHCONTENT_BASE_DIR=/vllm-build/deps

SHELL ["/bin/bash", "-euo", "pipefail", "-c"]

# ccache is the difference between a re-edit costing seconds and costing minutes;
# setup.py wires it in automatically once it is on PATH (CMAKE_HIP_COMPILER_LAUNCHER).
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    rm -f /etc/apt/apt.conf.d/docker-clean \
 && apt-get update \
 && apt-get install -y --no-install-recommends ccache git

# Coherent nightly ROCm SDK in its own venv. rocm-sdk-devel is lazily
# materialised, so `rocm-sdk init` MUST run in this same layer or ROCM_PATH
# points at a directory that does not exist yet and LoadHIP.cmake fails.
RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m venv /opt/rocm-toolchain \
 && /opt/rocm-toolchain/bin/pip install --index-url "${ROCM_INDEX}" \
        "rocm[libraries,devel]==${ROCM_NIGHTLY}" \
 && /opt/rocm-toolchain/bin/rocm-sdk init

# Sourceable env file rather than bare ENV: the devel root is only knowable after
# `rocm-sdk init`, and the device-bitcode directory has to be located inside it.
RUN ROOT="$(/opt/rocm-toolchain/bin/rocm-sdk path --root)"; \
    BC="$(find "$ROOT" -type d -name bitcode -print -quit)"; \
    { echo "export ROCM_PATH=$ROOT HIP_PATH=$ROOT ROCM_HOME=$ROOT"; \
      echo "export CMAKE_PREFIX_PATH=$ROOT/lib/cmake"; \
      echo "export PATH=$ROOT/bin:$ROOT/lib/llvm/bin:/usr/lib/ccache:\$PATH"; \
      echo "export CC=$ROOT/lib/llvm/bin/amdclang CXX=$ROOT/lib/llvm/bin/amdclang++"; \
      echo "export HIP_DEVICE_LIB_PATH=$BC"; \
      echo "export CMAKE_ARGS='-DROCM_PATH=$ROOT -DHIP_PATH=$ROOT -DAMDGPU_TARGETS=${PYTORCH_ROCM_ARCH} -DHIP_ARCHITECTURES=${PYTORCH_ROCM_ARCH}'"; \
    } > /etc/profile.d/vllm-hip-toolchain.sh

# Build front-end and test deps into the RELEASE venv (the one that runs code).
RUN --mount=type=cache,target=/root/.cache/pip \
    /opt/venv/bin/pip install \
        "setuptools>=77.0.3,<80.0.0" wheel "setuptools-scm>=8" \
        "setuptools-rust>=1.9.0" "cmake>=3.26.1,<4" pybind11 ninja \
        pytest pytest-asyncio tblib

COPY . /src/vllm

# vLLM's own runtime deps, minus anything that would replace the base image's
# working release torch/triton stack or drag in a second ROCm SDK.
RUN --mount=type=cache,target=/root/.cache/pip \
    cd /src/vllm; \
    grep -vhiE '^[[:space:]]*(-r|#|$)' requirements/rocm.txt requirements/common.txt \
      | grep -viE '^[[:space:]]*(torch|pytorch-triton-rocm|triton|torchvision|torchaudio|rocm[-_]?sdk[-_a-z]*)([[:space:]]|==|>|<|~|;|\[|$)' \
      | sort -u > /tmp/dev-reqs.txt; \
    /opt/venv/bin/pip install -r /tmp/dev-reqs.txt xxhash

# Editable install with VLLM_TARGET_DEVICE=empty: metadata and the .pth finder
# only, no compile. The finder maps the package name to /src/vllm/vllm, which the
# bind mount supplies at run time, so `import vllm` resolves to the checkout and
# picks up whatever .so the last incremental build wrote there. Compiling here
# would bake objects into the image and defeat the purpose of the stage.
RUN --mount=type=cache,target=/root/.cache/pip \
    cd /src/vllm \
 && VLLM_TARGET_DEVICE=empty /opt/venv/bin/pip install -e . \
        --no-build-isolation --no-deps

# Incremental build entry point. --build-temp is the load-bearing flag: `pip
# install -e .` puts the cmake tree in a throwaway /tmp dir, so ninja starts from
# scratch every time and only ccache saves any work. Pointing build_ext at the
# build volume keeps build.ninja and the objects, so an untouched tree is a
# no-op and a one-TU edit recompiles exactly that TU.
RUN { echo '#!/usr/bin/env bash'; \
      echo 'set -euo pipefail'; \
      echo '. /etc/profile.d/vllm-hip-toolchain.sh'; \
      echo 'mkdir -p "$VLLM_DEV_BUILD_ROOT/cmake" "$VLLM_DEV_BUILD_ROOT/deps" "$CCACHE_DIR"'; \
      echo 'cd /src/vllm'; \
      echo 'exec /opt/venv/bin/python setup.py build_ext --inplace \'; \
      echo '  --build-temp="$VLLM_DEV_BUILD_ROOT/cmake" "$@"'; \
    } > /usr/local/bin/vllm-hip-build \
 && chmod +x /usr/local/bin/vllm-hip-build

# git refuses to read a mount owned by another uid ("dubious ownership"), which
# setuptools-scm hits when deriving the version.
RUN git config --global --add safe.directory /src/vllm

LABEL org.opencontainers.image.title="vllm-strix-devtools" \
      org.opencontainers.image.description="Incremental HIP/C++ kernel build environment for AMD Strix Halo gfx1151" \
      org.opencontainers.image.source="https://github.com/randomvariable/vllm/tree/homelabs-main/homelab" \
      org.randomvariable.vllm.rocm="7.14" \
      org.randomvariable.vllm.gpu-arch="gfx1151" \
      org.randomvariable.vllm.role="devtools"

WORKDIR /src/vllm
ENTRYPOINT []
CMD ["bash"]

# ---------------------------------------------------------------------------
FROM ${BASE_IMAGE} AS runtime
ARG PYTORCH_ROCM_ARCH

# Runtime env for Strix Halo / gfx1151 (kyuz0-validated knobs). NO
# HSA_OVERRIDE_GFX_VERSION (gfx1151 is native in this ROCm). The base image's
# RELEASE torch + HSA runtime initialise the 8060S without the nightly InitDma
# crash, so the nightly SDK is deliberately absent here.
ENV DEBIAN_FRONTEND=noninteractive \
    VLLM_TARGET_DEVICE=rocm \
    VLLM_USE_RUST_FRONTEND=1 \
    PYTORCH_ROCM_ARCH=${PYTORCH_ROCM_ARCH} \
    HIP_VISIBLE_DEVICES=0 \
    ROCBLAS_USE_HIPBLASLT=1 \
    TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 \
    FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE \
    VLLM_USE_TRITON_AWQ=1 \
    MIOPEN_FIND_MODE=FAST \
    VLLM_DISABLE_COMPILE_CACHE=1 \
    HIP_FORCE_DEV_KERNARG=1 \
    RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES=1 \
    HF_HOME=/opt/vllm/cache/huggingface \
    VLLM_CACHE_ROOT=/opt/vllm/cache

COPY --from=builder /wheels /wheels
COPY --from=builder /wheels-gguf /wheels-gguf
COPY --from=builder /runtime-requirements /runtime-requirements
# Install runtime deps from the fork's ROCm requirements MINUS torch/triton/
# vision/audio (the base image already provides the working release rocm build);
# then the vLLM wheel --no-deps, then the GGUF plugin wheel if it built.
# xxhash128 prefix-cache support (--prefix-caching-hash-algo xxhash)
RUN --mount=type=cache,target=/root/.cache/pip \
    set -eux; \
    grep -vhiE '^[[:space:]]*(-r|#|$)' \
        /runtime-requirements/rocm.txt /runtime-requirements/common.txt \
      | grep -viE '^[[:space:]]*(torch|pytorch-triton-rocm|triton|torchvision|torchaudio|rocm[-_]?sdk[-_a-z]*)([[:space:]]|==|>|<|~|;|\[|$)' \
      | sort -u > /tmp/runtime-reqs.txt; \
    /opt/venv/bin/pip install --no-cache-dir -r /tmp/runtime-reqs.txt; \
    /opt/venv/bin/pip install --no-cache-dir xxhash; \
    /opt/venv/bin/pip install --no-cache-dir --no-deps /wheels/vllm-*.whl; \
    if ls /wheels-gguf/*.whl >/dev/null 2>&1; then \
        /opt/venv/bin/pip install --no-cache-dir --no-deps /wheels-gguf/*.whl \
          && /opt/venv/bin/pip install --no-cache-dir "gguf>=0.17.0" \
          && echo "GGUF plugin installed"; \
    else echo "GGUF plugin wheel absent; skipping"; fi; \
    rm -rf /wheels /wheels-gguf

# Build-time smoke. The HIP-extension import triggers torch GPU init, which needs
# /dev/kfd + /dev/dri -- absent in a rootless BuildKit pod -- so both checks are
# NON-FATAL here. Authoritative import+GPU validation is the on-box `docker run`
# with the devices mounted (see README "Validation"); this only surfaces gross
# packaging breakage in build logs without blocking a GPU-less image build.
RUN /opt/venv/bin/python -c "import vllm; print('vllm', vllm.__version__)" \
      || echo "WARN: import vllm failed at build time (expected without GPU device); validate on-box"; \
    /opt/venv/bin/python -c "import vllm._C, vllm._rocm_C; print('HIP extensions import OK')" \
      || echo "WARN: HIP-extension import needs GPU device; validate on-box with docker run --device /dev/kfd --device /dev/dri"

RUN mkdir -p "$HF_HOME" "$VLLM_CACHE_ROOT"

LABEL org.opencontainers.image.title="vllm-strix-runtime" \
      org.opencontainers.image.description="vLLM runtime for AMD Strix Halo gfx1151 (Radeon 8060S)" \
      org.opencontainers.image.source="https://github.com/randomvariable/vllm/tree/homelabs-main/homelab" \
      org.randomvariable.vllm.rocm="7.14" \
      org.randomvariable.vllm.gpu-arch="gfx1151" \
      org.randomvariable.vllm.build-policy="nightly TheRock SDK builds; release ROCm 7.14 runtime (InitDma-safe); no source patches"

WORKDIR /opt/vllm
ENTRYPOINT ["vllm"]
CMD ["serve"]

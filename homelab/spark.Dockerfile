# syntax=docker/dockerfile:1.7

ARG CUDA_TAG=13.0.2
ARG UBUNTU_TAG=ubuntu24.04

# Build parallelism. Defaults suit the DGX Spark (20 CPU / 128 GB); the CI
# harness overrides these low (e.g. MAX_JOBS=2) when building on the 4 CPU /
# 16 GB Raspberry Pi build nodes so the CUDA compile fits in memory.
ARG MAX_JOBS=6
ARG CMAKE_BUILD_PARALLEL_LEVEL=6
ARG NVCC_THREADS=3

# GGUF quantization plugin (out-of-tree; CUDA supported). Its extension is
# compiled in the builder and the resulting wheel is installed into the runtime.
ARG GGUF_PLUGIN_REPOSITORY=https://github.com/vllm-project/vllm-gguf-plugin.git
# Pinned SHA. Unpinned `main` on an out-of-tree repo is non-reproducible; bump
# deliberately.
ARG GGUF_PLUGIN_REF=1df60c43f1f1274681bb957e5bb9b8f5c44d2f4d

FROM nvidia/cuda:${CUDA_TAG}-devel-${UBUNTU_TAG} AS builder
ARG GGUF_PLUGIN_REPOSITORY
ARG GGUF_PLUGIN_REF
ARG MAX_JOBS
ARG CMAKE_BUILD_PARALLEL_LEVEL
ARG NVCC_THREADS
ENV DEBIAN_FRONTEND=noninteractive \
    MAX_JOBS=${MAX_JOBS} \
    CMAKE_BUILD_PARALLEL_LEVEL=${CMAKE_BUILD_PARALLEL_LEVEL} \
    NVCC_THREADS=${NVCC_THREADS} \
    TORCH_CUDA_ARCH_LIST=12.1a \
    VLLM_TARGET_DEVICE=cuda \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    CCACHE_DIR=/root/.ccache \
    CCACHE_MAXSIZE=20G \
    CCACHE_NOHASHDIR=true \
    CCACHE_COMPILERCHECK=content \
    CCACHE_SLOPPINESS=time_macros,include_file_mtime,include_file_ctime \
    VERBOSE=1 \
    CMAKE_VERBOSE_MAKEFILE=ON

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    rm -f /etc/apt/apt.conf.d/docker-clean && \
    apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl git build-essential cmake ninja-build pkg-config ccache \
    python3-dev python3-venv python3-pip

# This Dockerfile lives in the vllm fork (homelab/), so the build context IS
# the source -- a straight copy, no clone. An external CI harness owns which
# commit is checked out. .git comes along so
# setuptools-scm can derive the version; the built commit is recorded for
# image provenance.
COPY . /src/vllm
RUN cd /src/vllm && git rev-parse HEAD > /src/vllm-build-commit

# Rust vllm-rs frontend (experimental, opt-in via VLLM_USE_RUST_FRONTEND=1 at
# runtime -- our production serving path uses the default Python frontend and
# never touches this binary). Pure-Rust, no CUDA deps; aarch64-safe (mimalloc
# chosen upstream specifically for aarch64 64K-page compatibility). Must run
# BEFORE the wheel build: setup.py's rust_extensions is optional=True, so
# without a precompiled binary present setuptools-rust silently skips it and
# the wheel ships with no vllm-rs at all (no error, no warning that survives
# to build logs). Building it here means setup.py's precompiled_build_rust
# path finds vllm/vllm-rs already in place and bundles it into the wheel via
# package_data, instead of silently omitting it.
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    --mount=type=cache,target=/root/.cache/pip \
    --mount=type=cache,target=/root/.rustup,sharing=locked \
    --mount=type=cache,target=/root/.cargo/registry,sharing=locked \
    --mount=type=cache,target=/root/.cargo/git,sharing=locked \
    --mount=type=cache,target=/src/vllm/target,sharing=locked \
    rm -f /etc/apt/apt.conf.d/docker-clean && \
    apt-get update && apt-get install -y --no-install-recommends \
    curl python3-pip protobuf-compiler libprotobuf-dev && \
    pip3 install --break-system-packages --upgrade 'setuptools>=77.0.3' setuptools-rust && \
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | \
      sh -s -- -y --default-toolchain none && \
    . "$HOME/.cargo/env" && \
    cd /src/vllm && \
    ./build_rust.sh && \
    test -x /src/vllm/vllm/vllm-rs && \
    ls /src/vllm/vllm/_rust_tool_parser*.so >/dev/null 2>&1 && \
    cp /src/vllm/vllm/vllm-rs /tmp/vllm-rs.built

RUN --mount=type=cache,target=/root/.ccache \
    --mount=type=cache,target=/root/.cache/pip \
    --mount=type=cache,target=/opt/venv,sharing=locked \
    --mount=type=cache,target=/src/vllm/.deps,sharing=locked \
    python3 -m venv /opt/venv && \
    /opt/venv/bin/pip install --upgrade pip setuptools wheel && \
    /opt/venv/bin/pip install --index-url https://download.pytorch.org/whl/cu130 \
      'torch==2.11.0' && \
    cd /src/vllm && \
    /opt/venv/bin/pip install -r requirements/build/cuda.txt --extra-index-url https://download.pytorch.org/whl/cu130 && \
    ccache --version | head -1 && ccache -z && \
    /opt/venv/bin/pip wheel -v --no-build-isolation --no-deps --wheel-dir /wheels . && \
    ccache -s && \
    python3 -c "import glob, sys, zipfile; whl = glob.glob('/wheels/vllm-*.whl')[0]; names = zipfile.ZipFile(whl).namelist(); missing = [artifact for artifact, present in [('vllm-rs', any(n.endswith('vllm/vllm-rs') for n in names)), ('_rust_tool_parser', any(n.startswith('vllm/_rust_tool_parser') and n.endswith('.so') for n in names))] if not present]; sys.exit(0) if not missing else (print(f'FATAL: {\", \".join(missing)} missing from wheel -- rust build did not bundle', file=sys.stderr), sys.exit(1))"

# GGUF plugin wheel is built here with the CUDA toolchain and installed into the
# runtime later. Kept non-fatal so plugin-side breakage never blocks core vLLM.
RUN --mount=type=cache,target=/root/.cache/pip \
    set -eux; \
    git init -q /src/gguf-plugin; \
    git -C /src/gguf-plugin remote add origin "${GGUF_PLUGIN_REPOSITORY}"; \
    git -C /src/gguf-plugin fetch --depth 1 origin "${GGUF_PLUGIN_REF}"; \
    git -C /src/gguf-plugin checkout -q FETCH_HEAD; \
    /opt/venv/bin/pip wheel --no-build-isolation --no-deps \
        --wheel-dir /wheels-gguf /src/gguf-plugin \
      || { echo "WARN: GGUF plugin wheel build failed; core image continues"; \
           mkdir -p /wheels-gguf; }

# The wheel is installed --no-deps in the runtime stage, so its Requires-Dist
# metadata never drives dependency resolution here -- no FlashInfer metadata
# patching is needed. The fork's requirements/cuda.txt carries the full
# FlashInfer set (incl. the cu130 JIT cache) for the runtime install below.

RUN mkdir -p /runtime-requirements && \
    cp /src/vllm/requirements/cuda.txt /runtime-requirements/cuda.txt && \
    cp /src/vllm/requirements/common.txt /runtime-requirements/common.txt

# NOTE: vllm-rs is already built once (above, before the wheel step) and that
# binary is bundled into the wheel via package_data + verified present by the
# zipfile check. A second full `./build_rust.sh` invocation here used to
# recompile the identical binary from the same source a second time for no
# purpose it was never copied anywhere or reused; the wheel already carries
# vllm-rs. Removed as pure wasted compute (rustup toolchain fetch + full cargo
# build repeated with no output consumed).

FROM nvidia/cuda:${CUDA_TAG}-runtime-${UBUNTU_TAG} AS runtime
ARG CUDA_TAG
ENV DEBIAN_FRONTEND=noninteractive \
    UV_BREAK_SYSTEM_PACKAGES=1 \
    VLLM_TARGET_DEVICE=cuda \
    VLLM_USE_RUST_FRONTEND=1 \
    TORCH_CUDA_ARCH_LIST=12.1a \
    FLASHINFER_CUDA_ARCH_LIST=12.1a \
    CUTE_DSL_ARCH=sm_121a \
    TIKTOKEN_ENCODINGS_BASE=/opt/vllm/tiktoken_encodings \
    HF_HOME=/opt/vllm/cache/huggingface \
    VLLM_CACHE_ROOT=/opt/vllm/cache

# gcc/g++ are required at RUNTIME: Triton/Inductor JIT-compiles CUDA kernels
# during profile_run and CUDA graph capture (DFlash + sm_121a) by shelling out
# to a host C compiler. Without it the engine dies with
# "InductorError: Failed to find C compiler". cuda-nvcc provides ptxas/nvcc for
# the device side; the base runtime image lacks the host compiler.
# gcc/g++ + python3-dev are required at RUNTIME: Triton/Inductor JIT-compiles
# CUDA kernels during profile_run and CUDA graph capture (DFlash + sm_121a) by
# shelling out to a host C compiler. The generated cuda_utils.c does
# #include <Python.h>, so python3-dev (providing /usr/include/python3.12/Python.h)
# is needed alongside gcc; libcuda.so.1 is injected by the GPU runtime at run.
# Without these the engine dies with InductorError at determine_available_memory.
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    rm -f /etc/apt/apt.conf.d/docker-clean && \
    apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl git gcc g++ python3-dev \
    libibverbs1 librdmacm1 libnuma1 libgomp1 \
    libucx0 python3 python3-pip && \
    useradd --create-home --uid 10001 --shell /usr/sbin/nologin vllm

RUN python3 -m pip install --no-cache-dir --break-system-packages uv

COPY --from=builder /wheels /wheels
COPY --from=builder /wheels-gguf /wheels-gguf
COPY --from=builder /runtime-requirements /runtime-requirements
COPY --from=builder /src/vllm-build-commit /opt/vllm-build-commit

# requirements/cuda.txt carries the full runtime set (FlashInfer incl. the cu130
# JIT cache, nixl, ray, tiktoken) with the needed extra index URLs, so install
# straight from it -- no sed stripping or explicit per-package pins here.
# xxhash128 prefix-cache support (--prefix-caching-hash-algo xxhash)
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system -r /runtime-requirements/cuda.txt \
    --index-strategy unsafe-best-match && \
    uv pip install --system xxhash && \
    uv pip install --system --no-deps /wheels/*.whl && \
    if ls /wheels-gguf/*.whl >/dev/null 2>&1; then \
        uv pip install --system --no-deps /wheels-gguf/*.whl && \
        uv pip install --system "gguf>=0.17.0" && \
        echo "GGUF plugin installed"; \
    else echo "GGUF plugin wheel absent; skipping"; fi

RUN rm -rf /wheels /wheels-gguf

RUN python3 -m compileall -q "$(python3 -c 'import os, vllm; print(os.path.dirname(vllm.__file__))')" || true

RUN mkdir -p /opt/vllm/tiktoken_encodings "$HF_HOME" "$VLLM_CACHE_ROOT" && \
    curl --fail --silent --show-error --location \
      -o /opt/vllm/tiktoken_encodings/o200k_base.tiktoken \
      https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken && \
    curl --fail --silent --show-error --location \
      -o /opt/vllm/tiktoken_encodings/cl100k_base.tiktoken \
      https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken && \
    chown -R vllm:vllm /opt/vllm

LABEL org.opencontainers.image.title="vllm-spark-runtime" \
      org.opencontainers.image.description="Stock model-neutral vLLM runtime for DGX Spark sm_121a" \
      org.opencontainers.image.source="https://github.com/randomvariable/vllm/tree/homelabs-main/homelab" \
      org.randomvariable.vllm.cuda="13.0.2" \
      org.randomvariable.vllm.patch-policy="UMA clamp (PR #46932) compiled into fork source; no runtime patch" \
      org.randomvariable.vllm.distributed-executor-backend="ray"

USER vllm
WORKDIR /opt/vllm
ENTRYPOINT ["vllm"]
CMD ["serve"]

# syntax=docker/dockerfile:1.7

ARG MAX_JOBS=16
ARG CMAKE_BUILD_PARALLEL_LEVEL=16
ARG NVCC_THREADS=4
ARG GGUF_PLUGIN_REPOSITORY=https://github.com/vllm-project/vllm-gguf-plugin.git
ARG GGUF_PLUGIN_REF=1df60c43f1f1274681bb957e5bb9b8f5c44d2f4d

FROM --platform=linux/amd64 nvidia/cuda:13.0.2-devel-ubuntu24.04 AS builder
ARG MAX_JOBS
ARG CMAKE_BUILD_PARALLEL_LEVEL
ARG NVCC_THREADS
ARG GGUF_PLUGIN_REPOSITORY
ARG GGUF_PLUGIN_REF
ENV DEBIAN_FRONTEND=noninteractive \
    PATH=/opt/venv/bin:/root/.cargo/bin:${PATH} \
    CUDA_HOME=/usr/local/cuda-13.0 \
    VLLM_TARGET_DEVICE=cuda \
    TORCH_CUDA_ARCH_LIST="12.0 12.1a" \
    NVCC_PREPEND_FLAGS="-target-dir sbsa-linux -ccbin /usr/bin/aarch64-linux-gnu-g++" \
    CMAKE_ARGS="-DCMAKE_TOOLCHAIN_FILE=/opt/sbsa-toolchain.cmake -DTorch_DIR=/opt/torch-aarch64/torch/share/cmake/Torch -DPython_EXECUTABLE=/opt/venv/bin/python3 -DPython_INCLUDE_DIRS=/usr/include/python3.12 -DPython_FIND_STRATEGY=LOCATION --debug-find-pkg=Python" \
    MAX_JOBS=${MAX_JOBS} \
    CMAKE_BUILD_PARALLEL_LEVEL=${CMAKE_BUILD_PARALLEL_LEVEL} \
    NVCC_THREADS=${NVCC_THREADS} \
    CARGO_BUILD_TARGET=aarch64-unknown-linux-gnu \
    CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER=aarch64-linux-gnu-gcc \
    CC_aarch64_unknown_linux_gnu=aarch64-linux-gnu-gcc \
    CXX_aarch64_unknown_linux_gnu=aarch64-linux-gnu-g++ \
    AR_aarch64_unknown_linux_gnu=aarch64-linux-gnu-ar \
    PYO3_CROSS_PYTHON_VERSION=3.12 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    CCACHE_DIR=/root/.ccache-cross \
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
      binutils-aarch64-linux-gnu ca-certificates ccache cmake curl file git \
      gcc-aarch64-linux-gnu g++-aarch64-linux-gnu libc6-dev-arm64-cross \
      libprotobuf-dev make ninja-build perl pkg-config protobuf-compiler \
      python3-dev python3-pip python3-venv unzip wget && \
    wget -q -O /tmp/cuda-keyring.deb \
      https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/cross-linux-sbsa/cuda-keyring_1.1-1_all.deb && \
    dpkg -i /tmp/cuda-keyring.deb && \
    apt-get update && apt-get install -y --no-install-recommends \
      cuda-cross-sbsa-13-0 && \
    rm -f /tmp/cuda-keyring.deb && \
    printf '__global__ void k(){}' > /tmp/t.cu && \
    nvcc -target-dir sbsa-linux \
      -ccbin /usr/bin/aarch64-linux-gnu-g++ -arch=sm_121a \
      -c /tmp/t.cu -o /tmp/t.o && \
    file /tmp/t.o | grep -qi aarch64 && \
    rm -f /tmp/t.cu /tmp/t.o

COPY requirements/build/cuda.txt requirements/build/rust.txt /tmp/build-requirements/
# Builder Python imports native x86 torch for setup probes. CMake links only the
# separately extracted aarch64 torch tree below.
RUN --mount=type=cache,target=/root/.cache/uv \
    curl -LsSf https://astral.sh/uv/install.sh | sh && \
    /root/.local/bin/uv venv --seed --python 3.12 /opt/venv && \
    /root/.local/bin/uv pip install --python /opt/venv/bin/python \
      'torch==2.13.0' \
      --index-url https://download.pytorch.org/whl/cu130 && \
    /root/.local/bin/uv pip install --python /opt/venv/bin/python \
      -r /tmp/build-requirements/cuda.txt \
      -r /tmp/build-requirements/rust.txt \
      --extra-index-url https://download.pytorch.org/whl/cu130 \
      --index-strategy unsafe-best-match && \
    rm -rf /tmp/build-requirements && \
    /opt/venv/bin/python -c "import torch; print('builder torch OK', torch.__version__, 'cuda', torch.version.cuda)"

RUN curl --fail --silent --show-error --location \
      -o /tmp/torch-aarch64.whl \
      https://download.pytorch.org/whl/cu130/torch-2.13.0%2Bcu130-cp312-cp312-manylinux_2_28_aarch64.whl && \
    mkdir -p /opt/torch-aarch64 && \
    unzip -q /tmp/torch-aarch64.whl -d /opt/torch-aarch64 && \
    rm -f /tmp/torch-aarch64.whl && \
    test -f /opt/torch-aarch64/torch/share/cmake/Torch/TorchConfig.cmake

RUN printf '%s\n' \
      'set(CMAKE_SYSTEM_NAME Linux)' \
      'set(CMAKE_SYSTEM_PROCESSOR aarch64)' \
      'set(CMAKE_C_COMPILER /usr/bin/aarch64-linux-gnu-gcc)' \
      'set(CMAKE_CXX_COMPILER /usr/bin/aarch64-linux-gnu-g++)' \
      'set(CMAKE_CUDA_HOST_COMPILER /usr/bin/aarch64-linux-gnu-g++)' \
      'set(CMAKE_FIND_ROOT_PATH /usr/aarch64-linux-gnu /opt/torch-aarch64)' \
      'set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)' \
      'set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY BOTH)' \
      'set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE BOTH)' \
      'set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE BOTH)' \
      > /opt/sbsa-toolchain.cmake

RUN --mount=type=cache,target=/root/.rustup,sharing=locked \
    --mount=type=cache,target=/root/.cargo/registry,sharing=locked \
    --mount=type=cache,target=/root/.cargo/git,sharing=locked \
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | \
      sh -s -- -y --default-toolchain none && \
    rustup toolchain install 1.95 && \
    rustup target add --toolchain 1.95 aarch64-unknown-linux-gnu

# Build context is fork source. Keep .git so setuptools-scm derives its version,
# and record exact source commit in runtime image.
COPY . /src/vllm
RUN cd /src/vllm && git rev-parse HEAD > /src/vllm-build-commit

RUN --mount=type=cache,target=/root/.rustup,sharing=locked \
    --mount=type=cache,target=/root/.cargo/registry,sharing=locked \
    --mount=type=cache,target=/root/.cargo/git,sharing=locked \
    --mount=type=cache,target=/src/vllm/target,sharing=locked \
    cd /src/vllm && \
    ./build_rust.sh && \
    test -x vllm/vllm-rs && \
    readelf -h vllm/vllm-rs | grep -q 'Machine:.*AArch64' && \
    rust_so="$(find vllm -maxdepth 1 -name '_rust_tool_parser*.so' -print -quit)" && \
    test -n "$rust_so" && readelf -h "$rust_so" | grep -q 'Machine:.*AArch64'

RUN --mount=type=cache,target=/root/.ccache-cross,sharing=locked \
    --mount=type=cache,target=/root/.cache/uv \
    --mount=type=cache,target=/src/vllm/.deps,sharing=locked \
    cd /src/vllm && \
    ccache -z && \
    _PYTHON_HOST_PLATFORM=linux-aarch64 python3 setup.py bdist_wheel --dist-dir /wheels \
      --py-limited-api=cp38 --plat-name linux_aarch64 && \
    cache="$(find build -name CMakeCache.txt -print -quit)" && \
    test -n "$cache" && \
    grep -q '^CMAKE_CROSSCOMPILING:INTERNAL=TRUE$' "$cache" && \
    grep -q '^Torch_DIR:.*=/opt/torch-aarch64/torch/share/cmake/Torch$' "$cache" && \
    so="$(find build -type f -name '*.so' -print -quit)" && \
    test -n "$so" && readelf -h "$so" | grep -q 'Machine:.*AArch64' && \
    ccache -s && \
    python3 -c "import glob, sys, zipfile; whl = glob.glob('/wheels/vllm-*.whl')[0]; names = zipfile.ZipFile(whl).namelist(); missing = [artifact for artifact, present in [('vllm-rs', any(n.endswith('vllm/vllm-rs') for n in names)), ('_rust_tool_parser', any(n.startswith('vllm/_rust_tool_parser') and n.endswith('.so') for n in names))] if not present]; sys.exit(0) if not missing else (print(f'FATAL: {\", \".join(missing)} missing from wheel -- rust build did not bundle', file=sys.stderr), sys.exit(1))"

# GGUF remains outside core image's critical path until its extension reliably
# cross-compiles. Any fetch or build failure leaves an empty optional wheel dir.
RUN --mount=type=cache,target=/root/.cache/uv \
    mkdir -p /wheels-gguf && \
    if git init -q /src/gguf-plugin && \
       git -C /src/gguf-plugin remote add origin "${GGUF_PLUGIN_REPOSITORY}" && \
       git -C /src/gguf-plugin fetch --depth 1 origin "${GGUF_PLUGIN_REF}" && \
       git -C /src/gguf-plugin checkout -q FETCH_HEAD && \
       python3 -m pip wheel --no-build-isolation --no-deps \
         --wheel-dir /wheels-gguf /src/gguf-plugin; then \
      echo "GGUF plugin wheel built"; \
    else \
      echo "WARN: GGUF plugin wheel build failed; core image continues"; \
      rm -f /wheels-gguf/*.whl; \
    fi

RUN mkdir -p /runtime-requirements && \
    cp /src/vllm/requirements/cuda.txt /runtime-requirements/cuda.txt && \
    cp /src/vllm/requirements/common.txt /runtime-requirements/common.txt

FROM --platform=$TARGETPLATFORM nvidia/cuda:13.0.2-runtime-ubuntu24.04 AS runtime
ENV DEBIAN_FRONTEND=noninteractive \
    UV_BREAK_SYSTEM_PACKAGES=1 \
    VLLM_TARGET_DEVICE=cuda \
    VLLM_USE_RUST_FRONTEND=1 \
    VLLM_USE_FLASHINFER=0 \
    TORCH_CUDA_ARCH_LIST=12.1a \
    FLASHINFER_CUDA_ARCH_LIST=12.1a \
    CUTE_DSL_ARCH=sm_121a \
    TIKTOKEN_ENCODINGS_BASE=/opt/vllm/tiktoken_encodings \
    HF_HOME=/opt/vllm/cache/huggingface \
    VLLM_CACHE_ROOT=/opt/vllm/cache

# Triton/Inductor JIT needs gcc/g++ and Python.h while profiling and capturing
# CUDA graphs; these are runtime dependencies, not builder leftovers.
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

# FlashInfer 0.6.15 is incompatible with torch 2.13/cu130. Remove its three
# requirements until a CUDA 13-compatible FlashInfer release is available.
RUN --mount=type=cache,target=/root/.cache/uv \
    sed -E '/^[[:space:]]*flashinfer-(python|cubin|jit-cache)/d' \
      /runtime-requirements/cuda.txt \
      > /runtime-requirements/cuda-no-flashinfer.txt && \
    ! grep -Eq '^[[:space:]]*flashinfer-(python|cubin|jit-cache)' \
      /runtime-requirements/cuda-no-flashinfer.txt && \
    uv pip install --system -r /runtime-requirements/cuda-no-flashinfer.txt \
      --extra-index-url https://download.pytorch.org/whl/cu130 \
      --index-strategy unsafe-best-match && \
    uv pip install --system xxhash && \
    uv pip install --system --no-deps /wheels/*.whl && \
    if ls /wheels-gguf/*.whl >/dev/null 2>&1; then \
      uv pip install --system --no-deps /wheels-gguf/*.whl && \
      uv pip install --system 'gguf>=0.17.0' && \
      echo "GGUF plugin installed"; \
    else \
      echo "GGUF plugin wheel absent; skipping"; \
    fi

RUN rm -rf /wheels /wheels-gguf && \
    python3 -m compileall -q \
      "$(python3 -c 'import os, vllm; print(os.path.dirname(vllm.__file__))')" \
      || true

RUN mkdir -p /opt/vllm/tiktoken_encodings "$HF_HOME" "$VLLM_CACHE_ROOT" && \
    curl --fail --silent --show-error --location \
      -o /opt/vllm/tiktoken_encodings/o200k_base.tiktoken \
      https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken && \
    curl --fail --silent --show-error --location \
      -o /opt/vllm/tiktoken_encodings/cl100k_base.tiktoken \
      https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken && \
    chown -R vllm:vllm /opt/vllm

LABEL org.opencontainers.image.title="vllm-spark-cross-runtime" \
      org.opencontainers.image.description="Cross-compiled vLLM runtime for DGX Spark sm_121a" \
      org.opencontainers.image.source="https://github.com/randomvariable/vllm/tree/homelabs-main/homelab" \
      org.randomvariable.vllm.cuda="13.0.2" \
      org.randomvariable.vllm.distributed-executor-backend="ray"

USER vllm
WORKDIR /opt/vllm
ENTRYPOINT ["vllm"]
CMD ["serve"]

# syntax=docker/dockerfile:1.7

ARG VLLM_BUILD_COMMIT=unknown
ARG VLLM_BUILD_PIPELINE=local
ARG VLLM_BUILD_URL=
ARG VLLM_IMAGE_TAG=local/vllm-spark-cross:dev
ARG MAX_JOBS=8
ARG CMAKE_BUILD_PARALLEL_LEVEL=8
ARG NVCC_THREADS=2
ARG GGUF_PLUGIN_REPOSITORY=https://github.com/vllm-project/vllm-gguf-plugin.git
ARG GGUF_PLUGIN_REF=1df60c43f1f1274681bb957e5bb9b8f5c44d2f4d

FROM --platform=linux/amd64 nvidia/cuda:13.3.1-devel-ubuntu24.04 AS builder
ARG MAX_JOBS
ARG CMAKE_BUILD_PARALLEL_LEVEL
ARG NVCC_THREADS
ARG GGUF_PLUGIN_REPOSITORY
ARG GGUF_PLUGIN_REF
ENV DEBIAN_FRONTEND=noninteractive \
    PATH=/opt/venv/bin:/root/.local/bin:/root/.cargo/bin:${PATH} \
    CUDA_HOME=/usr/local/cuda-13.3 \
    CUDA_TOOLKIT_ROOT=/usr/local/cuda-13.3 \
    DEEPGEMM_CXX=/usr/bin/aarch64-linux-gnu-g++ \
    DEEPGEMM_PYTHON_INCLUDE=/usr/include/python3.12 \
    DEEPGEMM_EXT_SUFFIX=.cpython-312-aarch64-linux-gnu.so \
    DEEPGEMM_TORCH_ROOT=/opt/torch-aarch64/torch \
    DEEPGEMM_CUDA_LIB_DIR=/usr/local/cuda-13.3/targets/sbsa-linux/lib \
    DEEPGEMM_TORCH_CXX11_ABI=1 \
    DEEPGEMM_SRC_DIR=/src/vllm/third_party/deep_gemm \
    VLLM_TARGET_DEVICE=cuda \
    TORCH_CUDA_ARCH_LIST="12.1a" \
    NVCC_PREPEND_FLAGS="-target-dir sbsa-linux -ccbin /usr/bin/aarch64-linux-gnu-g++" \
    CMAKE_ARGS="-DCMAKE_TOOLCHAIN_FILE=/opt/sbsa-toolchain.cmake -DTorch_DIR=/opt/torch-aarch64/torch/share/cmake/Torch -DCUDAToolkit_ROOT=/usr/local/cuda-13.3 -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-13.3 -DCUDA_CUDART=/usr/local/cuda-13.3/targets/sbsa-linux/lib/libcudart.so -DVLLM_CUBLAS_LIBRARY=/usr/local/cuda-13.3/targets/sbsa-linux/lib/libcublas.so -DVLLM_CUTLASS_SRC_DIR=/src/vllm/third_party/flashinfer/3rdparty/cutlass" \
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
    PIP_EXTRA_INDEX_URL="https://flashinfer.ai/whl/ https://flashinfer.ai/whl/cu130/ https://download.pytorch.org/whl/cu130" \
    UV_INDEX="https://flashinfer.ai/whl/ https://flashinfer.ai/whl/cu130/ https://download.pytorch.org/whl/cu130" \
    UV_INDEX_STRATEGY=unsafe-best-match \
    CCACHE_DIR=/root/.ccache-cross \
    CCACHE_MAXSIZE=40G \
    CCACHE_NOHASHDIR=true \
    CCACHE_COMPILERCHECK=content \
    CCACHE_EXTRAFILES=/root/.ccache-keyfile \
    CCACHE_SLOPPINESS=time_macros,include_file_mtime,include_file_ctime \
    CCACHE_DEPEND=true \
    VERBOSE=1 \
    CMAKE_VERBOSE_MAKEFILE=ON

# NOHASHDIR keeps the compiler's working directory out of the hash; BASEDIR
# additionally rewrites absolute paths under the source root to relative before
# hashing, so -I/include arguments naming /src/vllm/... cannot key differently
# between builds. Both are needed: NOHASHDIR covers the CWD, BASEDIR covers the
# arguments. Neither affects CCACHE_EXTRAFILES, which is hashed by content and
# is what carries the effective cross-target flags into the key.
#
# DEPEND keys on the compiler's own dependency output instead of making ccache
# preprocess every unit itself. Both compile paths already pass a dependency
# option (CMake uses -MD, FlashInfer's AOT rules use nvcc
# --generate-dependencies-with-compile), and on the 2026-08-12 build 2740 of
# 2755 FlashInfer lookups fell out of direct mode into the slower preprocessed
# lookup; depend mode removes that second path for both hits and misses.
ENV CCACHE_BASEDIR=/src/vllm

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
      cuda-cross-sbsa-13-3 && \
    rm -f /tmp/cuda-keyring.deb && \
    ln -sfn /usr/local/cuda-13.3/targets/sbsa-linux \
      /usr/local/cuda-13.3/targets/aarch64-linux && \
    for stub in /usr/local/cuda-13.3/targets/sbsa-linux/lib/stubs/*.so; do \
      ln -sfn "stubs/$(basename "$stub")" \
        "/usr/local/cuda-13.3/targets/sbsa-linux/lib/$(basename "$stub")"; \
    done && \
    printf '__global__ void k(){}' > /tmp/t.cu && \
    nvcc -target-dir sbsa-linux \
      -ccbin /usr/bin/aarch64-linux-gnu-g++ -arch=sm_121a \
      -c /tmp/t.cu -o /tmp/t.o && \
    file /tmp/t.o | grep -qi aarch64 && \
    rm -f /tmp/t.cu /tmp/t.o

# Ubuntu 24.04 ships ccache 4.9.1, which cannot preprocess nvcc's
# --generate-dependencies-with-compile and reports every such call as
# "Preprocessing failed" (uncacheable). FlashInfer's AOT build uses that flag
# for all ~3400 of its compile units, so apt's ccache caches none of them.
# 4.13.x handles it, so pin a static build ahead of /usr/bin on PATH.
ARG CCACHE_VERSION=4.13.6
ARG CCACHE_SHA256=156ec57c5198cc849d92834023d09910b83dc5504c6cf405d09e6ae7b208a3e5
RUN cd /tmp && \
    wget -q -O ccache.tar.xz \
      "https://github.com/ccache/ccache/releases/download/v${CCACHE_VERSION}/ccache-${CCACHE_VERSION}-linux-x86_64-musl-static.tar.xz" && \
    echo "${CCACHE_SHA256}  ccache.tar.xz" | sha256sum -c - && \
    tar -xf ccache.tar.xz && \
    install -m755 "ccache-${CCACHE_VERSION}-linux-x86_64-musl-static/ccache" \
      /usr/local/bin/ccache && \
    rm -rf ccache.tar.xz "ccache-${CCACHE_VERSION}-linux-x86_64-musl-static" && \
    [ "$(command -v ccache)" = /usr/local/bin/ccache ] && \
    ccache --version | head -1

# The aarch64 cross compiler defines __aarch64__, so the multiarch wrapper at
# /usr/include/python3.12/pyconfig.h does
#   #include <aarch64-linux-gnu/python3.12/pyconfig.h>
# resolving via the cross multiarch dir /usr/include/aarch64-linux-gnu. Only x86
# python3-dev is installed, so that arch-specific pyconfig.h is absent. Python.h
# etc. are arch-neutral (already present); only pyconfig.h must come from the
# arm64 libpython3.12-dev. Pull just that .deb from ports.ubuntu.com and extract
# its arch-specific include dir. Version is read from the host's installed
# libpython3.12-dev so the arm64 headers always match the host python3.12 (same
# noble source package) -- no hardcoded point version, no multiarch apt state.
RUN set -eux; \
    ver="$(dpkg-query -W -f='${Version}' libpython3.12-dev)"; \
    curl -fsSL -o /tmp/libpython3.12-dev-arm64.deb \
      "https://ports.ubuntu.com/ubuntu-ports/pool/main/p/python3.12/libpython3.12-dev_${ver}_arm64.deb"; \
    dpkg-deb -x /tmp/libpython3.12-dev-arm64.deb /tmp/pyarm64; \
    mkdir -p /usr/include/aarch64-linux-gnu; \
    cp -a /tmp/pyarm64/usr/include/aarch64-linux-gnu/python3.12 \
          /usr/include/aarch64-linux-gnu/python3.12; \
    test -f /usr/include/aarch64-linux-gnu/python3.12/pyconfig.h; \
    rm -rf /tmp/pyarm64 /tmp/libpython3.12-dev-arm64.deb

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
      -r /tmp/build-requirements/rust.txt && \
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

# FlashInfer only needs its own submodule tree and the build script, so copy
# just those. Keeping this stage ahead of the full source COPY means an edit
# anywhere else in the fork no longer invalidates its ~2h AOT compile layer.
COPY third_party/flashinfer /src/vllm/third_party/flashinfer
COPY tools/flashinfer-build.sh /src/vllm/tools/flashinfer-build.sh

# FlashInfer publishes no cu133 wheel or JIT-cache index. Its cu130 runtime
# wheel remains compatible; CUDA_VERSION controls the 13.3 AOT compiler.
# Build both FlashInfer packages from the pinned recursive submodule. The local
# JIT-cache wheel carries the patched AArch64 SM121 native modules.
RUN --mount=type=cache,id=vllm-spark-ccache-cross,target=/root/.ccache-cross,sharing=locked \
    --mount=type=cache,target=/root/.cache/uv \
    --mount=type=cache,target=/root/.cache/flashinfer,sharing=locked \
    --mount=type=cache,target=/src/vllm/third_party/flashinfer/build/aot,sharing=locked \
    cd /src/vllm && \
    uv pip install --python /opt/venv/bin/python \
      'setuptools>=77' 'packaging>=24' wheel tqdm ninja requests numpy \
      nvidia-ml-py 'apache-tvm-ffi>=0.1,<0.2' filelock && \
    FI_NVCC_PREPEND="-target-dir sbsa-linux" && \
    # 12.1a alone is not sufficient. FlashInfer names each AOT module after the
    # arch it was built for (fp4_quantization_121), but its runtime resolver
    # rewrites SM12x to the family variant "120f" whenever CUDA >= 12.9 and
    # then looks up fp4_quantization_120f. With FLASHINFER_DISABLE_JIT=1 that
    # lookup raises MissingJITCacheError instead of compiling a fallback, so
    # both variants must be present in the wheel.
    FI_ARCH_LIST="12.0f 12.1a" && \
    printf '%s\n' "$FI_NVCC_PREPEND" "$FI_ARCH_LIST" > "$CCACHE_EXTRAFILES" && \
    ccache -z && \
    export CCACHE_LOGFILE=/tmp/ccache-flashinfer.log && \
    rm -f "$CCACHE_LOGFILE" && \
    { ccache --show-config || true; } && \
    CUDA_VERSION=13.3 \
    CC=/usr/bin/aarch64-linux-gnu-gcc \
    CXX=/usr/bin/aarch64-linux-gnu-g++ \
    FLASHINFER_NVCC=/usr/local/cuda-13.3/bin/nvcc \
    FLASHINFER_FMHA_V2_HOST_BUILD=1 \
    FLASHINFER_FMHA_V2_HOST_CXX=/usr/bin/g++ \
    NVCC_PREPEND_FLAGS="$FI_NVCC_PREPEND" \
    FLASHINFER_NVCC_LAUNCHER=ccache \
    FLASHINFER_CXX_LAUNCHER=ccache \
    LIBRARY_PATH="/usr/local/cuda-13.3/targets/sbsa-linux/lib:/usr/local/cuda-13.3/targets/sbsa-linux/lib/stubs" \
    FLASHINFER_EXTRA_LDFLAGS="-L/usr/local/cuda-13.3/targets/sbsa-linux/lib -L/usr/local/cuda-13.3/targets/sbsa-linux/lib/stubs -Wl,-rpath-link,/usr/local/cuda-13.3/targets/sbsa-linux/lib" \
    FLASHINFER_SOURCE_DIR=/src/vllm/third_party/flashinfer \
    FLASHINFER_DIST_DIR=/wheels-flashinfer \
    FLASHINFER_CUDA_ARCH_LIST="$FI_ARCH_LIST" \
    FLASHINFER_WHEEL_PLATFORM_TAG=manylinux_2_28_aarch64 \
    FLASHINFER_JIT_CACHE_LOCAL_VERSION=cu130 \
    # Eight parallel jobs once reached 44 GiB RSS and got the build evicted
    # under a 56 GiB cgroup limit, so this stage was pinned to four. Measured
    # again on 2026-08-13 mid-stage: four jobs held 16.5 GiB RSS and 3.8 of
    # server6's 32 cores, with ~40 GiB of node memory free -- the stage was
    # starved, not memory-bound. Six jobs is ~25 GiB at the observed 4.1 GiB
    # per job, inside the 44 GiB limit with node headroom to spare. Raise
    # further only from a measured build, not from spare core count: nvcc RSS
    # per unit climbs sharply through the FA3 and MLA kernels.
    MAX_JOBS=6 FLASHINFER_NVCC_THREADS=1 \
    BUILD_JIT_CACHE=true BUILD_NVEP=0 \
    ./tools/flashinfer-build.sh && \
    { ccache -sv | tee /src/ccache-stats-flashinfer.txt; \
      ccache --print-stats | tee -a /src/ccache-stats-flashinfer.txt; \
      echo '== effective ccache config =='; \
      ccache --show-config; \
      echo '== result breakdown =='; \
      grep -ao 'Result: .*' "$CCACHE_LOGFILE" | sort | uniq -c | sort -rn; \
      echo '== result and miss-reason frequency, whole log =='; \
      grep -aoE '(Result|[Rr]eason|[Mm]iss): .*' "$CCACHE_LOGFILE" \
        | sed -e 's|/[^ ]*|PATH|g' \
        | sort | uniq -c | sort -rn | head -n 40 || true; \
      echo '== top 40 log messages, timestamps and paths collapsed =='; \
      sed -e 's|^\[[^]]*\] ||' -e 's|/[^ ]*|PATH|g' "$CCACHE_LOGFILE" \
        | sort | uniq -c | sort -rn | head -n 40; \
      echo -n '== distinct collapsed messages: '; \
      sed -e 's|^\[[^]]*\] ||' -e 's|/[^ ]*|PATH|g' "$CCACHE_LOGFILE" \
        | sort -u | wc -l; \
      echo '== first 200 raw log lines =='; \
      head -n 200 "$CCACHE_LOGFILE"; \
      rm -f "$CCACHE_LOGFILE"; \
      true; \
    } 2>&1 | tee /src/ccache-log-flashinfer.txt

# Source commit and setuptools-scm version arrive as build args so the context
# can exclude .git. CI re-clones the fork for every run, and a fresh clone's
# .git bytes (FETCH_HEAD, reflogs, pack checksums) differ even for an
# identical commit -- with .git in the context the COPY layer digest changed
# every build and invalidated the multi-hour CUDA compile layers below.
ARG VLLM_SOURCE_COMMIT=unknown
ARG VLLM_SCM_VERSION=0.1.dev1
# Stage-wide, not per-RUN: the Rust build resolves the project version too, so
# a wheel-only override left ./build_rust.sh failing with "setuptools-scm was
# unable to detect version". setup.py's own VLLM_VERSION_OVERRIDE path just
# forwards to this variable, so setting it once covers every consumer.
ENV SETUPTOOLS_SCM_PRETEND_VERSION=${VLLM_SCM_VERSION}
COPY . /src/vllm
RUN printf '%s\n' "${VLLM_SOURCE_COMMIT}" > /src/vllm-build-commit

RUN --mount=type=cache,target=/root/.rustup,sharing=locked \
    --mount=type=cache,target=/root/.cargo/registry,sharing=locked \
    --mount=type=cache,target=/root/.cargo/git,sharing=locked \
    --mount=type=cache,target=/src/vllm/target,sharing=locked \
    cd /src/vllm && \
    rustup toolchain install 1.95 && \
    rustup target add --toolchain 1.95 aarch64-unknown-linux-gnu && \
    ./build_rust.sh && \
    test -x vllm/vllm-rs && \
    readelf -h vllm/vllm-rs | grep -q 'Machine:.*AArch64' && \
    rust_so="$(find vllm -maxdepth 1 -name '_rust_tool_parser*.so' -print -quit)" && \
    test -n "$rust_so" && readelf -h "$rust_so" | grep -q 'Machine:.*AArch64'

# id= is explicit rather than derived from target so the store keeps a stable
# identity across builds and across any future change of CCACHE_DIR.
RUN --mount=type=cache,id=vllm-spark-ccache-cross,target=/root/.ccache-cross,sharing=locked \
    --mount=type=cache,target=/root/.cache/uv \
    --mount=type=cache,target=/src/vllm/.deps,sharing=locked \
    cd /src/vllm && \
    printf '%s\n' "$NVCC_PREPEND_FLAGS" "$TORCH_CUDA_ARCH_LIST" > "$CCACHE_EXTRAFILES" && \
    { echo '== ccache at PipelineRun entry =='; \
      ccache -sv || true; \
      find "$CCACHE_DIR" -type f 2>/dev/null | wc -l || true; \
      true; \
    } && \
    ccache -z && \
    _PYTHON_HOST_PLATFORM=linux-aarch64 python3 setup.py bdist_wheel --dist-dir /wheels \
      --py-limited-api=cp38 --plat-name linux_aarch64 && \
    ccache -sv | tee /src/ccache-stats-vllm.txt

# Fail the build on an unresolved internal C++ symbol.
#
# A Python extension legitimately leaves CPython and libtorch symbols to be
# resolved at load time, so the link cannot use -z defs. But libtorch's stable
# ABI is C (aoti_torch_*), so no legitimately-deferred symbol takes a
# torch::stable::Tensor argument: any undefined symbol whose mangled name
# contains N5torch6stable6Tensor is one of ours that failed to link.
#
# On 2026-08-13 a call-site `extern` declaration of
# reshape_and_cache_nvfp4_dispatch omitted the trailing kv_cache_dtype
# argument. It mangled to a different symbol than the definition, the
# extension linked clean, and the failure only appeared when the deployed pod
# imported vllm -- after a three-hour build and a rollout. This check costs
# seconds and catches that class in the builder.
RUN set -eu; \
    tmp="$(mktemp -d)"; \
    unzip -q -o /wheels/vllm-*.whl -d "$tmp"; \
    undef="$(find "$tmp" -name '*.so' -exec aarch64-linux-gnu-nm -D -u {} + \
             | grep 'N5torch6stable6Tensor' || true)"; \
    if [ -n "$undef" ]; then \
      echo 'ERROR: unresolved internal symbols taking torch::stable::Tensor:' >&2; \
      echo "$undef" | sort -u | while read -r sym; do \
        echo "  $sym  ->  $(echo "$sym" | sed 's/^ *U *//' | c++filt)" >&2; \
      done; \
      exit 1; \
    fi; \
    echo 'symbol check: no unresolved internal torch::stable symbols'; \
    rm -rf "$tmp"

# GGUF remains outside core image's critical path until its extension reliably
# cross-compiles. Any fetch or build failure leaves an empty optional wheel dir.
# b12x is pinned as a submodule so SM120/SM121 MoE fixes can be carried
# ahead of a PyPI release. It declares no ext_modules, so the wheel is
# arch-independent and needs no cross-compile toolchain.
RUN --mount=type=cache,target=/root/.cache/uv \
    mkdir -p /wheels-b12x && \
    uv build --wheel --out-dir /wheels-b12x \
      /src/vllm/third_party/b12x

RUN --mount=type=cache,target=/root/.cache/uv \
    mkdir -p /wheels-gguf && \
    if git init -q /src/gguf-plugin && \
       git -C /src/gguf-plugin remote add origin "${GGUF_PLUGIN_REPOSITORY}" && \
       git -C /src/gguf-plugin fetch --depth 1 origin "${GGUF_PLUGIN_REF}" && \
       git -C /src/gguf-plugin checkout -q FETCH_HEAD && \
       CXX=aarch64-linux-gnu-g++ CC=aarch64-linux-gnu-gcc \
       LIBRARY_PATH="/opt/torch-aarch64/torch/lib${LIBRARY_PATH:+:$LIBRARY_PATH}" \
       TORCH_CUDA_ARCH_LIST="12.1a" \
       _PYTHON_HOST_PLATFORM=linux-aarch64 \
       python3 -m pip wheel --no-build-isolation --no-deps \
          --wheel-dir /wheels-gguf /src/gguf-plugin; then \
      echo "GGUF plugin wheel built"; \
    else \
      echo "WARN: GGUF plugin wheel build failed; core image continues"; \
      rm -f /wheels-gguf/*.whl; \
    fi

RUN mkdir -p /runtime-requirements && \
    cp /src/vllm/requirements/cuda.txt /runtime-requirements/cuda.txt && \
    cp /src/vllm/requirements/common.txt /runtime-requirements/common.txt && \
    python3 -c 'from pathlib import Path; p=Path("/runtime-requirements/cuda.txt"); p.write_text("".join(line for line in p.read_text().splitlines(keepends=True) if not line.startswith(("--extra-index-url ", "flashinfer-python==", "flashinfer-cubin==", "flashinfer-jit-cache=="))))'

FROM --platform=$TARGETPLATFORM nvidia/cuda:13.3.1-runtime-ubuntu24.04 AS runtime
ARG VLLM_BUILD_COMMIT
ARG VLLM_BUILD_PIPELINE
ARG VLLM_BUILD_URL
ARG VLLM_IMAGE_TAG
ENV DEBIAN_FRONTEND=noninteractive \
    PIP_EXTRA_INDEX_URL="https://flashinfer.ai/whl/ https://flashinfer.ai/whl/cu130/ https://download.pytorch.org/whl/cu130" \
    UV_BREAK_SYSTEM_PACKAGES=1 \
    UV_INDEX="https://flashinfer.ai/whl/ https://flashinfer.ai/whl/cu130/ https://download.pytorch.org/whl/cu130" \
    UV_INDEX_STRATEGY=unsafe-best-match \
    VLLM_TARGET_DEVICE=cuda \
    VLLM_USE_RUST_FRONTEND=1 \
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
      cuda-nvcc-13-3 cuda-cudart-dev-13-3 \
      libibverbs1 librdmacm1 libnuma1 libgomp1 \
      libucx0 python3 python3-pip libmimalloc-dev && \
    useradd --create-home --uid 10001 --shell /usr/sbin/nologin vllm

RUN python3 -m pip install --no-cache-dir --break-system-packages uv

COPY --from=builder /wheels /wheels
COPY --from=builder /wheels-flashinfer /wheels-flashinfer
COPY --from=builder /wheels-gguf /wheels-gguf
COPY --from=builder /wheels-b12x /wheels-b12x
COPY --from=builder /runtime-requirements /runtime-requirements
COPY --from=builder /src/vllm-build-commit /opt/vllm-build-commit
# Verbose cross-compile logs exceed the CI log limit mid-build, so persist
# ccache stats here: a low hit rate is the signal that keying regressed.
COPY --from=builder /src/ccache-stats-vllm.txt /src/ccache-stats-flashinfer.txt \
     /src/ccache-log-flashinfer.txt /opt/

# Resolve the local Python wheel's dependencies against the same indexes used by
# requirements/cuda.txt. The JIT-cache wheel has no dependencies of its own.
# The upstream cubin package is deliberately absent so it cannot shadow these
# TOPK=256-capable native artifacts.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system -r /runtime-requirements/cuda.txt && \
    uv pip install --system flashinfer-cubin==0.6.15.post1 && \
    uv pip install --system /wheels-flashinfer/flashinfer_python-*.whl && \
    uv pip install --system --no-deps \
      /wheels-flashinfer/flashinfer_jit_cache-*.whl && \
    uv pip install --system /wheels-b12x/*.whl && \
    uv pip install --system xxhash && \
    uv pip install --system --no-deps /wheels/*.whl && \
    if ls /wheels-gguf/*.whl >/dev/null 2>&1; then \
      uv pip install --system --no-deps /wheels-gguf/*.whl && \
      uv pip install --system 'gguf>=0.17.0' && \
      echo "GGUF plugin installed"; \
    else \
      echo "GGUF plugin wheel absent; skipping"; \
    fi

RUN rm -rf /wheels /wheels-flashinfer /wheels-gguf /wheels-b12x && \
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
      org.randomvariable.vllm.cuda="13.3.1" \
      org.randomvariable.vllm.distributed-executor-backend="ray" \
      org.opencontainers.image.revision="${VLLM_BUILD_COMMIT}" \
      org.opencontainers.image.version="${VLLM_IMAGE_TAG}" \
      org.opencontainers.image.url="${VLLM_BUILD_URL}" \
      ai.vllm.build.commit="${VLLM_BUILD_COMMIT}" \
      ai.vllm.build.pipeline="${VLLM_BUILD_PIPELINE}" \
      ai.vllm.build.url="${VLLM_BUILD_URL}" \
      ai.vllm.image.tag="${VLLM_IMAGE_TAG}"
ENV LD_PRELOAD="/usr/lib/aarch64-linux-gnu/libmimalloc.so.2"
USER vllm
WORKDIR /opt/vllm
ENTRYPOINT ["vllm"]
CMD ["serve"]

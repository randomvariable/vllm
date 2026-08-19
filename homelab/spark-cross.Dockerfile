# syntax=docker/dockerfile:1.7

ARG VLLM_BUILD_COMMIT=unknown
ARG VLLM_BUILD_PIPELINE=local
ARG VLLM_BUILD_URL=
ARG VLLM_IMAGE_TAG=local/vllm-spark-cross:dev
# SBSA cross-build concurrency + source identity, consumed by the
# vllm-runtime build pipeline (server6). Re-declared in the builder stage
# below so the values reach `make cross` (exported via ENV), the OCI labels
# resolve, and the wheel carries the setuptools-scm version.
ARG MAX_JOBS=4
ARG CMAKE_BUILD_PARALLEL_LEVEL=4
ARG NVCC_THREADS=1
ARG VLLM_SCM_VERSION=0.1.dev0

FROM --platform=linux/amd64 nvidia/cuda:13.3.1-devel-ubuntu26.04 AS builder

ARG VLLM_SCM_VERSION=0.1.dev0
ARG MAX_JOBS=4
ARG CMAKE_BUILD_PARALLEL_LEVEL=4
ARG NVCC_THREADS=1

# Nothing in this ENV may depend on the commit being built. VLLM_VERSION_OVERRIDE
# used to live here, and because setuptools-scm derives a new version for every
# commit, the layer changed on every build and invalidated everything below it --
# apt, the SBSA cross toolkit, uv, the aarch64 torch download, rustup and the
# whole `make cross` compile. Measured effect: exactly one CACHED layer per run.
# The version is passed to `make cross` inline instead, so only the compile layer
# is commit-sensitive.
#
# CCACHE_MAXSIZE is set here because the checked-in ccache.conf cannot carry it
# (ccache rejects storage options in a directory config) and nothing else set it,
# so the build ran at the 5 GiB default. FlashInfer AOT alone is ~3400 CUDA units
# of multi-MB objects, so the cache evicted its own output mid-stage and every
# stage re-entered effectively empty. Env beats the config file. 100G against the
# 400Gi PVC leaves room for the BuildKit layer store.
ENV DEBIAN_FRONTEND=noninteractive \
    PATH=/opt/venv/bin:/root/.local/bin:/root/.cargo/bin:${PATH} \
    VLLM_BUILD_TEMP=/vllm-build \
    MAX_JOBS=${MAX_JOBS} \
    CMAKE_BUILD_PARALLEL_LEVEL=${CMAKE_BUILD_PARALLEL_LEVEL} \
    NVCC_THREADS=${NVCC_THREADS} \
    CCACHE_DIR=/root/.cache/ccache \
    CCACHE_MAXSIZE=100G
COPY homelab/install-uv.sh /usr/local/bin/install-uv
COPY homelab/install-system-packages.sh /usr/local/bin/install-system-packages
RUN chmod 0755 /usr/local/bin/install-uv /usr/local/bin/install-system-packages

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    /usr/local/bin/install-system-packages \
      binutils-aarch64-linux-gnu ca-certificates ccache cmake curl file git \
      gcc g++ gcc-aarch64-linux-gnu g++-aarch64-linux-gnu libc6-dev-arm64-cross \
      libprotobuf-dev make ninja-build perl pkg-config protobuf-compiler \
      python3-dev python3-pip python3-venv unzip wget && \
    apt-get -y upgrade && \
    wget -q -O /tmp/cuda-keyring.deb \
      https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/cross-linux-sbsa/cuda-keyring_1.1-1_all.deb && \
    dpkg -i /tmp/cuda-keyring.deb && \
    apt-get update && apt-get install -y --no-install-recommends cuda-cross-sbsa-13-3 && \
    rm -f /tmp/cuda-keyring.deb && \
    ln -sfn /usr/local/cuda/targets/sbsa-linux /usr/local/cuda/targets/aarch64-linux && \
    for stub in /usr/local/cuda/targets/sbsa-linux/lib/stubs/*.so; do \
      ln -sfn "stubs/$(basename "$stub")" \
        "/usr/local/cuda/targets/sbsa-linux/lib/$(basename "$stub")"; \
    done

# Python's generic headers come from the uv-managed 3.12 in the builder image.
# Add only the arm64 pyconfig header required when the cross compiler defines
# __aarch64__. Ubuntu 26.04 ships no Python 3.12 (3.14 is default), so the arm64
# dev headers are pulled from the noble (24.04) archive; pyconfig.h is stable
# across 3.12 patch versions, so the pinned 3.12.3 header matches the uv 3.12.x
# used for the cross compile.
RUN set -eux; \
    curl -fsSL -o /tmp/libpython3.12-dev-arm64.deb \
      "https://ports.ubuntu.com/ubuntu-ports/pool/main/p/python3.12/libpython3.12-dev_3.12.3-1ubuntu0.16_arm64.deb"; \
    dpkg-deb -x /tmp/libpython3.12-dev-arm64.deb /tmp/pyarm64; \
    mkdir -p /usr/include/aarch64-linux-gnu; \
    cp -a /tmp/pyarm64/usr/include/aarch64-linux-gnu/python3.12 \
      /usr/include/aarch64-linux-gnu/python3.12; \
    test -f /usr/include/aarch64-linux-gnu/python3.12/pyconfig.h; \
    rm -rf /tmp/pyarm64 /tmp/libpython3.12-dev-arm64.deb

COPY requirements/build/cuda.txt requirements/build/rust.txt /tmp/build-requirements/
RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
    /usr/local/bin/install-uv && \
    uv venv --seed --python 3.12 /opt/venv && \
    uv pip install --python /opt/venv/bin/python \
      'torch==2.13.0' && \
    uv pip install --python /opt/venv/bin/python \
      -r /tmp/build-requirements/cuda.txt -r /tmp/build-requirements/rust.txt && \
    rm -rf /tmp/build-requirements

RUN curl --fail --silent --show-error --location \
      -o /tmp/torch-aarch64.whl \
      https://download.pytorch.org/whl/cu130/torch-2.13.0%2Bcu130-cp312-cp312-manylinux_2_28_aarch64.whl && \
    mkdir -p /opt/torch-aarch64 && \
    unzip -q /tmp/torch-aarch64.whl -d /opt/torch-aarch64 && \
    rm -f /tmp/torch-aarch64.whl

COPY homelab/sbsa-toolchain.cmake /opt/sbsa-toolchain.cmake

RUN --mount=type=cache,target=/root/.rustup,sharing=locked \
    --mount=type=cache,target=/root/.cargo/registry,sharing=locked \
    --mount=type=cache,target=/root/.cargo/git,sharing=locked \
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | \
      sh -s -- -y --default-toolchain none && \
    rustup toolchain install 1.95 && \
    rustup target add --toolchain 1.95 aarch64-unknown-linux-gnu
COPY . /src/vllm
RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
    --mount=type=cache,target=/root/.cargo/registry,sharing=locked \
    --mount=type=cache,target=/root/.cargo/git,sharing=locked \
    --mount=type=cache,target=/root/.rustup,sharing=locked \
    --mount=type=cache,id=vllm-spark-flashinfer-cubins-cross,target=/src/vllm/third_party/flashinfer/flashinfer-cubin/flashinfer_cubin/cubins,sharing=locked \
    --mount=type=cache,id=vllm-spark-ccache-cross,target=/root/.cache/ccache,sharing=locked \
    --mount=type=cache,target=/src/vllm/.deps,sharing=locked \
    mkdir -p /root/.cache/ccache && \
    cp /src/vllm/ccache.conf /root/.cache/ccache/ccache.conf && \
    echo "== ccache state at stage entry ==" && ccache -sv && ccache -z && \
    cd /src/vllm && \
    VLLM_VERSION_OVERRIDE=${VLLM_SCM_VERSION} make cross; \
    rc=$?; \
    echo "== ccache state at stage exit (rc=$rc) ==" && ccache -sv; \
    exit $rc

RUN mkdir -p /runtime-requirements
COPY requirements/cuda.txt /runtime-requirements/cuda.txt

FROM --platform=$TARGETPLATFORM nvidia/cuda:13.3.1-runtime-ubuntu26.04 AS runtime
ARG VLLM_BUILD_COMMIT
ARG VLLM_BUILD_PIPELINE
ARG VLLM_BUILD_URL
ARG VLLM_IMAGE_TAG

ENV DEBIAN_FRONTEND=noninteractive \
    PATH=/opt/venv/bin:/root/.local/bin:${PATH} \
    UV_PYTHON=3.12 \
    UV_PYTHON_INSTALL_DIR=/opt/python \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    VLLM_TARGET_DEVICE=cuda \
    VLLM_USE_RUST_FRONTEND=1 \
    TORCH_CUDA_ARCH_LIST=12.0f \
    FLASHINFER_CUDA_ARCH_LIST="12.0f 12.1a" \
    CUTE_DSL_ARCH=sm_121a \
    TIKTOKEN_ENCODINGS_BASE=/opt/vllm/tiktoken_encodings \
    HF_HOME=/opt/vllm/cache/huggingface \
    VLLM_CACHE_ROOT=/opt/vllm/cache \
    LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libmimalloc.so.3
COPY homelab/install-system-packages.sh /usr/local/bin/install-system-packages
RUN chmod 0755 /usr/local/bin/install-system-packages
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    /usr/local/bin/install-system-packages \
      ca-certificates curl gcc g++ git python3 python3-dev python3-pip \
      cuda-nvcc-13-3 cuda-cudart-dev-13-3 libgomp1 libibverbs1 libnuma1 \
      librdmacm1 libucx0 libmimalloc-dev ibverbs-providers && \
    useradd --create-home --uid 10001 --shell /usr/sbin/nologin vllm
COPY homelab/pip.conf /etc/pip.conf
COPY homelab/uv.toml /etc/uv/uv.toml
COPY --from=builder /runtime-requirements /runtime-requirements
COPY --from=builder /wheels /wheels
COPY --from=builder /wheels-b12x /wheels-b12x
COPY --from=builder /wheels-flashinfer /wheels-flashinfer
COPY homelab/install-uv.sh /usr/local/bin/install-uv
COPY homelab/install-runtime-wheels.sh /usr/local/bin/install-runtime-wheels
RUN chmod 0755 /usr/local/bin/install-runtime-wheels /usr/local/bin/install-uv && \
    /usr/local/bin/install-uv && \
    uv python install 3.12 && \
    uv venv --python 3.12 /opt/venv
RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
    /usr/local/bin/install-runtime-wheels
COPY homelab/setup-runtime.sh /usr/local/bin/setup-runtime
RUN chmod 0755 /usr/local/bin/setup-runtime && /usr/local/bin/setup-runtime
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
USER vllm
WORKDIR /opt/vllm
ENTRYPOINT ["vllm"]
CMD ["serve"]

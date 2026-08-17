# syntax=docker/dockerfile:1.7

ARG VLLM_BUILD_COMMIT=unknown
ARG VLLM_BUILD_PIPELINE=local
ARG VLLM_BUILD_URL=
ARG VLLM_IMAGE_TAG=local/vllm-spark-cross:dev

FROM --platform=linux/amd64 nvidia/cuda:13.3.1-devel-ubuntu26.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive \
    PATH=/opt/venv/bin:/root/.local/bin:/root/.cargo/bin:${PATH}
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

# Python's generic headers come from the builder image. Add only the arm64
# pyconfig header required when the cross compiler defines __aarch64__.
RUN set -eux; \
    ver="$(dpkg-query -W -f='${Version}' libpython3.12-dev)"; \
    curl -fsSL -o /tmp/libpython3.12-dev-arm64.deb \
      "https://ports.ubuntu.com/ubuntu-ports/pool/main/p/python3.12/libpython3.12-dev_${ver}_arm64.deb"; \
    dpkg-deb -x /tmp/libpython3.12-dev-arm64.deb /tmp/pyarm64; \
    mkdir -p /usr/include/aarch64-linux-gnu; \
    cp -a /tmp/pyarm64/usr/include/aarch64-linux-gnu/python3.12 \
      /usr/include/aarch64-linux-gnu/python3.12; \
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

COPY sbsa-toolchain.cmake /opt/sbsa-toolchain.cmake

RUN --mount=type=cache,target=/root/.rustup,sharing=locked \
    --mount=type=cache,target=/root/.cargo/registry,sharing=locked \
    --mount=type=cache,target=/root/.cargo/git,sharing=locked \
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | \
      sh -s -- -y --default-toolchain none && \
    rustup toolchain install 1.95 && \
    rustup target add --toolchain 1.95 aarch64-unknown-linux-gnu
COPY . /src/vllm
RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
    --mount=type=cache,target=/root/.cargo,sharing=locked \
    --mount=type=cache,target=/root/.rustup,sharing=locked \
    --mount=type=cache,id=vllm-spark-flashinfer-cubins-cross,target=/src/vllm/third_party/flashinfer/flashinfer-cubin/flashinfer_cubin/cubins,sharing=locked \
    --mount=type=cache,id=vllm-spark-ccache-cross,target=/root/.cache/ccache,sharing=locked \
    --mount=type=cache,target=/src/vllm/.deps,sharing=locked \
    mkdir -p /root/.cache/ccache && \
    cp /src/vllm/ccache.conf /root/.cache/ccache/ccache.conf && \
    cd /src/vllm && make cross

RUN mkdir -p /runtime-requirements
COPY requirements/cuda.txt /runtime-requirements/cuda.txt

FROM --platform=$TARGETPLATFORM nvidia/cuda:13.3.1-runtime-ubuntu26.04 AS runtime
ARG VLLM_BUILD_COMMIT
ARG VLLM_BUILD_PIPELINE
ARG VLLM_BUILD_URL
ARG VLLM_IMAGE_TAG

ENV DEBIAN_FRONTEND=noninteractive \
    PATH=/opt/venv/bin:/root/.local/bin:${PATH} \
    UV_PYTHON=3.13 \
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
    uv python install 3.13 && \
    uv venv --python 3.13 /opt/venv
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

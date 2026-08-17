# syntax=docker/dockerfile:1.7

ARG VLLM_BUILD_COMMIT=unknown
ARG VLLM_BUILD_PIPELINE=local
ARG VLLM_BUILD_URL=
ARG VLLM_IMAGE_TAG=local/vllm-spark:dev

FROM --platform=linux/arm64 nvidia/cuda:13.3.1-devel-ubuntu26.04 AS builder

WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive \
    PATH=/opt/venv/bin:/root/.local/bin:/root/.cargo/bin:${PATH} \
    VLLM_BUILD_TEMP=/vllm-build \
    UV_PYTHON=3.13 \
    PYTHON=/opt/venv/bin/python \
    FLASHINFER_CUDA_ARCH_LIST="12.0f 12.1a"
COPY ccache.conf /tmp/ccache.conf
COPY . .
COPY homelab/install-uv.sh /usr/local/bin/install-uv
COPY homelab/install-system-packages.sh /usr/local/bin/install-system-packages
COPY homelab/pip.conf /etc/pip.conf
COPY homelab/uv.toml /etc/uv/uv.toml
RUN chmod 0755 /usr/local/bin/install-uv /usr/local/bin/install-system-packages

RUN rm -f /etc/apt/apt.conf.d/docker-clean && \
    echo 'Binary::apt::APT::Keep-Downloaded-Packages "true";' \
      > /etc/apt/apt.conf.d/keep-cache

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    /usr/local/bin/install-system-packages --upgrade \
      ca-certificates ccache curl file git gcc g++ make ninja-build \
      pkg-config python3-dev python3-pip python3-venv wget && \
    /usr/local/bin/install-uv && \
    uv venv --seed --python 3.13 /opt/venv

RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
    --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    --mount=type=cache,target=/root/.cargo,sharing=locked \
    --mount=type=cache,target=/root/.rustup,sharing=locked \
    --mount=type=cache,target=/root/.cache/ccache,sharing=locked \
    --mount=type=cache,id=vllm-spark-flashinfer-cubins,target=/app/third_party/flashinfer/flashinfer-cubin/flashinfer_cubin/cubins,sharing=locked \
    --mount=type=cache,target=/app/.deps,sharing=locked \
    cp /tmp/ccache.conf /root/.cache/ccache/ccache.conf && \
    make sync && ccache --show-stats --verbose

FROM --platform=linux/arm64 nvidia/cuda:13.3.1-runtime-ubuntu26.04 AS runtime
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
    VLLM_CACHE_ROOT=/opt/vllm/cache
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
RUN chmod 0755 /usr/local/bin/install-runtime-wheels
RUN chmod 0755 /usr/local/bin/install-uv && \
    /usr/local/bin/install-uv && \
    uv python install 3.13 && \
    uv venv --python 3.13 /opt/venv
RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
    /usr/local/bin/install-runtime-wheels
COPY homelab/setup-runtime.sh /usr/local/bin/setup-runtime
RUN chmod 0755 /usr/local/bin/setup-runtime && /usr/local/bin/setup-runtime
LABEL org.opencontainers.image.title="vllm-spark-runtime" \
      org.opencontainers.image.description="Native vLLM runtime for DGX Spark sm_121a" \
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

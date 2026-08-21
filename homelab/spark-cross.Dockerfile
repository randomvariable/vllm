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

# PVC cache shuttle. Cache mounts are exec.cachemount refs whose lifetime is
# tied to buildkitd's session: a cancelled/killed/failed run orphans the refs
# and the next daemon start drops the contents (measured repeatedly -- ccache
# "Files: 0" at entry after a failed run, all 23k cubins re-downloaded). This
# stage restores both expensive mounts from plain tarballs on the build PVC
# before any compile runs. The pipeline binds the PVC shuttle dir as the
# shuttle-src context and exports the shuttle-flush stage back to it.
#
# The builder resumes FROM this stage (ordering edge: restore is sequenced
# before the AOT compile; a separate unconstrained stage could race it).
FROM --platform=linux/amd64 alpine:3.19 AS cache-prime
RUN --mount=type=bind,from=shuttle-src,target=/sh \
    --mount=type=cache,id=vllm-spark-ccache-cross,target=/ccache,sharing=locked \
    --mount=type=cache,id=vllm-spark-flashinfer-cubins-cross,target=/cubins,sharing=locked \
    sh -c 'set -e; \
      if [ -f /sh/ccache.tgz ]; then mkdir -p /ccache && tar -xzf /sh/ccache.tgz -C /ccache && echo "shuttle: restored ccache $(find /ccache -type f | wc -l) files"; else echo "shuttle: ccache cold"; fi; \
      if [ -f /sh/cubins.tgz ]; then mkdir -p /cubins && tar -xzf /sh/cubins.tgz -C /cubins && echo "shuttle: restored cubins $(find /cubins -type f | wc -l) files"; else echo "shuttle: cubins cold"; fi'

FROM --platform=linux/amd64 nvidia/cuda:13.3.1-devel-ubuntu26.04 AS builder

# VLLM_SCM_VERSION is deliberately NOT declared here. A build-arg's value is part
# of the cache key of every RUN after its ARG declaration in the same stage, even
# for RUNs that never reference it -- verified locally: changing only this arg's
# value, with no file touched, re-executed every RUN in the stage including a
# `RUN echo` that depends on nothing. Since setuptools-scm yields a new version
# per commit, declaring it up here would rebuild the toolchain and the FlashInfer
# AOT layer on every commit. It is declared immediately before the one RUN that
# consumes it instead.
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

# Ubuntu-derived images ship /etc/apt/apt.conf.d/docker-clean, whose
# DPkg::Post-Invoke hook deletes every .deb as soon as it is installed. With it
# in place the /var/cache/apt mount below can never hold anything, so every
# rebuild of the apt layer re-downloaded the entire package set -- including the
# ~544MB libcublas-cross-sbsa. Remove it and tell apt to keep its archives.
RUN rm -f /etc/apt/apt.conf.d/docker-clean && \
    printf '%s\n' 'Binary::apt::APT::Keep-Downloaded-Packages "true";' \
      > /etc/apt/apt.conf.d/keep-downloaded-packages

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

# The aarch64 torch wheel is ~2.5GB and was fetched to /tmp, so it was
# re-downloaded in full every time this layer rebuilt. Stage it in a download
# cache mount and only fetch when it is missing; the fetch is written to a temp
# name and renamed so an interrupted download is never mistaken for a complete
# one.
RUN --mount=type=cache,target=/downloads,sharing=locked \
    set -eux; \
    whl=/downloads/torch-2.13.0+cu130-cp312-cp312-manylinux_2_28_aarch64.whl; \
    if [ ! -s "$whl" ]; then \
      curl --fail --silent --show-error --location -o "$whl.part" \
        https://download.pytorch.org/whl/cu130/torch-2.13.0%2Bcu130-cp312-cp312-manylinux_2_28_aarch64.whl; \
      mv "$whl.part" "$whl"; \
    fi; \
    mkdir -p /opt/torch-aarch64; \
    unzip -q "$whl" -d /opt/torch-aarch64

COPY homelab/sbsa-toolchain.cmake /opt/sbsa-toolchain.cmake

RUN --mount=type=cache,target=/root/.rustup,sharing=locked \
    --mount=type=cache,target=/root/.cargo/registry,sharing=locked \
    --mount=type=cache,target=/root/.cargo/git,sharing=locked \
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | \
      sh -s -- -y --default-toolchain none && \
    rustup toolchain install 1.95 && \
    rustup target add --toolchain 1.95 aarch64-unknown-linux-gnu
# The FlashInfer AOT compile is ~3423 CUDA units and several hours, and it reads
# none of vLLM's Python source. Give it only the inputs it actually consumes so
# its layer is keyed on those alone: a source-only or version-only commit then
# reuses this layer instead of recompiling it. ccache cannot substitute for this
# -- measured on run z7hpr, the AOT dep-generation ran at 2.25s/unit with a warm
# cache directory present, because the generated build tree it keys on is
# recreated from scratch every build.
COPY Makefile common.mk ccache.conf pyproject.toml uv.lock /src/vllm/
COPY requirements /src/vllm/requirements
# Only flashinfer-build.sh is read here (verified: the script references no other
# repo file, and check_wheel_elf.py is reached from build-wheel, not
# build-flashinfer). Copying all of tools/ put unrelated scripts in this layer's
# key -- editing tools/build_deepgemm_C.py, which this stage never runs, threw
# away a completed 3423-unit AOT compile. The full tools/ tree still arrives
# with `COPY .` for the wheel stage below.
COPY tools/flashinfer-build.sh /src/vllm/tools/flashinfer-build.sh
COPY third_party/flashinfer /src/vllm/third_party/flashinfer
# Ordering edge for the cache shuttle: forces the cache-prime stage to be
# solved (i.e. the PVC tarballs restored into the mounts) before the AOT
# compile runs. buildkit only builds stages reachable from the target, so
# without this dependency cache-prime would never execute. Placed after the
# narrowed COPYs and immediately before the RUN it orders, so the shuttle
# tarball's per-build digest only keys this one layer, not the toolchain.
COPY --from=cache-prime /etc/alpine-release /tmp/.shuttle-primed
RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
    --mount=type=cache,id=vllm-spark-flashinfer-cubins-cross,target=/src/vllm/third_party/flashinfer/flashinfer-cubin/flashinfer_cubin/cubins,sharing=locked \
    --mount=type=cache,id=vllm-spark-ccache-cross,target=/root/.cache/ccache,sharing=locked \
    --mount=type=cache,target=/src/vllm/.deps,sharing=locked \
    mkdir -p /root/.cache/ccache && \
    cp /src/vllm/ccache.conf /root/.cache/ccache/ccache.conf && \
    echo "== ccache at flashinfer entry ==" && ccache -sv && ccache -z && \
    cd /src/vllm && make cross-flashinfer; \
    rc=$?; \
    echo "== ccache at flashinfer exit (rc=$rc) ==" && ccache -sv; \
    mkdir -p /shuttle-out && \
    tar -czf /shuttle-out/ccache.tgz -C /root/.cache/ccache . && \
    tar -czf /shuttle-out/cubins.tgz -C /src/vllm/third_party/flashinfer/flashinfer-cubin/flashinfer_cubin/cubins . && \
    echo "shuttle: flushed $(du -sh /shuttle-out | cut -f1)"; \
    exit $rc

# Everything that genuinely depends on the full source and the release version:
# the b12x wheel, the rust frontend and the vLLM wheel itself.
#
# `COPY .` puts every Python file in this layer's cache key, so a Python-only
# commit re-runs the whole compile. VLLM_BUILD_TEMP (/vllm-build) is the CMake
# binary dir, and without a mount it lives in the layer, so the re-run starts
# from a parent state that has none: ninja rebuilds all ~410 objects, including
# the vllm-flash-attn FA2 kernel matrix, and only ccache stands between that and
# a multi-hour stage. ccache is itself a mutable cache ref, which a cancelled or
# evicted run loses at the next buildkitd start, so that floor is not reliable.
# Mounting the binary dir lets ninja see the objects as up-to-date and skip them
# outright, which is both faster than a ccache hit and independent of the layer
# cache key. CMake regenerates torch_patched_headers inside it, so nothing from
# an earlier layer is masked.
COPY . /src/vllm
# Declared here, after the FlashInfer AOT layer, so its per-commit value only
# participates in the cache key of the wheel layer below.
ARG VLLM_SCM_VERSION=0.1.dev0
RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
    --mount=type=cache,target=/root/.cargo/registry,sharing=locked \
    --mount=type=cache,target=/root/.cargo/git,sharing=locked \
    --mount=type=cache,target=/root/.rustup,sharing=locked \
    --mount=type=cache,id=vllm-spark-ccache-cross,target=/root/.cache/ccache,sharing=locked \
    --mount=type=cache,target=/src/vllm/.deps,sharing=locked \
    --mount=type=cache,id=vllm-spark-cmake-cross,target=/vllm-build,sharing=locked \
    echo "== ccache at wheel entry ==" && ccache -sv && ccache -z && \
    cd /src/vllm && \
    VLLM_VERSION_OVERRIDE=${VLLM_SCM_VERSION} make cross-rest; \
    rc=$?; \
    echo "== ccache at wheel exit (rc=$rc) ==" && ccache -sv; \
    mkdir -p /shuttle-out && tar -czf /shuttle-out/ccache.tgz -C /root/.cache/ccache . && \
    echo "shuttle: flushed wheel ccache $(du -sh /shuttle-out/ccache.tgz | cut -f1)"; \
    exit $rc

RUN mkdir -p /runtime-requirements
COPY requirements/cuda.txt /runtime-requirements/cuda.txt

# Shuttle export: plain files on the build PVC via --output type=local in the
# pipeline. Not part of the final image; exists solely to carry the tarballs
# out of buildkit's ref lifecycle between runs.
FROM scratch AS shuttle-flush
COPY --from=builder /shuttle-out/ /shuttle/


FROM --platform=$TARGETPLATFORM nvidia/cuda:13.3.1-runtime-ubuntu26.04 AS runtime
# The per-commit ARGs (build commit, image tag, pipeline URL) are declared just
# above the LABEL that consumes them, not here: a build-arg value joins the cache
# key of every RUN after its declaration in the stage, so declaring them at the
# top invalidated this stage's qemu-emulated apt install on every single commit.

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
# Same docker-clean removal as the builder stage. This one matters more: the
# runtime stage runs under qemu-aarch64, so re-fetching and re-installing these
# packages is emulated work.
RUN rm -f /etc/apt/apt.conf.d/docker-clean && \
    printf '%s\n' 'Binary::apt::APT::Keep-Downloaded-Packages "true";' \
      > /etc/apt/apt.conf.d/keep-downloaded-packages
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
ARG VLLM_BUILD_COMMIT
ARG VLLM_BUILD_PIPELINE
ARG VLLM_BUILD_URL
ARG VLLM_IMAGE_TAG
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

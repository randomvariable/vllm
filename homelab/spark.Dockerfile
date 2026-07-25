# syntax=docker/dockerfile:1.7

ARG CUDA_TAG=13.0.2
ARG UBUNTU_TAG=ubuntu24.04
ARG VLLM_BRANCH=homelabs-main
# Optional source pin verified after COPY (the Tekton harness passes the trigger
# revision). Empty by default: build from whatever the build context is checked
# out to, since this Dockerfile lives in the vllm fork itself.
ARG VLLM_COMMIT=

FROM nvidia/cuda:${CUDA_TAG}-devel-${UBUNTU_TAG} AS builder
ARG VLLM_BRANCH
ARG VLLM_COMMIT
ENV DEBIAN_FRONTEND=noninteractive \
    MAX_JOBS=6 \
    CMAKE_BUILD_PARALLEL_LEVEL=6 \
    NVCC_THREADS=3 \
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
# the source -- copy it instead of cloning. .git comes along so setuptools-scm
# can derive the version. When VLLM_COMMIT is supplied (the Tekton harness
# passes the trigger revision) the checked-out source is verified against it.
COPY . /src/vllm
RUN cd /src/vllm && \
    git rev-parse HEAD > /src/vllm-build-commit && \
    if [ -n "$VLLM_COMMIT" ]; then \
      test "$(cat /src/vllm-build-commit)" = "$VLLM_COMMIT"; \
    fi

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
    sed -E '/^[[:space:]]*flashinfer-(python|cubin)==/d' requirements/build/cuda.txt > requirements/build/cuda-build.txt && \
    /opt/venv/bin/pip install -r requirements/build/cuda-build.txt --extra-index-url https://download.pytorch.org/whl/cu130 && \
    ccache --version | head -1 && ccache -z && \
    /opt/venv/bin/pip wheel -v --no-build-isolation --no-deps --wheel-dir /wheels . && \
    ccache -s && \
    python3 -c "import glob, sys, zipfile; whl = glob.glob('/wheels/vllm-*.whl')[0]; names = zipfile.ZipFile(whl).namelist(); sys.exit(0) if any(n.endswith('vllm/vllm-rs') for n in names) else (print('FATAL: vllm-rs binary missing from wheel -- rust build did not bundle', file=sys.stderr), sys.exit(1))"

# Replace only stale FlashInfer metadata; keep the fork's other requirements intact.
RUN python3 - <<'PYCODE'
import base64, hashlib, pathlib, re, tempfile, zipfile
wheel = next(pathlib.Path('/wheels').glob('vllm-*.whl'))
with zipfile.ZipFile(wheel) as source:
    files = {name: source.read(name) for name in source.namelist()}
metadata_name = next(name for name in files if name.endswith('.dist-info/METADATA'))
metadata = files[metadata_name].decode()
metadata, count = re.subn(r'(?m)^(Requires-Dist: flashinfer-(?:python|cubin))==[0-9][^\n]*$', r'\1>=0.6.15.post1', metadata)
if count < 1:
    raise SystemExit(f'expected at least one stale FlashInfer requirement (flashinfer-python; flashinfer-cubin is excluded from install_requires since 0.6.14 per requirements/cuda.txt), found {count}')
files[metadata_name] = metadata.encode()
record_name = next(name for name in files if name.endswith('.dist-info/RECORD'))
records = []
for line in files[record_name].decode().splitlines():
    name = line.split(',', 1)[0]
    if name == metadata_name:
        digest = base64.urlsafe_b64encode(hashlib.sha256(files[name]).digest()).rstrip(b'=').decode()
        line = f'{name},sha256={digest},{len(files[name])}'
    records.append(line)
files[record_name] = ('\n'.join(records) + '\n').encode()
with tempfile.NamedTemporaryFile(dir=wheel.parent, suffix='.whl', delete=False) as output:
    replacement = pathlib.Path(output.name)
with zipfile.ZipFile(replacement, 'w', zipfile.ZIP_DEFLATED) as target:
    for name, content in files.items():
        target.writestr(name, content)
replacement.replace(wheel)
PYCODE

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
ARG VLLM_BRANCH
ARG VLLM_COMMIT
ENV DEBIAN_FRONTEND=noninteractive \
    UV_BREAK_SYSTEM_PACKAGES=1 \
    VLLM_TARGET_DEVICE=cuda \
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
COPY --from=builder /runtime-requirements /runtime-requirements
COPY --from=builder /src/vllm-build-commit /opt/vllm-build-commit

RUN --mount=type=cache,target=/root/.cache/uv \
    sed -i -E '/^[[:space:]]*flashinfer-(python|cubin)==/d' /runtime-requirements/cuda.txt && \
    uv pip install --system \
    -r /runtime-requirements/cuda.txt \
    'flashinfer-python==0.6.15.post1' \
    'flashinfer-cubin==0.6.15.post1' \
    'flashinfer-jit-cache==0.6.15.post1+cu130' \
    'nixl>=1.3.1' 'nixl-cu13>=1.3.1' \
    'ray[default]' 'tiktoken>=0.9.0' \
    --extra-index-url https://download.pytorch.org/whl/cu130 \
    --extra-index-url https://flashinfer.ai/whl/ \
    --extra-index-url https://flashinfer.ai/whl/cu130/ \
    --index-strategy unsafe-best-match && \
    uv pip install --system --no-deps /wheels/*.whl

# cutlass-dsl sm_121a arch guard: NOT applied. eugr's spark-vllm-docker sed
# patches (run.sh 55-83) target CuTe DSL Python source (warp/mma.py,
# tcgen05/mma.py, tcgen05/copy.py) present in nvidia-cutlass-dsl 4.4.2. In the
# pinned 4.6.0 (requirements/cuda.txt), those modules ship compiled into
# libcute_dsl_runtime.so / _cutlass_ir.*.so inside nvidia-cutlass-dsl-libs-*;
# no `.py` file contains `if not arch == Arch.sm_120a:`, `admissible_archs`, or
# `is_family_of(Arch.sm_110f)`. There is nothing to sed, so the patch is a
# no-op here and is intentionally omitted (arch selection is instead forced
# via CUTE_DSL_ARCH=sm_121a in the ENV block above -- validated on-box:
# FlashInferB12xExperts grouped GEMM matches BF16 reference with this env var
# set and cutlass-dsl 4.6.0 unmodified. The pod manifest may still set it
# explicitly for clarity but the image default is now load-bearing, not a
# placeholder.)

# NVFP4 profiler workspace fix (FlashInfer PR #3738 regression): native SM100+
# NVFP4 MoE uses FP4 activations AND FP4 weights, but PR #3738 narrowed the
# profiler workspace allocation to the FP8-activation family, so autotune
# allocates null quant workspaces and fails in prepareQuantParams(). This only
# affects the TRT-LLM grouped-GEMM backend (FLASHINFER_CUTLASS), which is
# excluded from auto-selection (oracle/nvfp4.py) and unreachable via our
# production --moe-backend flashinfer_b12x flag -- b12x is a separate CuteDSL
# kernel (flashinfer.fused_moe.b12x_fused_moe) that never touches this file's
# GemmProfilerBackend. Confirmed on-box: flashinfer-python==0.6.15.post1 does
# not even ship the isNativeWfp4Afp8Family predicate this patch targets, so
# the pattern-not-found case below is expected and must be a no-op, not a
# build failure. Retained only as forward-looking defense-in-depth in case a
# future flashinfer release reintroduces the unfixed predicate. Ported
# verbatim from eugr's spark-vllm-docker Dockerfile (215-259); target path is
# self-located under the installed flashinfer package.
RUN python3 - <<'PY'
import flashinfer, pathlib

pkg_dir = pathlib.Path(flashinfer.__file__).parent
target = pkg_dir / "data" / "csrc" / "fused_moe" / "cutlass_backend" / "cutlass_fused_moe_kernels.cuh"
old_predicate = (
    "  bool const is_native_wfp4afp8_family = isNativeWfp4Afp8Family();\n"
)
fixed_predicates = """  bool const is_native_wfp4afp8_family = isNativeWfp4Afp8Family();
  // Native Blackwell NVFP4 uses FP4 activations and FP4 weights.
  bool const is_native_wfp4afp4_family =
      mSM >= 100 &&
      (mDType == nvinfer1::DataType::kFP4 || mDType == nvinfer1::DataType::kINT64) &&
      (mWType == nvinfer1::DataType::kFP4 || mWType == nvinfer1::DataType::kINT64);
"""
old_branch = "  if (is_native_wfp4afp8_family) {"
fixed_branch = (
    "  if (is_native_wfp4afp8_family || is_native_wfp4afp4_family) {"
)

if not target.exists():
    raise SystemExit(f"{target} not found; cannot apply NVFP4 profiler patch")

text = target.read_text()
already_fixed = fixed_predicates in text and fixed_branch in text
if already_fixed:
    print("FlashInfer native NVFP4 profiler workaround already present; skipping")
else:
    if text.count(old_predicate) != 1 or text.count(old_branch) != 1:
        # b12x (our production MoE backend) never touches this file. Newer
        # flashinfer releases (validated: 0.6.15.post1) have already dropped
        # or rewritten this predicate outside our target pattern -- skip
        # gracefully rather than fail the build over a defense-in-depth patch
        # for an unreachable code path.
        print(
            "FlashInfer PR #3738 profiler pattern not found (predicate="
            f"{text.count(old_predicate)}, branch={text.count(old_branch)}); "
            "not applicable to this flashinfer version, skipping patch"
        )
    else:
        text = text.replace(old_predicate, fixed_predicates, 1)
        text = text.replace(old_branch, fixed_branch, 1)
        target.write_text(text)
        print("Applied FlashInfer native NVFP4 profiler workspace workaround")
        patched = target.read_text()
        if fixed_predicates not in patched or fixed_branch not in patched:
            raise SystemExit("FlashInfer native NVFP4 profiler patch verification failed")
PY
RUN rm -rf /wheels

# No source patches are applied here. UMA wedge protection (PR #46932
# negative-cudagraph clamp) is compiled into the wheel from the fork's SM121 +
# UMA-clamp commits (homelabs-main). instanttensor is deliberately excluded: it
# causes hard power-cycle-requiring lockups on DGX Spark GB10 UMA
# (NV_ERR_NO_MEMORY during CUDA graph compile); fastsafetensors is the
# GB10-safe loader.
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
      org.randomvariable.vllm.source-branch="homelabs-main" \
      org.randomvariable.vllm.source-commit="a7617c3e0ea7" \
      org.randomvariable.vllm.patch-policy="UMA clamp (PR #46932) compiled into fork source; no runtime patch" \
      org.randomvariable.vllm.distributed-executor-backend="ray"

USER vllm
WORKDIR /opt/vllm
ENTRYPOINT ["vllm"]
CMD ["serve"]

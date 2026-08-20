ROOT_DIR_RELATIVE := .
include $(ROOT_DIR_RELATIVE)/common.mk

.EXPORT_ALL_VARIABLES:

export PS1 :=
DOCKER_BUILD := docker buildx build
UV_LINK_MODE ?= copy
UV_TORCH_BACKEND ?= cu130
VLLM_TARGET_DEVICE ?= cuda
VERBOSE ?= 1
CMAKE_VERBOSE_MAKEFILE ?= ON
TORCH_CUDA_ARCH_LIST ?= 12.0f
MAX_JOBS ?= 18
NVCC_THREADS ?= 1
VLLM_BUILD_TEMP ?= /vllm-build
CUDA_HOME ?= /usr/local/cuda
CUDA_TOOLKIT_ROOT ?= /usr/local/cuda
UV_PYTHON ?= 3.13
PYTHON ?= $(UV_PYTHON)
FLASHINFER_MAX_JOBS ?= $(MAX_JOBS)
FLASHINFER_NVCC_THREADS ?= $(NVCC_THREADS)
FLASHINFER_CUDA_ARCH_LIST ?= 12.0f 12.1a
VLLM_DISABLE_SCCACHE ?= 1
USE_CUDNN ?= 1
USE_CUSPARSELT ?= 1
USE_CUDSS ?= 1
USE_CUFILE ?= 1
VLLM_VERSION_OVERRIDE ?= 0.23.0
PYTHON_HOST_PLATFORM ?=
WHEEL_BUILD_ENV ?=
WHEEL_EXPECTED_ELF_MACHINE ?=
CUDA_VERSION ?= 13.3
WHEEL_DIR ?= /wheels
B12X_WHEEL_DIR ?= /wheels-b12x
FLASHINFER_DIST_DIR ?= /wheels-flashinfer
CCACHE_EXTRAFILES ?= /root/.cache/ccache/keyfile
FLASHINFER_JIT_CACHE_LOCAL_VERSION ?= cu130
FLASHINFER_NVCC_LAUNCHER ?= ccache
FLASHINFER_CXX_LAUNCHER ?= ccache
FLASHINFER_NVCC ?= nvcc
FLASHINFER_WHEEL_PLATFORM_TAG ?=
FLASHINFER_FMHA_V2_HOST_BUILD ?= 1
FLASHINFER_FMHA_V2_HOST_CXX ?= g++
FLASHINFER_EXTRA_LDFLAGS ?=
CMAKE_JOB_POOLS ?= compile=5
RUST_TOOLCHAIN ?= 1.95
UV ?= uv

# Cross compilation keeps all target-specific paths, toolchain flags, and
# memory-sensitive concurrency in one place instead of duplicating them in the
# Dockerfile. Concurrency uses ?= so an exported MAX_JOBS /
# CMAKE_BUILD_PARALLEL_LEVEL / NVCC_THREADS (e.g. from the spark-cross
# Dockerfile build-arg ENV) can raise it, defaulting to the memory-safe 4/4/1.
#
# Make propagates target-specific variables to prerequisites, which is how this
# block reaches build-flashinfer/build-wheel. It is therefore attached to one
# shared target list rather than to `cross` alone: the image build invokes the
# split entry points below as separate layers, and any setting that failed to
# reach them -- CXX or NVCC_PREPEND_FLAGS above all -- would silently produce a
# host-arch link instead of an aarch64 one.
CROSS_TARGETS := cross cross-flashinfer cross-rest
$(CROSS_TARGETS): MAX_JOBS?=4
$(CROSS_TARGETS): CMAKE_BUILD_PARALLEL_LEVEL?=4
$(CROSS_TARGETS): NVCC_THREADS?=1
$(CROSS_TARGETS): FLASHINFER_MAX_JOBS?=2
$(CROSS_TARGETS): FLASHINFER_NVCC_THREADS?=1
$(CROSS_TARGETS): CUDA_HOME=/usr/local/cuda
$(CROSS_TARGETS): CUDA_TOOLKIT_ROOT=/usr/local/cuda
$(CROSS_TARGETS): DEEPGEMM_CXX=/usr/bin/aarch64-linux-gnu-g++
# FlashInfer's AOT .so link (flashinfer/jit/cpp_ext.py) reads CXX and defaults
# to host c++, which links the aarch64 nvcc objects with the x86_64 linker
# ("file in wrong format"). Point it at the cross compiler.
$(CROSS_TARGETS): CXX=/usr/bin/aarch64-linux-gnu-g++
$(CROSS_TARGETS): DEEPGEMM_EXT_SUFFIX=.cpython-312-aarch64-linux-gnu.so
$(CROSS_TARGETS): DEEPGEMM_TORCH_ROOT=/opt/torch-aarch64/torch
$(CROSS_TARGETS): DEEPGEMM_CUDA_LIB_DIR=/usr/local/cuda/targets/sbsa-linux/lib
$(CROSS_TARGETS): DEEPGEMM_TORCH_CXX11_ABI=1
$(CROSS_TARGETS): DEEPGEMM_SRC_DIR=$(CURDIR)/third_party/deep_gemm
$(CROSS_TARGETS): VLLM_TARGET_DEVICE=cuda
$(CROSS_TARGETS): TORCH_CUDA_ARCH_LIST=12.0f
$(CROSS_TARGETS): NVCC_PREPEND_FLAGS=-target-dir sbsa-linux -ccbin /usr/bin/aarch64-linux-gnu-g++
$(CROSS_TARGETS): CMAKE_ARGS=-DCMAKE_TOOLCHAIN_FILE=/opt/sbsa-toolchain.cmake -DTorch_DIR=/opt/torch-aarch64/torch/share/cmake/Torch -DCUDAToolkit_ROOT=/usr/local/cuda -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda -DCUDA_CUDART=/usr/local/cuda/targets/sbsa-linux/lib/libcudart.so -DVLLM_CUBLAS_LIBRARY=/usr/local/cuda/targets/sbsa-linux/lib/libcublas.so -DVLLM_CUTLASS_SRC_DIR=$(CURDIR)/third_party/flashinfer/3rdparty/cutlass
$(CROSS_TARGETS): CARGO_BUILD_TARGET=aarch64-unknown-linux-gnu
$(CROSS_TARGETS): CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER=aarch64-linux-gnu-gcc
$(CROSS_TARGETS): CC_aarch64_unknown_linux_gnu=aarch64-linux-gnu-gcc
$(CROSS_TARGETS): CXX_aarch64_unknown_linux_gnu=aarch64-linux-gnu-g++
$(CROSS_TARGETS): AR_aarch64_unknown_linux_gnu=aarch64-linux-gnu-ar
$(CROSS_TARGETS): PYO3_CROSS_PYTHON_VERSION=3.12
$(CROSS_TARGETS): PYTHON=/opt/venv/bin/python
$(CROSS_TARGETS): PYTHON_HOST_PLATFORM=linux-aarch64
$(CROSS_TARGETS): WHEEL_EXPECTED_ELF_MACHINE=AArch64
$(CROSS_TARGETS): WHEEL_BUILD_ENV=_PYTHON_HOST_PLATFORM=$(PYTHON_HOST_PLATFORM) CXX=$(CXX) CUDA_HOME=$(CUDA_HOME) CUDA_TOOLKIT_ROOT=$(CUDA_TOOLKIT_ROOT) VLLM_TARGET_DEVICE=$(VLLM_TARGET_DEVICE) TORCH_CUDA_ARCH_LIST=$(TORCH_CUDA_ARCH_LIST) NVCC_PREPEND_FLAGS='$(NVCC_PREPEND_FLAGS)' CMAKE_ARGS='$(CMAKE_ARGS)'
$(CROSS_TARGETS): WHEEL_DIR=/wheels
$(CROSS_TARGETS): B12X_WHEEL_DIR=/wheels-b12x
$(CROSS_TARGETS): FLASHINFER_DIST_DIR=/wheels-flashinfer
$(CROSS_TARGETS): FLASHINFER_WHEEL_PLATFORM_TAG=manylinux_2_28_aarch64
$(CROSS_TARGETS): FLASHINFER_EXPECTED_ELF_MACHINE=AArch64
$(CROSS_TARGETS): FLASHINFER_CUDA_ARCH_LIST=12.0f 12.1a
$(CROSS_TARGETS): FLASHINFER_EXTRA_LDFLAGS=-L/usr/local/cuda/targets/sbsa-linux/lib -L/usr/local/cuda/targets/sbsa-linux/lib/stubs -Wl,-rpath-link,/usr/local/cuda/targets/sbsa-linux/lib
cross: sync

# Split halves of `sync`, for the staged image build. The FlashInfer AOT objects
# depend on third_party/flashinfer, the build script and the installed build deps
# -- never on vLLM's Python source or the release version -- so this half can be
# reused across source commits as a cached layer.
cross-flashinfer: build-flashinfer

cross-rest: sync-b12x build-rust build-wheel

strix: VLLM_TARGET_DEVICE=rocm
strix: MAX_JOBS=8
strix: CMAKE_BUILD_PARALLEL_LEVEL=8
strix: PYTORCH_ROCM_ARCH=gfx1151
strix: PYTHON=/opt/venv/bin/python
strix: sync-strix

sync-strix: sync-strix-deps build-rust build-strix-wheel

sync-strix-deps:
	$(UV) pip install -r requirements/build/rocm.txt -r requirements/build/rust.txt
	$(UV) pip install -r requirements/rocm.txt

build-strix-wheel: ccache-keyfile
	mkdir -p /wheels
	_PYTHON_HOST_PLATFORM= $(UV) build --python $(PYTHON) --no-build-isolation --wheel \
		--out-dir /wheels .

sync-deps: | .logs
	mkdir -p /runtime-requirements
	cp requirements/common.txt /runtime-requirements/common.txt
	cp requirements/cuda.txt /runtime-requirements/cuda.txt
	UV_LINK_MODE=$(UV_LINK_MODE) $(UV) sync --frozen --no-install-project -v 2>&1 | tee .logs/gb10.log
	UV_LINK_MODE=$(UV_LINK_MODE) $(UV) pip -v install --python $(PYTHON) -r requirements/build/cuda.txt -r requirements/build/rust.txt
	UV_LINK_MODE=$(UV_LINK_MODE) $(UV) pip -v install --python $(PYTHON) setuptools wheel
	UV_LINK_MODE=$(UV_LINK_MODE) $(UV) pip -v install --python $(PYTHON) -r requirements/cuda.txt

sync: sync-deps sync-b12x build-rust build-flashinfer build-wheel ## install all Spark build dependencies and vLLM

sync-b12x:
	mkdir -p $(B12X_WHEEL_DIR)
	$(UV) build --wheel --out-dir $(B12X_WHEEL_DIR) ./third_party/b12x

# ccache.conf sets `extra_files` to this path unconditionally, and ccache treats
# an unreadable extra file as an error that bypasses the cache -- "Error hashing
# extra file", counted as neither hit nor miss. It exists because ccache does not
# hash NVCC_PREPEND_FLAGS, so without it a cross build could reuse host-arch
# objects (see also CCACHE_EXTRAFILES in the cross env). Every compiling target
# must therefore write it, not just the FlashInfer one: the FlashInfer layer is
# usually restored from the layer cache and never re-runs, so a wheel stage that
# relied on it left the file absent and compiled all ~412 objects uncached.
ccache-keyfile:
	mkdir -p "$(dir $(CCACHE_EXTRAFILES))"
	printf '%s\n' "$(NVCC_PREPEND_FLAGS)" "$(TORCH_CUDA_ARCH_LIST)" > "$(CCACHE_EXTRAFILES)"

build-flashinfer: sync-deps ccache-keyfile
	mkdir -p $(FLASHINFER_DIST_DIR)
	CUDA_VERSION=$(CUDA_VERSION) \
	FLASHINFER_SOURCE_DIR=$(CURDIR)/third_party/flashinfer \
	FLASHINFER_DIST_DIR=$(FLASHINFER_DIST_DIR) \
	FLASHINFER_CUDA_ARCH_LIST="$(FLASHINFER_CUDA_ARCH_LIST)" \
	FLASHINFER_JIT_CACHE_LOCAL_VERSION=$(FLASHINFER_JIT_CACHE_LOCAL_VERSION) \
	FLASHINFER_NVCC_LAUNCHER=$(FLASHINFER_NVCC_LAUNCHER) \
	FLASHINFER_CXX_LAUNCHER=$(FLASHINFER_CXX_LAUNCHER) \
	NVCC_PREPEND_FLAGS="$(NVCC_PREPEND_FLAGS)" \
	FLASHINFER_EXTRA_LDFLAGS="$(FLASHINFER_EXTRA_LDFLAGS)" \
	FLASHINFER_EXPECTED_ELF_MACHINE="$(FLASHINFER_EXPECTED_ELF_MACHINE)" \
	FLASHINFER_WHEEL_PLATFORM_TAG=$(FLASHINFER_WHEEL_PLATFORM_TAG) \
	FLASHINFER_FMHA_V2_HOST_BUILD=$(FLASHINFER_FMHA_V2_HOST_BUILD) \
	FLASHINFER_FMHA_V2_HOST_CXX=$(FLASHINFER_FMHA_V2_HOST_CXX) \
	FLASHINFER_NVCC=$(FLASHINFER_NVCC) \
	MAX_JOBS=$(FLASHINFER_MAX_JOBS) FLASHINFER_NVCC_THREADS=$(FLASHINFER_NVCC_THREADS) \
	BUILD_JIT_CACHE=true BUILD_NVEP=0 \
	./tools/flashinfer-build.sh && \
	$(UV) build --python $(PYTHON) --no-build-isolation --wheel \
		--out-dir $(FLASHINFER_DIST_DIR) ./third_party/flashinfer/flashinfer-cubin

build-rust:
	if ! command -v rustup >/dev/null 2>&1; then \
		wget -qO- https://sh.rustup.rs | sh -s -- -y --default-toolchain none; \
	fi
	export PATH="$(HOME)/.cargo/bin:$$PATH"
	rustup toolchain install $(RUST_TOOLCHAIN)
	rustup default $(RUST_TOOLCHAIN)

build-wheel: ccache-keyfile
	mkdir -p $(WHEEL_DIR)
	$(WHEEL_BUILD_ENV) $(UV) build --python $(PYTHON) --no-build-isolation --wheel \
		--out-dir $(WHEEL_DIR) .
	if [ -n "$(WHEEL_EXPECTED_ELF_MACHINE)" ]; then \
		set -e; \
		for whl in $(WHEEL_DIR)/vllm-*.whl; do \
			$(PYTHON) tools/check_wheel_elf.py "$$whl" "$(WHEEL_EXPECTED_ELF_MACHINE)"; \
		done; \
	fi

SPARK_IMAGE ?= local/vllm-spark:dev
SPARK_CROSS_IMAGE ?= local/vllm-spark-cross:dev
SPARK_CROSS_PLATFORM ?= linux/arm64
SPARK_CROSS_MAX_JOBS ?= 4
SPARK_CROSS_CMAKE_BUILD_PARALLEL_LEVEL ?= 4
SPARK_CROSS_NVCC_THREADS ?= 1

spark-docker: | .logs
	$(DOCKER_BUILD) --load -t $(SPARK_IMAGE) -f homelab/spark.Dockerfile . 2>&1 | tee .logs/spark-docker.log

spark-cross-docker: | .logs
	$(DOCKER_BUILD) --load --platform $(SPARK_CROSS_PLATFORM) \
		--build-arg MAX_JOBS=$(SPARK_CROSS_MAX_JOBS) \
		--build-arg CMAKE_BUILD_PARALLEL_LEVEL=$(SPARK_CROSS_CMAKE_BUILD_PARALLEL_LEVEL) \
		--build-arg NVCC_THREADS=$(SPARK_CROSS_NVCC_THREADS) \
		-t $(SPARK_CROSS_IMAGE) -f homelab/spark-cross.Dockerfile . \
		2>&1 | tee .logs/spark-cross-docker.log

strix-docker: | .logs
	$(DOCKER_BUILD) --load -t $(STRIX_IMAGE) -f homelab/strix.Dockerfile . 2>&1 | tee .logs/strix-docker.log

.deps:
	mkdir -p .deps

.logs:
	mkdir -p .logs

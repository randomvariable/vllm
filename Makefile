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
cross: MAX_JOBS?=4
cross: CMAKE_BUILD_PARALLEL_LEVEL?=4
cross: NVCC_THREADS?=1
cross: FLASHINFER_MAX_JOBS?=2
cross: FLASHINFER_NVCC_THREADS?=1
cross: CUDA_HOME=/usr/local/cuda
cross: CUDA_TOOLKIT_ROOT=/usr/local/cuda
cross: DEEPGEMM_CXX=/usr/bin/aarch64-linux-gnu-g++
# FlashInfer's AOT .so link (flashinfer/jit/cpp_ext.py) reads CXX and defaults
# to host c++, which links the aarch64 nvcc objects with the x86_64 linker
# ("file in wrong format"). Point it at the cross compiler.
cross: CXX=/usr/bin/aarch64-linux-gnu-g++
cross: DEEPGEMM_PYTHON_INCLUDE=/usr/include/python3.12
cross: DEEPGEMM_EXT_SUFFIX=.cpython-312-aarch64-linux-gnu.so
cross: DEEPGEMM_TORCH_ROOT=/opt/torch-aarch64/torch
cross: DEEPGEMM_CUDA_LIB_DIR=/usr/local/cuda/targets/sbsa-linux/lib
cross: DEEPGEMM_TORCH_CXX11_ABI=1
cross: DEEPGEMM_SRC_DIR=$(CURDIR)/third_party/deep_gemm
cross: VLLM_TARGET_DEVICE=cuda
cross: TORCH_CUDA_ARCH_LIST=12.0f
cross: NVCC_PREPEND_FLAGS=-target-dir sbsa-linux -ccbin /usr/bin/aarch64-linux-gnu-g++
cross: CMAKE_ARGS=-DCMAKE_TOOLCHAIN_FILE=/opt/sbsa-toolchain.cmake -DTorch_DIR=/opt/torch-aarch64/torch/share/cmake/Torch -DCUDAToolkit_ROOT=/usr/local/cuda -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda -DCUDA_CUDART=/usr/local/cuda/targets/sbsa-linux/lib/libcudart.so -DVLLM_CUBLAS_LIBRARY=/usr/local/cuda/targets/sbsa-linux/lib/libcublas.so -DVLLM_CUTLASS_SRC_DIR=$(CURDIR)/third_party/flashinfer/3rdparty/cutlass
cross: CARGO_BUILD_TARGET=aarch64-unknown-linux-gnu
cross: CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER=aarch64-linux-gnu-gcc
cross: CC_aarch64_unknown_linux_gnu=aarch64-linux-gnu-gcc
cross: CXX_aarch64_unknown_linux_gnu=aarch64-linux-gnu-g++
cross: AR_aarch64_unknown_linux_gnu=aarch64-linux-gnu-ar
cross: PYO3_CROSS_PYTHON_VERSION=3.12
cross: PYTHON=/opt/venv/bin/python
cross: PYTHON_HOST_PLATFORM=linux-aarch64
cross: WHEEL_DIR=/wheels
cross: B12X_WHEEL_DIR=/wheels-b12x
cross: FLASHINFER_DIST_DIR=/wheels-flashinfer
cross: FLASHINFER_WHEEL_PLATFORM_TAG=manylinux_2_28_aarch64
cross: FLASHINFER_EXPECTED_ELF_MACHINE=AArch64
cross: FLASHINFER_CUDA_ARCH_LIST=12.0f 12.1a
cross: FLASHINFER_EXTRA_LDFLAGS=-L/usr/local/cuda/targets/sbsa-linux/lib -L/usr/local/cuda/targets/sbsa-linux/lib/stubs -Wl,-rpath-link,/usr/local/cuda/targets/sbsa-linux/lib
cross: sync

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

build-strix-wheel:
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

build-flashinfer: sync-deps
	mkdir -p $(FLASHINFER_DIST_DIR) "$(dir $(CCACHE_EXTRAFILES))"
	printf '%s\n' "$(NVCC_PREPEND_FLAGS)" "$(TORCH_CUDA_ARCH_LIST)" > "$(CCACHE_EXTRAFILES)"
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

build-wheel:
	mkdir -p $(WHEEL_DIR)
	_PYTHON_HOST_PLATFORM=$(PYTHON_HOST_PLATFORM) $(UV) build --python $(PYTHON) --no-build-isolation --wheel \
		--out-dir $(WHEEL_DIR) .

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

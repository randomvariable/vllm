#!/usr/bin/env bash
# This script is used to build FlashInfer wheels with AOT kernels

set -ex

# FlashInfer configuration
FLASHINFER_GIT_REPO="${FLASHINFER_GIT_REPO:-https://github.com/flashinfer-ai/flashinfer.git}"
FLASHINFER_SOURCE_DIR="${FLASHINFER_SOURCE_DIR:-}"
FLASHINFER_SOURCE_IS_LOCAL=false
BUILD_WHEEL="${BUILD_WHEEL:-true}"
BUILD_JIT_CACHE="${BUILD_JIT_CACHE:-false}"
FLASHINFER_DIST_DIR="${FLASHINFER_DIST_DIR:-flashinfer-dist}"
FLASHINFER_WHEEL_PLATFORM_TAG="${FLASHINFER_WHEEL_PLATFORM_TAG:-}"
FLASHINFER_JIT_CACHE_LOCAL_VERSION="${FLASHINFER_JIT_CACHE_LOCAL_VERSION:-}"

if [[ -z "${FLASHINFER_SOURCE_DIR}" && -z "${FLASHINFER_GIT_REF:-}" ]]; then
    echo "❌ FLASHINFER_GIT_REF must be specified" >&2
    exit 1
fi

if [[ -z "${CUDA_VERSION}" ]]; then
    echo "❌ CUDA_VERSION must be specified" >&2
    exit 1
fi

if [[ -n "${FLASHINFER_SOURCE_DIR}" ]]; then
    FLASHINFER_SOURCE_IS_LOCAL=true
    FLASHINFER_SOURCE_DIR="$(realpath "${FLASHINFER_SOURCE_DIR}")"
    echo "🏗️  Building FlashInfer from ${FLASHINFER_SOURCE_DIR} for CUDA ${CUDA_VERSION}"

    required_submodule_files=(
        "3rdparty/cutlass/include/cutlass/cutlass.h"
        "3rdparty/spdlog/include/spdlog/spdlog.h"
        "3rdparty/cccl/cub/cub/cub.cuh"
    )
    for required_file in "${required_submodule_files[@]}"; do
        if [[ ! -f "${FLASHINFER_SOURCE_DIR}/${required_file}" ]]; then
            echo "❌ Local FlashInfer source is missing ${required_file}; initialize nested submodules before building" >&2
            exit 1
        fi
    done
else
    echo "🏗️  Building FlashInfer ${FLASHINFER_GIT_REF} for CUDA ${CUDA_VERSION}"
    FLASHINFER_SOURCE_DIR="$(pwd)/flashinfer"
    git clone --depth 1 --recursive --shallow-submodules \
        --branch "${FLASHINFER_GIT_REF}" \
        "${FLASHINFER_GIT_REPO}" "${FLASHINFER_SOURCE_DIR}"
fi

# Set CUDA arch list based on CUDA version
# Exclude CUDA arches for older versions (11.x and 12.0-12.7)
if [[ "${CUDA_VERSION}" == 11.* ]]; then
    FI_TORCH_CUDA_ARCH_LIST="7.5 8.0 8.9"
elif [[ "${CUDA_VERSION}" == 12.[0-7]* ]]; then
    FI_TORCH_CUDA_ARCH_LIST="7.5 8.0 8.9 9.0a"
elif [[ "${CUDA_VERSION}" == 12.[8-9]* ]]; then
    # CUDA 12.8–12.9
    FI_TORCH_CUDA_ARCH_LIST="7.5 8.0 8.9 9.0a 10.0a 10.3a 12.0"
else
    # CUDA 13.0+
    FI_TORCH_CUDA_ARCH_LIST="7.5 8.0 8.9 9.0a 10.0f 11.0 12.0f 12.1"
fi
FI_TORCH_CUDA_ARCH_LIST="${FLASHINFER_CUDA_ARCH_LIST:-${FI_TORCH_CUDA_ARCH_LIST}}"

echo "🏗️ Building FlashInfer AOT for arches: ${FI_TORCH_CUDA_ARCH_LIST}"

mkdir -p "${FLASHINFER_DIST_DIR}"
FLASHINFER_DIST_DIR="$(realpath "${FLASHINFER_DIST_DIR}")"

pushd "${FLASHINFER_SOURCE_DIR}"
    # Make sure the wheel is built for the correct CUDA version
    # FlashInfer's published CUDA 13 wheels use the cu130 uv backend for
    # CUDA 13.x toolkits; uv has no separate cu133 backend.
    export UV_TORCH_BACKEND=cu130

    export TORCH_CUDA_ARCH_LIST="${FI_TORCH_CUDA_ARCH_LIST}"
    export FLASHINFER_CUDA_ARCH_LIST="${FI_TORCH_CUDA_ARCH_LIST}"
    if [[ "${BUILD_JIT_CACHE}" != "true" ]]; then
        python3 -m flashinfer.aot
    fi
    
    if [[ "${BUILD_WHEEL}" == "true" ]]; then
        # Build wheel for distribution
        uv build --python /opt/venv/bin/python --no-build-isolation --wheel --out-dir "${FLASHINFER_DIST_DIR}" .
        if [[ "${BUILD_JIT_CACHE}" == "true" ]]; then
            FLASHINFER_LOCAL_VERSION="${FLASHINFER_JIT_CACHE_LOCAL_VERSION}" \
            FLASHINFER_WHEEL_PLATFORM_TAG="${FLASHINFER_WHEEL_PLATFORM_TAG}" \
                uv build --python /opt/venv/bin/python --no-build-isolation --wheel \
                --out-dir "${FLASHINFER_DIST_DIR}" ./flashinfer-jit-cache
        fi
        # Fail closed on host-arch leakage: nvcc with no cross flags silently
        # emits host objects inside an aarch64-tagged wheel. Runs only when the
        # caller declares the expected machine (cross builds); native builds
        # skip it.
        if [[ -n "${FLASHINFER_EXPECTED_ELF_MACHINE:-}" ]]; then
            mapfile -t AOT_OBJS < <(find build -name '*.cuda.o')
            if [[ ${#AOT_OBJS[@]} -eq 0 ]]; then
                echo "❌ no AOT objects found under build/" >&2
                exit 1
            fi
            for OBJ in "${AOT_OBJS[@]}"; do
                if ! readelf -h "$OBJ" | grep -q "Machine:.*${FLASHINFER_EXPECTED_ELF_MACHINE}"; then
                    echo "❌ $OBJ is not ${FLASHINFER_EXPECTED_ELF_MACHINE} ELF" >&2
                    exit 1
                fi
            done
        fi
        echo "✅ FlashInfer wheels built successfully in ${FLASHINFER_DIST_DIR}/"
    else
        # Install directly (for Dockerfile)
        uv pip install --python /opt/venv/bin/python --no-build-isolation --force-reinstall .
        echo "✅ FlashInfer installed successfully"
    fi
popd

if [[ "${FLASHINFER_SOURCE_IS_LOCAL}" != "true" ]]; then
    rm -rf "${FLASHINFER_SOURCE_DIR}"
fi

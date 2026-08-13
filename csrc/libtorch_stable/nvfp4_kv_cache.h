// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#pragma once

#include "libtorch_stable/torch_utils.h"

#include <string>

// Defined in nvfp4_kv_cache_kernels.cu, which CMake compiles only when an
// SM100+/SM12x NVFP4 arch is in the build (ENABLE_NVFP4_SM100 /
// ENABLE_NVFP4_SM120).
//
// Declared here rather than with a local `extern` at the call site: a call-site
// declaration that omitted kv_cache_dtype mangled to a different symbol than
// the definition, so the extension linked with an undefined
// reshape_and_cache_nvfp4_dispatch and every `import vllm` in the runtime image
// failed. A shared declaration makes that class of drift a compile error.
//
// kv_cache_dtype selects the store-time scale search: "nvfp4" (max/6 default)
// or "nvfp4_4over6" (picks the lower-error of max/4 and max/6).
void reshape_and_cache_nvfp4_dispatch(
    torch::stable::Tensor& key, torch::stable::Tensor& value,
    torch::stable::Tensor& key_cache, torch::stable::Tensor& value_cache,
    torch::stable::Tensor& slot_mapping, torch::stable::Tensor& k_scale,
    torch::stable::Tensor& v_scale, const std::string& kv_cache_dtype);

#include "common.hpp"
#include <torch/csrc/stable/macros.h>

int32_t get_sm_version_num(int device) {
  int32_t major_capability, minor_capability;
  STD_CUDA_CHECK(cudaDeviceGetAttribute(
      &major_capability, cudaDevAttrComputeCapabilityMajor, device));
  STD_CUDA_CHECK(cudaDeviceGetAttribute(
      &minor_capability, cudaDevAttrComputeCapabilityMinor, device));
  int32_t version_num = major_capability * 10 + minor_capability;
  return version_num;
}

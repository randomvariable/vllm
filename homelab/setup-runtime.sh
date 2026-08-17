#!/bin/sh
set -eu

: "${HF_HOME:=/opt/vllm/cache/huggingface}"
: "${VLLM_CACHE_ROOT:=/opt/vllm/cache}"
: "${TIKTOKEN_ENCODINGS_BASE:=/opt/vllm/tiktoken_encodings}"

mkdir -p "$TIKTOKEN_ENCODINGS_BASE" "$HF_HOME" "$VLLM_CACHE_ROOT"
curl --fail --silent --show-error --location \
  -o "$TIKTOKEN_ENCODINGS_BASE/o200k_base.tiktoken" \
  https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken
curl --fail --silent --show-error --location \
  -o "$TIKTOKEN_ENCODINGS_BASE/cl100k_base.tiktoken" \
  https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken
chown -R vllm:vllm /opt/vllm

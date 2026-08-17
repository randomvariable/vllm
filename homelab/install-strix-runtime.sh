#!/bin/sh
set -eu

filter_runtime_requirements() {
    grep -vhiE '^[[:space:]]*(-r|#|$)' \
        /runtime-requirements/rocm.txt /runtime-requirements/common.txt \
      | grep -viE '^[[:space:]]*(torch|pytorch-triton-rocm|triton|torchvision|torchaudio|rocm[-_]?sdk[-_a-z]*)([[:space:]]|==|>|<|~|;|\[|$)' \
      | sort -u
}

filter_runtime_requirements > /tmp/runtime-reqs.txt
/opt/venv/bin/pip install --no-cache-dir -r /tmp/runtime-reqs.txt
/opt/venv/bin/pip install --no-cache-dir xxhash
/opt/venv/bin/pip install --no-cache-dir --no-deps /wheels/vllm-*.whl

if ls /wheels-gguf/*.whl >/dev/null 2>&1; then
    /opt/venv/bin/pip install --no-cache-dir --no-deps /wheels-gguf/*.whl
    /opt/venv/bin/pip install --no-cache-dir 'gguf>=0.17.0'
fi

rm -rf /wheels /wheels-gguf /tmp/runtime-reqs.txt

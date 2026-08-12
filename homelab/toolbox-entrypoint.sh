#!/bin/sh
# Toolbox entrypoint wrapper.
#
# Makes one image serve two launch modes:
#   * Interactive: `docker run -it ... vllm-<arch>-toolbox` (a TTY, no args)
#     drops into the textual TUI model launcher.
#   * Headless / Kubernetes: any args (or a non-TTY) exec `vllm "$@"`, so the
#     default `serve` behaviour and k8s command/args overrides work unchanged.
if [ -t 0 ] && [ "$#" -eq 0 ]; then
  exec python3 /opt/vllm/toolbox/vllm_toolbox.py
fi
exec vllm "$@"

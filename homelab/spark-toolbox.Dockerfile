# syntax=docker/dockerfile:1.7
#
# Interactive "toolbox" variant of the DGX Spark (CUDA sm_121a) runtime.
#
# Layers a kyuz0-toolbox-style experience on top of the headless, Kubernetes-
# optimised spark image: a textual TUI model launcher, Hugging Face download
# tooling, and GPU diagnostics, for standalone `docker run -it` use. The
# entrypoint wrapper keeps the headless contract too -- with args (or no TTY)
# it execs `vllm "$@"` exactly like the base image, so the same image works as
# a k8s pod or an interactive toolbox.
#
# BASE_IMAGE defaults to the public GHCR package; the private build harness
# overrides it with its own registry via --build-arg BASE_IMAGE=...
ARG BASE_IMAGE=ghcr.io/randomvariable/vllm:spark-latest
FROM ${BASE_IMAGE}

USER root

# Interactive tooling: TUI deps (textual/rich/pyyaml), HF model download
# (huggingface_hub + hf_transfer), and CUDA GPU diagnostics (nvtop). The base
# image sets UV_BREAK_SYSTEM_PACKAGES, so pip installs into the system python
# that `python3` resolves to.
RUN apt-get update && \
    apt-get install -y --no-install-recommends nvtop vim sudo procps && \
    rm -rf /var/lib/apt/lists/* && \
    pip install --no-cache-dir textual rich pyyaml huggingface_hub hf_transfer

# The toolbox TUI + presets, and the TTY-or-serve entrypoint wrapper.
COPY homelab/toolbox/ /opt/vllm/toolbox/
COPY homelab/toolbox-entrypoint.sh /usr/local/bin/toolbox-entrypoint.sh
RUN chmod +x /usr/local/bin/toolbox-entrypoint.sh && \
    chown -R vllm:vllm /opt/vllm/toolbox

USER vllm
EXPOSE 8000
ENTRYPOINT ["/usr/local/bin/toolbox-entrypoint.sh"]
CMD ["serve"]

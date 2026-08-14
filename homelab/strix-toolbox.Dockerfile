# syntax=docker/dockerfile:1.7
#
# Interactive "toolbox" variant of the Strix Halo (ROCm gfx1151) runtime.
#
# Layers a kyuz0-toolbox-style experience on top of the headless, Kubernetes-
# optimised strix image: a textual TUI model launcher, Hugging Face download
# tooling, and ROCm GPU diagnostics, for standalone `docker run -it` use. The
# entrypoint wrapper keeps the headless contract too -- with args (or no TTY)
# it execs `vllm "$@"` exactly like the base image, so the same image works as
# a k8s pod or an interactive toolbox.
#
# BASE_IMAGE defaults to the public GHCR package; the private build harness
# overrides it with its own registry via --build-arg BASE_IMAGE=...
ARG BASE_IMAGE=ghcr.io/randomvariable/vllm:strix-latest
FROM ${BASE_IMAGE}

# Interactive tooling: TUI deps (textual/rich/pyyaml), HF model download
# (huggingface_hub + hf_transfer), and ROCm GPU diagnostics (radeontop; rocminfo
# ships with the ROCm base). Installs go into the /opt/venv the runtime uses;
# put /opt/venv/bin first on PATH so the entrypoint's `python3` finds textual.
ENV PATH=/opt/venv/bin:${PATH}
RUN apt-get update && \
    apt-get install -y --no-install-recommends radeontop vim sudo procps && \
    rm -rf /var/lib/apt/lists/* && \
    /opt/venv/bin/pip install --no-cache-dir textual rich pyyaml huggingface_hub hf_transfer

# The toolbox TUI + presets, and the TTY-or-serve entrypoint wrapper.
COPY homelab/toolbox/ /opt/vllm/toolbox/
COPY homelab/toolbox-entrypoint.sh /usr/local/bin/toolbox-entrypoint.sh
RUN chmod +x /usr/local/bin/toolbox-entrypoint.sh

EXPOSE 8000
ENTRYPOINT ["/usr/local/bin/toolbox-entrypoint.sh"]
CMD ["serve"]

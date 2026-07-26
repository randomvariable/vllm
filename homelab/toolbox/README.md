# vLLM Toolbox

Interactive, platform-neutral launcher for vLLM toolbox containers. It detects
CUDA or ROCm, browses models already present in the Hugging Face cache, applies
optional presets, and streams `vllm serve` output in the terminal.

## Requirements

- Python 3.10+
- `vllm`, `textual`, `rich`, and `PyYAML` installed in the toolbox image
- Optional GPU tools: `nvidia-smi`, or `rocm-smi` and `rocminfo`

Run it inside either toolbox image:

```bash
python homelab/toolbox/vllm_toolbox.py
```

The launcher uses `$HF_HOME` when set. Otherwise it checks
`/opt/vllm/cache/huggingface`, then falls back to
`~/.cache/huggingface`. Set `VLLM_TOOLBOX_PRESETS` to override the default
`models.yaml` preset path:

```bash
VLLM_TOOLBOX_PRESETS=/models/models.yaml python vllm_toolbox.py
```

## Models

The browser recognizes:

- Hugging Face snapshots containing `config.json` and `*.safetensors`
- `*.gguf` files in Hugging Face snapshots
- Remote IDs or local paths typed directly or declared in `models.yaml`

Copy `models.yaml.example` to `models.yaml`. Each item under `models` supports
`name`, `source`, `hardware`, `gguf_file`, `tokenizer`, `max_model_len`,
`gpu_memory_utilization`, `extra_args`, and `notes`. `hardware` is `gfx1151`,
`sm_121a`, or `both`. GGUF entries can use `gguf_file` to select a matching
cached quant and should supply a compatible tokenizer.

The launcher detects ROCm/Strix Halo (`gfx1151`) or CUDA/DGX Spark (`sm_121a`).
Matching presets appear first under "Recommended for your hardware". Portable
presets and cached model names are also ranked when they advertise common
ROCmFPX/AWQ/UMA-MoE or Blackwell FP4/FP8 formats. Missing GPU tools, malformed
YAML, absent presets, and empty caches are handled without blocking direct model
entry.

## Keys

| Key | Action |
| --- | --- |
| `l` | Launch selected model |
| `p` | Print launch command |
| `g` | Refresh GPU diagnostics |
| `r` | Rescan models and presets |
| `q` | Quit |
| `Esc` / `Ctrl-C` | Stop server and return |

Common launch options are editable in the right pane. Free-text extra arguments
are parsed with shell quoting rules and appended to the command without invoking
a shell. Prefix caching uses `xxhash`; the Rust frontend sets
`VLLM_USE_RUST_FRONTEND=1` for the launched process.

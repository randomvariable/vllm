#!/usr/bin/env python3
"""Interactive launcher for vLLM toolbox containers."""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from rich.text import Text
from textual import work
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import Screen
from textual.widgets import (
    Button,
    Checkbox,
    DataTable,
    Footer,
    Header,
    Input,
    Label,
    RichLog,
    Select,
    Static,
)


@dataclass
class Model:
    label: str
    path: str
    kind: str
    tokenizer: str = ""
    context: str = "8192"
    gpu_memory: str = "0.90"
    extra_args: str = ""
    size: int | None = None
    hardware: str = "both"
    gguf_file: str = ""
    notes: str = ""


@dataclass
class Hardware:
    platform: str
    tag: str
    name: str
    details: str


def human_size(size: int | None) -> str:
    if size is None:
        return "-"
    value = float(size)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            return f"{value:.1f} {unit}"
        value /= 1024
    return "-"


def hf_home() -> Path:
    """Return Hugging Face cache root used by this container."""
    if value := os.environ.get("HF_HOME"):
        return Path(value).expanduser()
    container_cache = Path("/opt/vllm/cache/huggingface")
    if container_cache.exists():
        return container_cache
    return Path.home() / ".cache/huggingface"


def _model_id(cache_dir: Path) -> str:
    name = cache_dir.name.removeprefix("models--")
    return name.replace("--", "/")


def scan_models(cache_root: Path) -> list[Model]:
    """Find usable safetensors snapshots and GGUF files in the HF cache."""
    found: list[Model] = []
    hub = cache_root / "hub"
    if not hub.is_dir():
        return found

    for cache_dir in sorted(hub.glob("models--*")):
        snapshots = cache_dir / "snapshots"
        if not snapshots.is_dir():
            continue
        for snapshot in sorted(snapshots.iterdir()):
            if not snapshot.is_dir():
                continue
            model_id = _model_id(cache_dir)
            if (snapshot / "config.json").is_file() and any(
                snapshot.glob("*.safetensors")
            ):
                size = sum(
                    path.stat().st_size for path in snapshot.glob("*.safetensors")
                )
                found.append(Model(model_id, str(snapshot), "safetensors", size=size))
            for gguf in sorted(snapshot.glob("*.gguf")):
                label = f"{model_id} / {gguf.name}"
                found.append(
                    Model(
                        label,
                        str(gguf),
                        "GGUF",
                        tokenizer=model_id,
                        size=gguf.stat().st_size,
                    )
                )
    return found


def _preset_value(entry: dict[str, Any], key: str, default: str = "") -> str:
    value = entry.get(key, default)
    return str(value).strip() if value is not None else default


def load_presets(path: Path) -> list[Model]:
    """Load optional YAML presets; malformed entries are ignored independently."""
    if not path.is_file():
        return []
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, UnicodeError, yaml.YAMLError):
        return []
    if not isinstance(document, dict) or not isinstance(document.get("models", []), list):
        return []

    presets: list[Model] = []
    for entry in document.get("models", []):
        if not isinstance(entry, dict):
            continue
        source = _preset_value(entry, "source")
        hardware = _preset_value(entry, "hardware", "both").casefold()
        if not source or hardware not in {"gfx1151", "sm_121a", "both"}:
            continue
        gguf_file = _preset_value(entry, "gguf_file")
        presets.append(
            Model(
                _preset_value(entry, "name", source),
                os.path.expandvars(os.path.expanduser(source)),
                "GGUF preset" if gguf_file else "preset",
                _preset_value(entry, "tokenizer"),
                _preset_value(entry, "max_model_len", "8192"),
                _preset_value(entry, "gpu_memory_utilization", "0.90"),
                _preset_value(entry, "extra_args"),
                hardware=hardware,
                gguf_file=gguf_file,
                notes=_preset_value(entry, "notes"),
            )
        )
    return presets


def _run(command: list[str], timeout: float = 3) -> str:
    if not shutil.which(command[0]):
        return ""
    try:
        return subprocess.run(
            command,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
            check=False,
        ).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        return ""


def detect_hardware() -> Hardware:
    """Detect toolbox hardware and return compact, refreshable diagnostics."""
    if shutil.which("rocm-smi"):
        output = _run(
            ["rocm-smi", "--showproductname", "--showmeminfo", "vram", "--showuse"]
        )
        arch = ""
        if rocminfo := _run(["rocminfo"], timeout=4):
            arch = next(
                (
                    token
                    for line in rocminfo.splitlines()
                    for token in line.split()
                    if token.startswith("gfx")
                ),
                "",
            )
        tag = "gfx1151" if arch == "gfx1151" else "both"
        name = "Strix Halo" if tag == "gfx1151" else "AMD GPU"
        details = output or "rocm-smi found, but GPU diagnostics are unavailable"
        if arch:
            details += f"\nArchitecture: {arch}"
        return Hardware("ROCm", tag, name, details)

    query = "name,memory.total,memory.used,utilization.gpu,compute_cap"
    if shutil.which("nvidia-smi") and (output := _run(
        ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"]
    )):
        rows = []
        names = []
        compute_caps = []
        for index, row in enumerate(output.splitlines()):
            parts = [part.strip() for part in row.split(",")]
            if len(parts) >= 4:
                names.append(parts[0])
                if len(parts) >= 5:
                    compute_caps.append(parts[4])
                rows.append(
                    f"GPU {index}: {parts[0]} | {parts[2]}/{parts[1]} MiB | "
                    f"{parts[3]}% util"
                    + (f" | compute {parts[4]}" if len(parts) >= 5 else "")
                )
        is_spark = any(cap.startswith("12.1") for cap in compute_caps) or any(
            marker in name.casefold()
            for name in names
            for marker in ("gb10", "dgx spark")
        )
        tag = "sm_121a" if is_spark else "both"
        return Hardware("CUDA", tag, ", ".join(names) or "NVIDIA GPU", "\n".join(rows))

    return Hardware(
        "Unknown",
        "both",
        "Unknown GPU",
        "No nvidia-smi or rocm-smi GPU data available",
    )


STRIX_HINTS = (
    "rocmfp4",
    "rocmfpx",
    "ifp2",
    "awq",
    "laguna-xs",
    "qwen3-moe",
    "a3b",
)
SPARK_HINTS = ("fp4", "fp8", "nvfp4", "blackwell")


def is_recommended(model: Model, hardware_tag: str) -> bool:
    if hardware_tag == "both":
        return model.hardware == "both"
    if model.hardware == hardware_tag:
        return True
    if model.hardware != "both":
        return False
    text = f"{model.label} {model.path} {model.gguf_file} {model.notes}".casefold()
    hints = STRIX_HINTS if hardware_tag == "gfx1151" else SPARK_HINTS
    return any(hint in text for hint in hints)


def resolve_gguf(model: Model, cache_root: Path) -> str:
    """Resolve a preset's GGUF glob against local paths or cached snapshots."""
    if not model.gguf_file:
        return model.path
    source = Path(model.path)
    roots: list[Path] = []
    if source.exists():
        roots.append(source if source.is_dir() else source.parent)
    cache_dir = cache_root / "hub" / f"models--{model.path.replace('/', '--')}"
    roots.extend(sorted((cache_dir / "snapshots").glob("*")))
    for root in roots:
        if root.is_dir():
            matches = sorted(root.glob(model.gguf_file))
            if matches:
                return str(matches[0])
    return model.path


class LaunchScreen(Screen[None]):
    """Streaming vLLM process output."""

    BINDINGS = [("escape", "cancel", "Stop / back"), ("ctrl+c", "cancel", "Stop")]

    def __init__(self, command: list[str], environment: dict[str, str]) -> None:
        super().__init__()
        self.command = command
        self.environment = environment
        self.process: subprocess.Popen[str] | None = None

    def compose(self) -> ComposeResult:
        yield Header()
        yield Static("Starting vLLM...", id="launch-status")
        yield RichLog(id="launch-log", wrap=True, highlight=True, markup=False)
        yield Footer()

    def on_mount(self) -> None:
        self.query_one(RichLog).write("$ " + shlex.join(self.command))
        self.run_server()

    @work(thread=True, exclusive=True)
    def run_server(self) -> None:
        try:
            self.process = subprocess.Popen(
                self.command,
                env=self.environment,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=1,
            )
            assert self.process.stdout is not None
            for line in self.process.stdout:
                self.app.call_from_thread(self.query_one(RichLog).write, line.rstrip())
            code = self.process.wait()
            message = f"vLLM exited with status {code}."
        except OSError as error:
            message = f"Launch failed: {error}"
        self.app.call_from_thread(self.notify, message)
        self.app.call_from_thread(self.dismiss)

    def action_cancel(self) -> None:
        if self.process and self.process.poll() is None:
            self.process.terminate()
        else:
            self.dismiss()

    def on_unmount(self) -> None:
        if self.process and self.process.poll() is None:
            self.process.terminate()


class ConfirmScreen(Screen[bool]):
    """Confirm the exact command before replacing the screen with its output."""

    CSS = """
    ConfirmScreen { align: center middle; }
    #confirm-box { width: 80%; height: auto; padding: 1 2; border: solid $accent; }
    #confirm-command { height: auto; margin: 1 0; }
    #confirm-buttons { height: 3; }
    #confirm-buttons Button { width: 1fr; }
    """

    def __init__(self, command: str) -> None:
        super().__init__()
        self.command = command

    def compose(self) -> ComposeResult:
        with Vertical(id="confirm-box"):
            yield Label("Launch this command?")
            yield Static(self.command, id="confirm-command")
            with Horizontal(id="confirm-buttons"):
                yield Button("Launch", variant="success", id="confirm-launch")
                yield Button("Cancel", id="confirm-cancel")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss(event.button.id == "confirm-launch")


class ToolboxApp(App[None]):
    """Browse cached models, tune common options, and launch vLLM."""

    TITLE = "vLLM Toolbox"
    CSS = """
    Screen { layout: vertical; }
    #banner { height: 3; padding: 0 2; content-align: left middle; }
    #body { height: 1fr; }
    #models { width: 3fr; height: 1fr; }
    #config { width: 2fr; padding: 0 1; border-left: solid $accent; }
    #config Label { margin-top: 1; }
    #checks { height: auto; }
    #checks Checkbox { width: 1fr; }
    #buttons { height: 3; margin-top: 1; }
    #buttons Button { width: 1fr; }
    #gpu { height: 8; padding: 1 2; border-top: solid $accent; }
    #launch-log { height: 1fr; padding: 0 1; }
    #launch-status { height: 3; padding: 1 2; }
    """
    BINDINGS = [
        ("q", "quit", "Quit"),
        ("enter", "select_model", "Select"),
        ("l", "launch", "Launch"),
        ("p", "print_command", "Print command"),
        ("g", "refresh_gpu", "Refresh GPU"),
        ("r", "refresh_models", "Reload presets"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self.cache_root = hf_home()
        default_presets = Path(__file__).with_name("models.yaml")
        preset_value = os.environ.get("VLLM_TOOLBOX_PRESETS", str(default_presets))
        self.preset_path = Path(preset_value).expanduser()
        self.models: list[Model] = []
        self.hardware = Hardware("Unknown", "both", "Unknown GPU", "")

    def compose(self) -> ComposeResult:
        yield Header()
        yield Static(id="banner")
        with Horizontal(id="body"):
            yield DataTable(id="models", cursor_type="row", zebra_stripes=True)
            with VerticalScroll(id="config"):
                yield Label("Model (HF repo ID or local path)")
                yield Input(id="model", placeholder="org/model or /models/model")
                yield Label("Tokenizer (required for GGUF)")
                yield Input(id="tokenizer", placeholder="org/model or local path")
                yield Label("GGUF file (preset glob or resolved local file)")
                yield Input(id="gguf-file", placeholder="*Q4_0*.gguf")
                yield Label("Context length / GPU memory / tensor parallel")
                with Horizontal():
                    yield Input("8192", id="context", type="integer")
                    yield Input("0.90", id="gpu-memory", type="number")
                    yield Input("1", id="tensor-parallel", type="integer")
                yield Label("Dtype / quantization / host / port")
                with Horizontal():
                    yield Select.from_values(
                        ["auto", "half", "bfloat16", "float"],
                        value="auto",
                        allow_blank=False,
                        id="dtype",
                    )
                    yield Input("auto", id="quantization")
                    yield Input("0.0.0.0", id="host")
                    yield Input("8000", id="port", type="integer")
                with Horizontal(id="checks"):
                    yield Checkbox("Prefix cache (xxhash128)", True, id="prefix-cache")
                    yield Checkbox("Rust frontend", True, id="rust-frontend")
                yield Label("Extra vllm serve arguments")
                yield Input(id="extra-args", placeholder="--served-model-name ...")
                yield Static(id="model-notes")
                with Horizontal(id="buttons"):
                    yield Button("Launch", variant="success", id="launch")
                    yield Button("Print", id="print")
                    yield Button("Refresh", id="refresh")
        yield Static(id="gpu")
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one(DataTable)
        table.add_columns("Section", "Model", "Format", "Size", "Location")
        self.refresh_gpu()
        self.refresh_models()
        self.set_interval(5, self.refresh_gpu)

    def refresh_models(self) -> None:
        discovered = load_presets(self.preset_path) + scan_models(self.cache_root)
        recommended = [
            model for model in discovered if is_recommended(model, self.hardware.tag)
        ]
        other = [model for model in discovered if model not in recommended]
        self.models = recommended + other
        table = self.query_one(DataTable)
        table.clear()
        for index, model in enumerate(self.models):
            section = (
                "Recommended for your hardware"
                if index < len(recommended)
                else "Other presets and cached models"
            )
            table.add_row(
                section, model.label, model.kind, human_size(model.size), model.path
            )
        self.query_one("#banner", Static).update(
            f"vLLM Toolbox | {self.hardware.platform} {self.hardware.tag} | "
            f"{len(self.models)} models ({len(recommended)} recommended) | "
            f"cache: {self.cache_root} | presets: {self.preset_path}"
        )
        if self.models:
            table.move_cursor(row=0)
            self._apply_model(self.models[0])

    def refresh_gpu(self) -> None:
        self.hardware = detect_hardware()
        self.query_one("#gpu", Static).update(
            Text.from_markup(
                f"[bold]{self.hardware.platform} | {self.hardware.name} | "
                f"{self.hardware.tag}[/bold]\n{self.hardware.details}"
            )
        )
        banner = self.query_one("#banner", Static)
        banner.update(
            f"vLLM Toolbox | {self.hardware.platform} {self.hardware.tag} | "
            f"{len(self.models)} models | "
            f"cache: {self.cache_root} | presets: {self.preset_path}"
        )

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        if 0 <= event.cursor_row < len(self.models):
            self._apply_model(self.models[event.cursor_row])

    def _apply_model(self, model: Model) -> None:
        resolved = resolve_gguf(model, self.cache_root)
        self.query_one("#model", Input).value = model.path
        self.query_one("#tokenizer", Input).value = model.tokenizer
        self.query_one("#gguf-file", Input).value = (
            resolved if resolved != model.path else model.gguf_file
        )
        self.query_one("#context", Input).value = model.context
        self.query_one("#gpu-memory", Input).value = model.gpu_memory
        self.query_one("#quantization", Input).value = (
            "gguf" if model.kind.casefold() == "gguf" else "auto"
        )
        self.query_one("#extra-args", Input).value = model.extra_args
        self.query_one("#model-notes", Static).update(model.notes)

    def selected_model(self) -> Model | None:
        row = self.query_one(DataTable).cursor_row
        return self.models[row] if 0 <= row < len(self.models) else None

    def build_command(self) -> tuple[list[str], dict[str, str]]:
        get = lambda widget_id: self.query_one(widget_id, Input).value.strip()
        model_path = get("#model")
        if not model_path:
            raise ValueError("Select or enter a model first")
        gguf_file = get("#gguf-file")
        if gguf_file:
            if any(character in gguf_file for character in "*?["):
                raise ValueError(f"No cached GGUF matches {gguf_file}")
            model_path = gguf_file
        command = ["vllm", "serve", model_path]
        tokenizer = get("#tokenizer")
        if model_path.casefold().endswith(".gguf") and not tokenizer:
            raise ValueError("GGUF models require a tokenizer")
        quantization = get("#quantization")
        options: Iterable[tuple[str, str]] = (
            ("--tokenizer", tokenizer),
            ("--max-model-len", get("#context")),
            ("--gpu-memory-utilization", get("#gpu-memory")),
            ("--tensor-parallel-size", get("#tensor-parallel")),
            ("--dtype", str(self.query_one("#dtype", Select).value)),
            ("--quantization", "" if quantization == "auto" else quantization),
            ("--host", get("#host")),
            ("--port", get("#port")),
        )
        for flag, value in options:
            if value:
                command.extend((flag, value))
        if self.query_one("#prefix-cache", Checkbox).value:
            command.extend(
                ("--enable-prefix-caching", "--prefix-caching-hash-algo", "xxhash")
            )
        extra = get("#extra-args")
        if extra:
            command.extend(shlex.split(extra))
        environment = os.environ.copy()
        environment["HF_HOME"] = str(self.cache_root)
        if self.query_one("#rust-frontend", Checkbox).value:
            environment["VLLM_USE_RUST_FRONTEND"] = "1"
        else:
            environment.pop("VLLM_USE_RUST_FRONTEND", None)
        return command, environment

    def action_launch(self) -> None:
        try:
            command, environment = self.build_command()
        except (ValueError, TypeError) as error:
            self.notify(str(error), severity="error")
            return
        rendered = shlex.join(command)
        self.push_screen(
            ConfirmScreen(rendered),
            lambda confirmed: self._launch_confirmed(
                bool(confirmed), command, environment
            ),
        )

    def _launch_confirmed(
        self,
        confirmed: bool,
        command: list[str],
        environment: dict[str, str],
    ) -> None:
        if confirmed:
            self.push_screen(LaunchScreen(command, environment))

    def action_print_command(self) -> None:
        try:
            command, environment = self.build_command()
        except (ValueError, TypeError) as error:
            self.notify(str(error), severity="error")
            return
        prefix = (
            "VLLM_USE_RUST_FRONTEND=1 "
            if environment.get("VLLM_USE_RUST_FRONTEND") == "1"
            else ""
        )
        rendered = prefix + shlex.join(command)
        print(rendered)
        self.notify(rendered, title="Command", timeout=10)

    def action_refresh_gpu(self) -> None:
        self.refresh_gpu()
        self.refresh_models()

    def action_select_model(self) -> None:
        if model := self.selected_model():
            self._apply_model(model)

    def action_refresh_models(self) -> None:
        self.refresh_models()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        actions = {
            "launch": self.action_launch,
            "print": self.action_print_command,
            "refresh": self.action_refresh_models,
        }
        if event.button.id in actions:
            actions[event.button.id]()


if __name__ == "__main__":
    ToolboxApp().run()

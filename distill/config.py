"""Settings live in config.toml; this module loads them and resolves paths.

Everything a run depends on comes from the file, including which machine size
it is aimed at. A preset is a named patch on top of the defaults, so switching
from a laptop-sized rehearsal to the real thing is one word rather than a set of
edits that are easy to get half-right.

Paths are resolved against the package directory, not the working directory, so
the whole folder can be copied to a stick and run from anywhere.
"""

from __future__ import annotations

import copy
import os
import tomllib
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent


# ==============================
# ===  Loading               ===
# ==============================

def _merge(base: dict, patch: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


def _coerce(text: str) -> Any:
    stripped = text.strip()
    lowered = stripped.lower()
    if lowered in ("true", "false"):
        return lowered == "true"
    if lowered in ("none", "null"):
        return None
    # A list on the command line has to arrive as a list: settings like
    # data.subsets are iterated, and a string iterates character by character
    # without complaining about it.
    if stripped.startswith(("[", "{")):
        import json

        try:
            return json.loads(stripped)
        except ValueError as problem:
            raise ValueError(f"could not read {stripped!r} as a list: {problem}")
    if "," in stripped and not stripped.replace(",", "").strip().isdigit():
        return [part.strip() for part in stripped.split(",") if part.strip()]
    try:
        return int(stripped)
    except ValueError:
        pass
    try:
        return float(stripped)
    except ValueError:
        return text


class Config:
    """Dotted read access over the parsed table, plus path resolution."""

    def __init__(self, data: dict, source: Path):
        self.data = data
        self.source = source
        # Which keys came from --set rather than from the file. A setting the
        # user typed out is a different thing from a default, and refusing to
        # honour it should be an error where ignoring a default is not.
        self.overridden: set[str] = set()

    def was_set(self, path: str) -> bool:
        """True when this key was given on the command line."""
        return path in self.overridden

    # --- reading ---

    def get(self, path: str, default: Any = None) -> Any:
        node: Any = self.data
        for part in path.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def require(self, path: str) -> Any:
        value = self.get(path, _MISSING)
        if value is _MISSING:
            raise KeyError(f"{self.source.name} is missing [{path}]")
        return value

    def set(self, path: str, value: Any) -> None:
        parts = path.split(".")
        node = self.data
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value

    # --- paths ---

    def path(self, key: str) -> Path:
        raw = Path(str(self.require(key))).expanduser()
        return raw if raw.is_absolute() else (ROOT / raw).resolve()

    def run_dir(self) -> Path:
        return self.path("paths.runs") / str(self.get("run.name", "default"))

    def __repr__(self) -> str:
        return f"Config({self.source.name}, preset={self.get('preset')})"


_MISSING = object()


def load(path: str | os.PathLike | None = None,
         preset: str | None = None,
         overrides: list[str] | None = None) -> Config:
    """Read the file, fold in the chosen preset, then any --set overrides."""
    source = Path(path) if path else (ROOT / "config.toml")
    if not source.is_absolute():
        source = (ROOT / source).resolve()
    if not source.exists():
        raise FileNotFoundError(f"no config at {source}")

    with source.open("rb") as handle:
        raw = tomllib.load(handle)

    presets = raw.pop("presets", {})
    chosen = preset or raw.get("preset")
    if chosen:
        if chosen not in presets:
            names = ", ".join(sorted(presets)) or "none defined"
            raise KeyError(f"unknown preset '{chosen}' (have: {names})")
        raw = _merge(raw, presets[chosen])
        raw["preset"] = chosen

    config = Config(raw, source)
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"--set expects key=value, got '{item}'")
        key, value = item.split("=", 1)
        config.set(key.strip(), _coerce(value))
        config.overridden.add(key.strip())
    return config


# ==============================
# ===  Device                ===
# ==============================

def device_type(device: str) -> str:
    """"cuda:1" names a device, "cuda" names its type.

    Autocast, DistributedDataParallel and every comparison in this package want
    the type; only `.to()` and `set_device` want the full string with the index.
    """
    return device.split(":", 1)[0]


def resolve_device(requested: str = "auto") -> str:
    """Pick a device once, here, so nothing downstream has to guess."""
    import torch

    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_dtype(name: str, device: str):
    """bfloat16 where it is real, float32 where it would be emulated slowly."""
    import torch

    table = {"bfloat16": torch.bfloat16, "float16": torch.float16,
             "float32": torch.float32}
    dtype = table.get(name, torch.bfloat16)
    kind = device_type(device)
    if kind == "cpu" and dtype is not torch.float32:
        return torch.float32
    if kind == "mps" and dtype is torch.bfloat16:
        return torch.float16
    return dtype

"""Saving a run so that losing the machine costs minutes, not days.

Only the trainable tensors are stored — the master weights and the group scales.
Everything else is identical to the teacher checkpoint already on disk, so there
is no reason to write it out again every few hundred steps.

Writes go to a temporary name and are renamed into place. A run killed during a
save then still has its previous checkpoint intact, which matters because that
is exactly when machines get killed.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from . import ui


def _slots(run_dir: Path) -> list[Path]:
    """Every checkpoint in the directory, oldest first.

    Ordered by when the file was written, not by the number in its name. The
    number counts steps, and a step is only the same size within one run: a run
    resumed with a larger micro-batch renumbers from a smaller figure, so a
    checkpoint from an earlier run can carry a larger number than anything the
    current run will ever reach. Sorted by name, that stale file is picked as
    the newest for ever - it is handed to `eval` and `export`, and a resumed run
    rewinds to it, throwing away every hour since. Renaming into place is what
    sets the modification time, so it is exactly the moment the file became
    readable.
    """
    return sorted(run_dir.glob("step*.pt"), key=lambda p: p.stat().st_mtime)


def latest(run_dir: Path) -> Path | None:
    found = _slots(run_dir)
    return found[-1] if found else None


def save(run_dir: Path, model, optimizer, step: int, tokens: int,
         keep: int = 3, extra: dict | None = None) -> Path:
    import torch

    run_dir.mkdir(parents=True, exist_ok=True)
    target = run_dir / f"step{step}.pt"
    staging = target.with_suffix(".pt.partial")

    trainable = {name: param.detach().cpu()
                 for name, param in model.named_parameters()
                 if param.requires_grad}
    payload = {
        "step": step,
        "tokens": tokens,
        "written": time.time(),
        "parameters": trainable,
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "extra": extra or {},
    }
    torch.save(payload, staging)
    staging.rename(target)

    for stale in _slots(run_dir)[:-keep]:
        stale.unlink(missing_ok=True)
    return target


def load_into(model, optimizer, path: Path) -> int:
    """Restore trainable tensors, and the optimizer if one was given."""
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=False)
    stored = payload["parameters"]

    own = dict(model.named_parameters())
    missing = [name for name in own if name not in stored and own[name].requires_grad]
    unexpected = [name for name in stored if name not in own]
    for name, value in stored.items():
        if name in own:
            own[name].data.copy_(value.to(own[name].dtype))

    if missing:
        ui.warn(f"{len(missing)} trainable tensors were not in the checkpoint")
    if unexpected:
        ui.warn(f"{len(unexpected)} tensors in the checkpoint had no home")
    if optimizer is not None and payload.get("optimizer"):
        try:
            optimizer.load_state_dict(payload["optimizer"])
        except ValueError:
            ui.warn("optimizer state did not fit; continuing with a fresh one")
    return int(payload.get("step", 0))


def read_meta(path: Path) -> dict:
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=False)
    return {"step": payload.get("step", 0), "tokens": payload.get("tokens", 0),
            "extra": payload.get("extra", {})}


# ==============================
# ===  Metrics log           ===
# ==============================

class MetricsLog:
    """One json object per line; survives a kill, opens in anything."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path

    def write(self, record: dict) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")

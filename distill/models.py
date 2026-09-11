"""Fetching the teacher once, and building the student out of it.

Both models are the same checkpoint. The teacher stays in full precision and
frozen; the student has every linear layer inside its blocks replaced by a
binary one, initialised from the very weights the teacher is still using. That
initialisation is the whole point — the signs start where the real model put
them, and training only moves the ones that are worth moving.

Weights are pulled into `assets/models/` rather than the shared cache so that
the folder can be copied to a stick and used on a machine with no network.
"""

from __future__ import annotations

import os
from pathlib import Path

from . import ui

KEEP = ["*.json", "*.txt", "*.model", "*.safetensors", "tokenizer*"]
DROP = ["*.pth", "*.bin", "*.h5", "*.msgpack", "*.onnx", "*consolidated*"]


# ==============================
# ===  Local copy            ===
# ==============================

def local_dir(config) -> Path:
    name = str(config.require("model.teacher")).replace("/", "_")
    return config.path("paths.models") / name


def fetch_model(config) -> Path:
    """Download the checkpoint into the package. Safe to re-run."""
    from huggingface_hub import snapshot_download

    repo = str(config.require("model.teacher"))
    target = local_dir(config)
    target.mkdir(parents=True, exist_ok=True)
    ui.step(f"downloading {repo}")
    snapshot_download(repo, local_dir=str(target),
                      allow_patterns=KEEP, ignore_patterns=DROP)
    ui.good(f"model in {target}")
    return target


def model_source(config) -> str:
    """Prefer the local copy; fall back to the hub name if it is not there."""
    target = local_dir(config)
    if (target / "config.json").exists():
        return str(target)
    if os.environ.get("HF_HUB_OFFLINE") == "1":
        raise FileNotFoundError(
            f"no local copy at {target} and the hub is disabled — "
            f"run `make fetch` where there is network access")
    ui.warn(f"no local copy at {target}, using the hub")
    return str(config.require("model.teacher"))


def load_tokenizer(config):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_source(config))


# ==============================
# ===  Version shims         ===
# ==============================

def _from_pretrained(cls, source: str, dtype):
    """`torch_dtype` was renamed to `dtype`; accept whichever this build takes."""
    try:
        return cls.from_pretrained(source, dtype=dtype)
    except TypeError:
        return cls.from_pretrained(source, torch_dtype=dtype)


# ==============================
# ===  Teacher and student   ===
# ==============================

def load_teacher(config, device: str, dtype):
    from transformers import AutoModelForCausalLM

    model = _from_pretrained(AutoModelForCausalLM, model_source(config), dtype)
    model.to(device).eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model


def load_student(config, device: str):
    """Full precision master weights, binary forward, blocks only trainable."""
    import torch
    from transformers import AutoModelForCausalLM

    from .binary import binarise_model

    model = _from_pretrained(
        AutoModelForCausalLM, model_source(config), torch.float32)
    replaced = binarise_model(
        model,
        group=int(config.get("model.group", 128)),
        clip=float(config.get("model.ste_clip", 1.0)))

    if config.get("model.freeze_full_precision", True):
        binary_names = set()
        for name, module in model.named_modules():
            if module.__class__.__name__ == "BinaryLinear":
                binary_names.add(name)
        for name, param in model.named_parameters():
            owner = name.rsplit(".", 1)[0]
            param.requires_grad_(owner in binary_names)

    model.to(device)
    from .config import device_type

    if config.get("train.gradient_checkpointing", True) \
            and device_type(device) != "cpu":
        # Reentrant checkpointing silently drops gradients when the block input
        # does not require one, which is exactly the case here: the embedding is
        # frozen. Ask for the non-reentrant path, and make the input require a
        # gradient anyway on builds that do not take the argument.
        try:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False})
        except TypeError:
            model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        model.config.use_cache = False
    return model, replaced


def load_naive_student(config, device: str, dtype):
    """The untrained baseline: same sign rule, mean magnitude, no gradients."""
    import torch
    from transformers import AutoModelForCausalLM

    from .binary import naive_binarise_model

    model = _from_pretrained(
        AutoModelForCausalLM, model_source(config), torch.float32)
    count = naive_binarise_model(model, group=int(config.get("model.group", 128)))
    model.to(device=device, dtype=dtype).eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model, count


def describe(model, replaced: list[str]) -> None:
    from .binary import binary_parameters

    total = sum(p.numel() for p in model.parameters())
    binary, bits = binary_parameters(model)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    ui.field("parameters", f"{total / 1e6:.1f}M")
    ui.field("binary layers", f"{len(replaced)} ({binary / 1e6:.1f}M weights)")
    ui.field("bits per weight", f"{bits:.3f}")
    ui.field("trainable tensors", f"{trainable / 1e6:.1f}M values")

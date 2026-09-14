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
import time
from pathlib import Path

from . import ui

KEEP = ["*.json", "*.txt", "*.model", "*.safetensors", "tokenizer*"]
DROP = ["*.pth", "*.bin", "*.h5", "*.msgpack", "*.onnx", "*consolidated*"]


# ==============================
# ===  Local copy            ===
# ==============================

def use_local_cache(config) -> Path | None:
    """Point the hub at a cache inside the project.

    Must run before huggingface_hub is imported: it reads these once. An
    HF_HOME the user set themselves is left alone - that is a deliberate
    choice, usually a shared cache someone does not want duplicated.
    """
    if os.environ.get("HF_HOME"):
        return None
    try:
        cache = config.path("paths.cache")
    except Exception:  # noqa: BLE001 - an old config file has no such key
        return None
    cache.mkdir(parents=True, exist_ok=True)
    if not os.access(cache, os.W_OK):
        raise PermissionError(f"cannot write to {cache}; set HF_HOME to a "
                              f"directory you own")
    os.environ["HF_HOME"] = str(cache)
    # Older releases read these instead; setting them costs nothing.
    os.environ.setdefault("HF_HUB_CACHE", str(cache / "hub"))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(cache / "hub"))
    # The hub's own read timeout is ten seconds. On a slow or shared link that
    # expires mid-file, and the retry starts the same file again: the bar sits
    # at the same place indefinitely instead of reporting anything.
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT",
                          str(int(float(config.get("data.timeout", 120)))))
    return cache


def local_dir(config) -> Path:
    name = str(config.require("model.teacher")).replace("/", "_")
    return config.path("paths.models") / name


def _log_remote_files(repo: str) -> None:
    """What the hub says should arrive, so a short download is obvious."""
    try:
        from huggingface_hub import HfApi

        info = HfApi().model_info(repo, files_metadata=True)
        for item in info.siblings or []:
            size = getattr(item, "size", None)
            room = f"{size / 2 ** 20:.1f} MB" if size else "size unknown"
            ui.detail(f"  remote {item.rfilename} {room}")
    except Exception as problem:  # noqa: BLE001 - a listing is a nicety
        ui.detail(f"could not list remote files: {problem}")


def folder_size(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def fetch_model(config) -> Path:
    """Download the checkpoint into the package. Safe to re-run."""
    from huggingface_hub import snapshot_download

    repo = str(config.require("model.teacher"))
    target = local_dir(config)
    target.mkdir(parents=True, exist_ok=True)
    attempts = max(1, int(config.get("data.retries", 5)))

    workers = int(config.get("data.workers", 4))
    for attempt in range(1, attempts + 1):
        ui.step(f"downloading {repo}")
        # The progress bar reaches 100% when the bytes are in; hashing them and
        # laying them out in the folder happens after that, with nothing on
        # screen. On a slow disk it looks like a hang, so say what is going on.
        ui.say("      after the bar fills, files are checked and unpacked - "
               "this part is silent", "overlay")
        ui.detail(f"attempt {attempt}/{attempts}, {workers} workers, "
                  f"into {target}")
        ui.detail(f"on disk before: {folder_size(target) / 2 ** 20:.1f} MB")
        started = time.time()
        try:
            _log_remote_files(repo)
            snapshot_download(repo, local_dir=str(target),
                              allow_patterns=KEEP, ignore_patterns=DROP,
                              max_workers=workers)
            ui.detail(f"download returned after {time.time() - started:.0f}s")
            for item in sorted(target.rglob("*")):
                if item.is_file():
                    ui.detail(f"  {item.relative_to(target)} "
                              f"{item.stat().st_size / 2 ** 20:.1f} MB")
            break
        except (OSError, EOFError) as problem:
            ui.detail(f"failed after {time.time() - started:.0f}s: "
                      f"{type(problem).__name__}: {problem}")
            ui.detail(f"on disk now: {folder_size(target) / 2 ** 20:.1f} MB")
            if attempt == attempts:
                raise
            delay = min(60.0, 5.0 * 2 ** (attempt - 1))
            ui.warn(f"{problem} — retrying in {delay:.0f}s "
                    f"({attempt} of {attempts})")
            time.sleep(delay)

    ui.good(f"model in {target} ({folder_size(target) / 2 ** 30:.2f} GB)")
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


def tokenizer_cores(config) -> int:
    """How many cores the tokeniser may use.

    Its thread pool takes every core on the machine by default, which is the
    wrong thing to do on a shared one: it is not our machine to fill. Half is
    a decent neighbour and still an order of magnitude more than one.
    """
    asked = int(config.get("data.cpu_cores", 0))
    if asked > 0:
        return asked
    return max(1, (os.cpu_count() or 2) // 2)


def load_tokenizer(config):
    # The Rust tokeniser splits a batch across cores by itself, but
    # transformers silences that the moment it suspects a fork, and then
    # tokenising a terabyte runs on one core. Nothing here forks.
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
    # Read once, when the pool is built, so it has to be set before the
    # tokeniser is imported - not after.
    os.environ.setdefault("RAYON_NUM_THREADS", str(tokenizer_cores(config)))

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_source(config))
    # A document longer than the context window is not a problem here: it is
    # tokenised whole and then cut into windows. The warning says otherwise,
    # once per long document, straight through the progress bar.
    tokenizer.model_max_length = int(1e12)
    return tokenizer


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
    """The model being trained.

    With `model.binary` the linear layers inside the blocks are replaced by
    one-bit ones and only those train. Without it nothing is replaced and this
    is an ordinary fine-tune — same corpus, same loop, same checkpoints.
    """
    import torch
    from transformers import AutoModelForCausalLM

    from .binary import binarise_model

    model = _from_pretrained(
        AutoModelForCausalLM, model_source(config), torch.float32)

    if not config.get("model.binary", True):
        model.to(device)
        _enable_checkpointing(config, model, device)
        return model, []

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
    _enable_checkpointing(config, model, device)
    return model, replaced


def _enable_checkpointing(config, model, device: str) -> None:
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

    if not replaced:
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        ui.field("parameters", f"{total / 1e6:.1f}M")
        ui.field("mode", "full precision fine-tune")
        ui.field("trainable", f"{trainable / 1e6:.1f}M values")
        return

    total = sum(p.numel() for p in model.parameters())
    binary, bits = binary_parameters(model)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    ui.field("parameters", f"{total / 1e6:.1f}M")
    ui.field("binary layers", f"{len(replaced)} ({binary / 1e6:.1f}M weights)")
    ui.field("bits per weight", f"{bits:.3f}")
    ui.field("trainable tensors", f"{trainable / 1e6:.1f}M values")

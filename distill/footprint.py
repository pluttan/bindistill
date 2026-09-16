"""What the compression is actually worth, end to end.

Sixteen bits to one is the ratio for a weight inside a block, and quoting it for
the model is how a paper earns the reviewer it deserves. A model is not only its
blocks: the embedding table, the output head and the norms stay in full
precision, and in a small model they are most of what is left. The number that
belongs in a paper is the one measured over everything that has to be shipped
and held in memory.

Speed is measured for the same reason and reported the same way. This package
computes with the sign of a weight but multiplies in bfloat16, because a kernel
that multiplies by packed bits is not part of it. Storage shrinks; arithmetic
does not, and one is not evidence for the other.

Signed: pluttan
"""

from __future__ import annotations

import json
from pathlib import Path

from . import ui


def group_bytes(count: int, group: int) -> int:
    """Bytes a binarised matrix of `count` weights needs, scales included.

    One bit per weight, packed eight to a byte, plus one half-precision scale
    per group. Partial bytes at the end of a row are rounded up because that is
    what a file does.
    """
    return (count + 7) // 8 + (count + group - 1) // group * 2


def split_model(model) -> dict:
    """Weights of a binarised model, divided into what shrinks and what does not.

    Returns counts, not bytes, so the same split can be priced at any width.
    """
    binary = full = 0
    scales = 0
    group = 0
    parts = {}

    owners = {}
    for name, module in model.named_modules():
        if module.__class__.__name__ == "BinaryLinear":
            owners[name] = module

    for name, param in model.named_parameters():
        owner = name.rsplit(".", 1)[0]
        module = owners.get(owner)
        leaf = name.rsplit(".", 1)[-1]
        if module is not None and leaf == "master":
            binary += param.numel()
            group = group or int(getattr(module, "group", 128))
            continue
        if module is not None and leaf == "log_scale":
            scales += param.numel()
            continue
        full += param.numel()
        # Where the full-precision remainder sits, so the reader can see why
        # the end-to-end ratio is not the per-weight one.
        head = name.split(".")[0] if "." in name else name
        key = ("embedding" if "embed" in name else
               "head" if "lm_head" in name else
               "norm" if "norm" in name else head)
        parts[key] = parts.get(key, 0) + param.numel()

    return {"binary_weights": binary, "full_weights": full,
            "scale_values": scales, "group": group or 128, "parts": parts}


def price(split: dict, dense_bits: int = 16) -> dict:
    """Bytes for the split, stored densely and stored packed."""
    binary, full = split["binary_weights"], split["full_weights"]
    dense = (binary + full) * dense_bits // 8
    packed = group_bytes(binary, split["group"]) + full * 2
    return {
        "dense_bytes": dense,
        "packed_bytes": packed,
        "blocks_only_ratio": (binary * dense_bits / 8) /
                             max(1, group_bytes(binary, split["group"])),
        "whole_model_ratio": dense / max(1, packed),
        "full_precision_share": full / max(1, binary + full),
    }


def on_disk(folder: Path) -> dict:
    """What the exported folder actually weighs, by file.

    The check that the arithmetic above did not drift from what gets written:
    `binary.safetensors` plus the full-precision tensors is the model that
    would be shipped, and `model.safetensors` is the dense copy kept for
    loading it into an unmodified transformers.
    """
    sizes = {p.name: p.stat().st_size for p in folder.glob("*")
             if p.is_file()}
    return sizes


# ==============================
# ===  Speed                 ===
# ==============================

def generation_speed(model, tokenizer, device: str, tokens: int = 64,
                     runs: int = 3) -> dict:
    """Tokens per second when generating, and the memory it takes.

    Greedy, one sequence, short prompt: this measures the cost of a step, not
    the quality of anything. The first run is thrown away because it pays for
    the kernels being chosen.
    """
    import time

    import torch

    prompt = tokenizer("The history of the steam engine begins",
                       return_tensors="pt").to(device)
    if torch.cuda.is_available() and device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(device)

    model.eval()
    best = 0.0
    with torch.no_grad():
        for attempt in range(runs + 1):
            if torch.cuda.is_available() and device.startswith("cuda"):
                torch.cuda.synchronize(device)
            started = time.perf_counter()
            model.generate(**prompt, max_new_tokens=tokens, do_sample=False,
                           pad_token_id=tokenizer.eos_token_id)
            if torch.cuda.is_available() and device.startswith("cuda"):
                torch.cuda.synchronize(device)
            spent = time.perf_counter() - started
            if attempt == 0:
                continue  # warm-up
            best = max(best, tokens / max(1e-6, spent))

    peak = 0
    if torch.cuda.is_available() and device.startswith("cuda"):
        peak = torch.cuda.max_memory_allocated(device)
    return {"tokens_per_second": round(best, 1),
            "peak_bytes": int(peak)}


# ==============================
# ===  The report            ===
# ==============================

def megabytes(value: float) -> float:
    return round(value / 2 ** 20, 1)


def run(config, checkpoint: Path | None = None, speed: bool = True) -> dict:
    """Size and speed of teacher and student, side by side."""
    import torch

    from . import models
    from .checkpoint import load_into
    from .config import resolve_device

    device = resolve_device(str(config.get("run.device", "auto")))
    ui.head("Footprint")
    ui.field("device", device)
    ui.field("teacher", config.require("model.teacher"))

    tokenizer = models.load_tokenizer(config)
    report = {}

    student, _ = models.load_student(config, device)
    if checkpoint is not None:
        report["step"] = load_into(student, None, checkpoint)
    split = split_model(student)
    sizes = price(split)
    report["split"] = split
    report["sizes"] = sizes

    ui.say()
    ui.field("weights in blocks, binarised", f"{split['binary_weights'] / 1e6:.1f}M")
    ui.field("weights left in full precision", f"{split['full_weights'] / 1e6:.1f}M")
    for part, count in sorted(split["parts"].items(), key=lambda kv: -kv[1]):
        ui.detail(f"{part}: {count / 1e6:.1f}M")

    ui.say()
    ui.table(
        [("dense, 16 bits everywhere", f"{megabytes(sizes['dense_bytes'])}"),
         ("packed, one bit in blocks", f"{megabytes(sizes['packed_bytes'])}")],
        ("storage", "MB"))
    ui.say()
    ui.field("ratio over the blocks alone", f"x{sizes['blocks_only_ratio']:.2f}")
    ui.field("ratio over the whole model", f"x{sizes['whole_model_ratio']:.2f}")
    ui.field("share left in full precision",
             f"{sizes['full_precision_share'] * 100:.1f}%")
    if sizes["whole_model_ratio"] < sizes["blocks_only_ratio"] / 2:
        ui.warn("the full-precision remainder dominates at this model size; "
                "the end-to-end ratio is the one to quote")

    if speed:
        ui.say()
        student_speed = generation_speed(student, tokenizer, device)
        report["student_speed"] = student_speed
        del student
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        teacher = models.load_teacher(config, device, torch.bfloat16)
        teacher_speed = generation_speed(teacher, tokenizer, device)
        report["teacher_speed"] = teacher_speed
        del teacher
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        ui.table(
            [("teacher", f"{teacher_speed['tokens_per_second']}",
              f"{megabytes(teacher_speed['peak_bytes'])}"),
             ("one-bit student", f"{student_speed['tokens_per_second']}",
              f"{megabytes(student_speed['peak_bytes'])}")],
            ("model", "tokens/s", "peak MB"))
        ui.say()
        # Saying this out loud is the point. The storage number is real; a
        # reader who assumes it carries over to speed has been misled, and the
        # measurement above is what stops that.
        ui.detail("generation runs in bfloat16 for both rows: the weights are "
                  "one bit in storage, the multiplication is not. Speed here "
                  "measures this implementation, not the representation.")

    room = config.run_dir()
    room.mkdir(parents=True, exist_ok=True)
    (room / "footprint.json").write_text(json.dumps(report, indent=2))
    ui.good(f"written to {room / 'footprint.json'}")
    return report

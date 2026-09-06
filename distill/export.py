"""Writing a finished run out as a model directory and as packed bits.

Two files come out of this. `model.safetensors` holds the weights the way the
network uses them — sign times scale, in bfloat16 — so the result loads in
anything that reads the teacher's architecture and can be pointed at directly by
bitprobe's `compare` and `flips`. `binary.safetensors` holds the same thing as it
would actually be stored: one bit per weight, packed eight to a byte, with one
scale per group. The second file is what the size claim rests on.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from . import checkpoint, models, ui
from .binary import BinaryLinear, bits_per_weight


def run(config, source: Path, destination: Path | None = None) -> Path:
    import numpy as np
    import torch
    from safetensors.torch import save_file

    destination = Path(destination) if destination else \
        config.run_dir() / "export"
    destination.mkdir(parents=True, exist_ok=True)

    ui.head("Export")
    ui.field("checkpoint", source.name)

    student, replaced = models.load_student(config, "cpu")
    step = checkpoint.load_into(student, None, source)
    ui.field("step", step)

    dense: dict[str, torch.Tensor] = {}
    packed: dict[str, torch.Tensor] = {}
    dense_bytes = packed_bytes = 0

    with torch.no_grad():
        for name, module in student.named_modules():
            if not isinstance(module, BinaryLinear):
                continue
            signs, scales = module.packed()
            weight = torch.where(signs, 1.0, -1.0).view(
                module.out_features, -1, module.group) * scales.unsqueeze(-1)
            weight = weight.view(module.out_features, module.in_features)

            dense[f"{name}.weight"] = weight.to(torch.bfloat16)
            bits = np.packbits(signs.numpy().astype(np.uint8), axis=-1)
            packed[f"{name}.signs"] = torch.from_numpy(bits)
            packed[f"{name}.scales"] = scales.to(torch.float16)
            packed_bytes += bits.nbytes + scales.numel() * 2
            dense_bytes += weight.numel() * 2

    # Everything that was never binarised still has to travel with the model.
    for name, param in student.named_parameters():
        owner = name.rsplit(".", 1)[0]
        module = dict(student.named_modules()).get(owner)
        if isinstance(module, BinaryLinear):
            continue
        dense[name] = param.detach().to(torch.bfloat16)

    save_file(dense, str(destination / "model.safetensors"))
    save_file(packed, str(destination / "binary.safetensors"))

    source_dir = Path(models.model_source(config))
    if source_dir.exists():
        for extra in ("config.json", "generation_config.json",
                      "tokenizer.json", "tokenizer_config.json",
                      "special_tokens_map.json", "vocab.json", "merges.txt"):
            candidate = source_dir / extra
            if candidate.exists():
                shutil.copy2(candidate, destination / extra)

    average = sum(bits_per_weight(m.group) for m in student.modules()
                  if isinstance(m, BinaryLinear))
    count = sum(1 for m in student.modules() if isinstance(m, BinaryLinear))
    summary = {
        "checkpoint": str(source),
        "step": step,
        "binary_layers": count,
        "bits_per_weight": round(average / max(1, count), 4),
        "dense_megabytes": round(dense_bytes / 2 ** 20, 1),
        "packed_megabytes": round(packed_bytes / 2 ** 20, 1),
        "teacher": config.require("model.teacher"),
    }
    (destination / "export.json").write_text(json.dumps(summary, indent=2))

    ui.field("binary layers", count)
    ui.field("bits per weight", summary["bits_per_weight"])
    ui.field("dense bf16", f"{summary['dense_megabytes']} MB")
    ui.field("packed bits", f"{summary['packed_megabytes']} MB")
    ui.good(f"written to {destination}")
    return destination

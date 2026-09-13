"""What the training left behind in the weights.

The claim that a released one-bit model was trained rather than quantised rests
on a signature in the weights: signs disagree with the full-precision base, the
disagreement is graded by magnitude — small weights cross zero easily, large ones
do not — and the learned scales end up well above the mean absolute value that a
formula would give.

This measures the same three things on our own result. If a model trained here
carries the signature, the argument stops being an inference about someone else's
checkpoint and becomes a reproduction.
"""

from __future__ import annotations

import json
from pathlib import Path

from . import checkpoint, models, ui

DECILES = 10


def _teacher_weights(config) -> dict:
    """The full-precision weights the signs started from."""
    import torch
    from transformers import AutoModelForCausalLM

    model = models._from_pretrained(
        AutoModelForCausalLM, models.model_source(config), torch.float32)
    return {name: param.detach()
            for name, param in model.named_parameters()
            if name.endswith(".weight") and param.dim() == 2}


def run(config, source: Path) -> dict:
    import torch

    ui.head("Weight signature")
    ui.field("checkpoint", source.name)

    payload = torch.load(source, map_location="cpu", weights_only=False)
    stored = payload["parameters"]
    ui.field("step", payload.get("step", 0))
    ui.field("tokens", f"{payload.get('tokens', 0) / 1e6:.0f}M")

    teacher = _teacher_weights(config)

    flipped_total = counted_total = 0
    scale_ratios: list[float] = []
    bucket_flipped = [0] * DECILES
    bucket_counted = [0] * DECILES

    for name, master in stored.items():
        if not name.endswith(".master"):
            continue
        original = teacher.get(name[: -len(".master")] + ".weight")
        if original is None or original.shape != master.shape:
            continue

        base = original.float()
        trained = master.float()
        flips = (torch.sign(base) != torch.sign(trained))
        flipped_total += int(flips.sum())
        counted_total += flips.numel()

        # Deciles of the original magnitude, per matrix: the profile is about
        # where in the distribution a sign moved, not about absolute size.
        magnitude = base.abs().flatten()
        order = torch.argsort(magnitude)
        flat = flips.flatten()[order]
        edges = torch.linspace(0, len(order), DECILES + 1).long()
        for index in range(DECILES):
            piece = flat[edges[index]: edges[index + 1]]
            bucket_flipped[index] += int(piece.sum())
            bucket_counted[index] += piece.numel()

        log_scale = stored.get(name[: -len(".master")] + ".log_scale")
        if log_scale is not None:
            group = base.shape[1] // log_scale.shape[1]
            naive = base.view(base.shape[0], -1, group).abs().mean(dim=2)
            ratio = (log_scale.float().exp() / naive.clamp_min(1e-8))
            scale_ratios.append(float(ratio.mean()))

    share = flipped_total / max(1, counted_total)
    ui.say()
    ui.field("matrices measured", len(scale_ratios))
    ui.field("signs flipped", f"{share * 100:.1f}%", "peach")
    if scale_ratios:
        mean_ratio = sum(scale_ratios) / len(scale_ratios)
        ui.field("scale vs mean|w|", f"x{mean_ratio:.2f}", "peach")

    ui.say()
    rows = []
    for index in range(DECILES):
        part = bucket_flipped[index] / max(1, bucket_counted[index])
        rows.append((f"{index + 1}", f"{part * 100:.1f}%"))
    ui.table(rows, ("decile of |w|", "signs flipped"))
    ui.say("  decile 1 is the smallest weights, 10 the largest", "overlay")

    smallest = bucket_flipped[0] / max(1, bucket_counted[0])
    largest = bucket_flipped[-1] / max(1, bucket_counted[-1])
    ui.say()
    if smallest > largest * 2:
        ui.good("graded by magnitude: the signature of training, not rounding")
    else:
        ui.warn("flips are not graded by magnitude — this does not look like "
                "what training leaves behind")

    summary = {
        "step": payload.get("step", 0),
        "tokens": payload.get("tokens", 0),
        "flipped_share": round(share, 5),
        "flipped_smallest_decile": round(smallest, 5),
        "flipped_largest_decile": round(largest, 5),
        "scale_vs_naive": round(sum(scale_ratios) / len(scale_ratios), 4)
        if scale_ratios else None,
        "deciles": [round(bucket_flipped[i] / max(1, bucket_counted[i]), 5)
                    for i in range(DECILES)],
    }
    out = source.parent / "signature.json"
    out.write_text(json.dumps(summary, indent=2))
    ui.good(f"written to {out}")
    return summary

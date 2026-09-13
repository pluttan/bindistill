"""Measuring the student against the teacher and against doing nothing.

A perplexity on its own says nothing about whether the training worked. The
number that matters is where the trained model sits between two fixed points:
the full-precision teacher, and the same weights binarised by the sign rule with
no training at all. If the trained model does not clear the second, the run
failed regardless of how good its loss curve looked.

Three quantities per model: perplexity on held-out text, how often the next
token agrees with the teacher's, and the average divergence between the two
distributions.
"""

from __future__ import annotations

import math

from . import ui


# ==============================
# ===  One model             ===
# ==============================

def measure(model, teacher, inputs, targets, device, chunk: int = 256) -> dict:
    """Run both models over the held-out windows and accumulate the sums.

    Logits are the largest thing in this function — a full vocabulary times a
    full window — so they are reduced a slice at a time rather than cast to
    float32 all at once.
    """
    import torch
    import torch.nn.functional as F

    nll = teacher_nll = agree = divergence = 0.0
    counted = 0

    model.eval()
    with torch.no_grad():
        for index in range(inputs.shape[0]):
            ids = inputs[index: index + 1].to(device)
            gold = targets[index: index + 1].to(device)

            student_logits = model(ids).logits
            teacher_logits = teacher(ids).logits

            length = ids.shape[1]
            for start in range(0, length, chunk):
                stop = min(length, start + chunk)
                s = student_logits[:, start:stop].float()
                t = teacher_logits[:, start:stop].float()
                g = gold[:, start:stop]

                nll += F.cross_entropy(
                    s.flatten(0, 1), g.flatten(), reduction="sum").item()
                teacher_nll += F.cross_entropy(
                    t.flatten(0, 1), g.flatten(), reduction="sum").item()
                agree += (s.argmax(-1) == t.argmax(-1)).sum().item()

                s_log = F.log_softmax(s, dim=-1)
                t_log = F.log_softmax(t, dim=-1)
                divergence += (t_log.exp() * (t_log - s_log)).sum().item()
                counted += g.numel()

            del student_logits, teacher_logits

    return {
        "tokens": counted,
        "perplexity": math.exp(min(30.0, nll / max(1, counted))),
        "teacher_perplexity": math.exp(min(30.0, teacher_nll / max(1, counted))),
        "agreement": agree / max(1, counted),
        "kl": divergence / max(1, counted),
    }


# ==============================
# ===  The comparison        ===
# ==============================

def run(config, checkpoint=None) -> dict:
    """Teacher, untrained binarisation, and the trained student side by side."""
    import torch

    from . import data, models
    from .config import device_type, resolve_device, resolve_dtype

    device = resolve_device(str(config.get("run.device", "auto")))
    dtype = resolve_dtype(str(config.get("train.dtype", "bfloat16")), device)
    chunk = int(config.get("train.loss_chunk", 256))

    ui.head("Evaluation")
    ui.field("device", device)
    ui.field("teacher", config.require("model.teacher"))

    inputs, targets = data.held_out_windows(config)
    ui.field("held-out windows", f"{inputs.shape[0]} x {inputs.shape[1]} tokens")

    teacher = models.load_teacher(config, device, dtype)
    rows, results = [], {}

    # The teacher measured against itself: perplexity real, agreement 1 by
    # construction. It is the ceiling every other row is read against.
    baseline = measure(teacher, teacher, inputs, targets, device, chunk)
    results["teacher"] = baseline
    rows.append(("teacher (full precision)", f"{baseline['perplexity']:.2f}",
                 "1.000", "0.0000"))

    if config.get("eval.baselines", True) and config.get("model.binary", True):
        naive, count = models.load_naive_student(config, device, dtype)
        scores = measure(naive, teacher, inputs, targets, device, chunk)
        results["naive"] = scores
        rows.append((f"naive binarisation ({count} layers)",
                     f"{scores['perplexity']:.2f}",
                     f"{scores['agreement']:.3f}", f"{scores['kl']:.4f}"))
        del naive
        if device_type(device) == "cuda":
            torch.cuda.empty_cache()

    if checkpoint is not None:
        student, replaced = models.load_student(config, device)
        from .checkpoint import load_into
        step = load_into(student, None, checkpoint)
        scores = measure(student, teacher, inputs, targets, device, chunk)
        results["trained"] = scores
        rows.append((f"trained student (step {step})",
                     f"{scores['perplexity']:.2f}",
                     f"{scores['agreement']:.3f}", f"{scores['kl']:.4f}"))
        del student

    ui.say()
    ui.table(rows, ("model", "perplexity", "agree@1", "KL"))

    if "naive" in results and "trained" in results:
        ui.say()
        gain = results["naive"]["perplexity"] / results["trained"]["perplexity"]
        if gain > 1.05:
            ui.good(f"training beats the untrained binarisation by {gain:.2f}x")
        elif gain > 0.95:
            ui.warn("training has not moved past the untrained binarisation yet")
        else:
            ui.fail("the trained model is worse than doing nothing — "
                    "check the learning rate and the straight-through window")
    return results

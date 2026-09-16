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
    # Beating the untrained binarisation says only that training happened. The
    # question that matters is how much of the teacher survived, and a single
    # top-1 rate answers it badly: it counts a disagreement on a token the
    # teacher itself was unsure about the same as one where it was certain.
    # These four separate the two cases.
    in_top5 = supported = sure = sure_agree = 0.0
    counted = 0

    model.eval()
    with torch.no_grad():
        for index in range(inputs.shape[0]):
            ids = inputs[index: index + 1].to(device)
            gold = targets[index: index + 1].to(device)

            student_logits = model(ids).logits
            # Plain fine-tuning has no teacher. Perplexity is the model's own
            # score and is measurable either way; the comparison columns are
            # simply left out rather than the whole check being skipped.
            teacher_logits = teacher(ids).logits if teacher is not None else None

            length = ids.shape[1]
            for start in range(0, length, chunk):
                stop = min(length, start + chunk)
                s = student_logits[:, start:stop].float()
                g = gold[:, start:stop]

                nll += F.cross_entropy(
                    s.flatten(0, 1), g.flatten(), reduction="sum").item()
                counted += g.numel()

                if teacher_logits is None:
                    continue

                t = teacher_logits[:, start:stop].float()
                teacher_nll += F.cross_entropy(
                    t.flatten(0, 1), g.flatten(), reduction="sum").item()
                choice = t.argmax(-1)
                same = s.argmax(-1) == choice
                agree += same.sum().item()

                s_log = F.log_softmax(s, dim=-1)
                t_log = F.log_softmax(t, dim=-1)
                divergence += (t_log.exp() * (t_log - s_log)).sum().item()

                # Where the student does not repeat the teacher's choice, it
                # still matters whether that choice stayed near the top or was
                # dropped altogether.
                top5 = s.topk(5, dim=-1).indices
                in_top5 += (top5 == choice.unsqueeze(-1)).any(-1).sum().item()

                # How much probability the student puts on what the teacher
                # picked: a near miss and a flat refusal both count as one
                # disagreement above, and they are not the same thing.
                picked = choice.unsqueeze(-1)
                supported += s_log.gather(-1, picked).exp().sum().item()

                # Disagreeing where the teacher hesitated is cheap. Disagreeing
                # where it was certain is the damage worth reporting.
                certain = t_log.gather(-1, picked).exp().squeeze(-1) >= 0.9
                sure += certain.sum().item()
                sure_agree += (same & certain).sum().item()

            del student_logits, teacher_logits

    scores = {
        "tokens": counted,
        "perplexity": math.exp(min(30.0, nll / max(1, counted))),
    }
    if teacher is not None:
        scores.update({
            "teacher_perplexity": math.exp(min(30.0,
                                               teacher_nll / max(1, counted))),
            "agreement": agree / max(1, counted),
            "agreement_top5": in_top5 / max(1, counted),
            "teacher_choice_probability": supported / max(1, counted),
            "agreement_when_certain": sure_agree / max(1.0, sure),
            "certain_tokens": sure / max(1, counted),
            "kl": divergence / max(1, counted),
        })
    return scores


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
                 "1.000", "1.000", "1.000", "0.0000"))

    if config.get("eval.baselines", True) and config.get("model.binary", True):
        naive, count = models.load_naive_student(config, device, dtype)
        scores = measure(naive, teacher, inputs, targets, device, chunk)
        results["naive"] = scores
        rows.append((f"naive binarisation ({count} layers)",
                     f"{scores['perplexity']:.2f}",
                     f"{scores['agreement']:.3f}",
                     f"{scores['agreement_top5']:.3f}",
                     f"{scores['agreement_when_certain']:.3f}",
                     f"{scores['kl']:.4f}"))
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
                     f"{scores['agreement']:.3f}",
                     f"{scores['agreement_top5']:.3f}",
                     f"{scores['agreement_when_certain']:.3f}",
                     f"{scores['kl']:.4f}"))
        del student

    ui.say()
    ui.table(rows, ("model", "perplexity", "agree@1", "agree@5",
                    "agree|sure", "KL"))

    if "trained" in results:
        ui.say()
        trained = results["trained"]
        # What the run is for: how much of the teacher is left. Beating the
        # untrained binarisation is only the check that training happened at
        # all, so it is reported second and briefly.
        ratio = trained["perplexity"] / max(1e-9, baseline["perplexity"])
        ui.field("perplexity vs. teacher", f"x{ratio:.3f}")
        ui.field("teacher's choice repeated", f"{trained['agreement']:.3f}")
        ui.field("... kept in the top five", f"{trained['agreement_top5']:.3f}")
        ui.field("... where the teacher was sure",
                 f"{trained['agreement_when_certain']:.3f} "
                 f"(on {trained['certain_tokens'] * 100:.0f}% of tokens)")
        ui.field("probability on that choice",
                 f"{trained['teacher_choice_probability']:.3f}")

    if "naive" in results and "trained" in results:
        gain = results["naive"]["perplexity"] / results["trained"]["perplexity"]
        ui.say()
        if gain > 1.05:
            ui.detail(f"control: training beats the untrained binarisation "
                      f"by {gain:.0f}x")
        elif gain > 0.95:
            ui.warn("training has not moved past the untrained binarisation yet")
        else:
            ui.fail("the trained model is worse than doing nothing — "
                    "check the learning rate and the straight-through window")
    return results

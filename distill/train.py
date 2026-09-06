"""The training loop: match the teacher's distribution, one bit per weight.

The loss is the teacher's next-token distribution, anchored by the real token so
the student cannot drift somewhere the teacher is merely confident and wrong.
Both parts are computed a slice of the sequence at a time: at a vocabulary of a
hundred and fifty thousand, a full window of float32 log-probabilities is larger
than the model, and holding three of them at once is what makes this fall over
on a card that should have been big enough.

A block trained against its own teacher cannot see what the rest of the network
does with its output, which is why the per-block experiments plateaued. The loss
here is on the logits, so the whole stack is in the graph and the error has
nowhere to accumulate.
"""

from __future__ import annotations

import math
import os
import time
from pathlib import Path

from . import checkpoint, data, models, ui
from .binary import split_parameters
from .config import resolve_device, resolve_dtype


class _nullcontext:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False


# ==============================
# ===  Loss                  ===
# ==============================

def distil_backward(student_logits, teacher_logits, targets, alpha: float,
                    temperature: float, chunk: int) -> tuple[float, float]:
    """Accumulate the gradient into `student_logits` a slice at a time.

    Each slice gets its own leaf so the float32 softmax it needs is freed before
    the next one is built; the collected gradient is then pushed through the
    model in a single backward pass.
    """
    import torch
    import torch.nn.functional as F

    detached = student_logits.detach()
    grad = torch.zeros_like(detached)
    length = detached.shape[1]
    total = detached.shape[0] * length
    kl_sum = ce_sum = 0.0

    for start in range(0, length, chunk):
        stop = min(length, start + chunk)
        piece = detached[:, start:stop].clone().requires_grad_(True)
        with torch.no_grad():
            reference = teacher_logits[:, start:stop].float()
        gold = targets[:, start:stop]
        share = (stop - start) * detached.shape[0] / total

        student_log = F.log_softmax(piece.float() / temperature, dim=-1)
        teacher_log = F.log_softmax(reference / temperature, dim=-1)
        kl = (teacher_log.exp() * (teacher_log - student_log)).sum(-1).mean()
        kl = kl * temperature * temperature
        ce = F.cross_entropy(piece.float().flatten(0, 1), gold.flatten())

        loss = (alpha * kl + (1.0 - alpha) * ce) * share
        loss.backward()
        grad[:, start:stop] = piece.grad
        kl_sum += kl.item() * share
        ce_sum += ce.item() * share
        del piece, student_log, teacher_log, reference

    student_logits.backward(grad)
    return kl_sum, ce_sum


# ==============================
# ===  Schedule              ===
# ==============================

def learning_rate_factor(step: int, warmup: int, total: int,
                         floor: float = 0.1) -> float:
    if step < warmup:
        return (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
    return floor + (1.0 - floor) * cosine


# ==============================
# ===  Distributed           ===
# ==============================

def distributed_setup() -> tuple[int, int, bool]:
    """torchrun sets these; without it the run is plain single-process."""
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    if world > 1:
        import torch
        import torch.distributed as dist

        dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo")
        if torch.cuda.is_available():
            torch.cuda.set_device(rank % torch.cuda.device_count())
    return rank, world, world > 1


# ==============================
# ===  Loop                  ===
# ==============================

def run(config, resume: bool = True) -> Path:
    import torch

    rank, world, distributed = distributed_setup()
    lead = rank == 0

    device = resolve_device(str(config.get("run.device", "auto")))
    dtype = resolve_dtype(str(config.get("train.dtype", "bfloat16")), device)
    torch.manual_seed(int(config.get("run.seed", 0)) + rank)

    seq = int(config.require("train.seq"))
    micro = int(config.require("train.micro_batch"))
    accum = int(config.require("train.accum"))
    chunk = int(config.get("train.loss_chunk", 256))
    alpha = float(config.get("train.alpha", 0.9))
    temperature = float(config.get("train.temperature", 1.0))
    grad_clip = float(config.get("train.grad_clip", 1.0))
    per_step = micro * accum * seq * world
    budget = int(float(config.require("train.max_tokens")))
    total_steps = max(1, budget // per_step)

    run_dir = config.run_dir()
    if lead:
        run_dir.mkdir(parents=True, exist_ok=True)
        ui.head("Training")
        ui.field("preset", config.get("preset", "-"))
        ui.field("device", f"{device} x{world}" if world > 1 else device)
        ui.field("teacher", config.require("model.teacher"))
        ui.field("tokens per step", f"{per_step / 1e3:.1f}k")
        ui.field("steps planned", total_steps)
        ui.field("token budget", f"{budget / 1e9:.3f}B")
        ui.field("run directory", run_dir)

    teacher = models.load_teacher(config, device, dtype)
    student, replaced = models.load_student(config, device)
    if lead:
        models.describe(student, replaced)

    weights, scales = split_parameters(student)
    optimizer = torch.optim.AdamW(
        [{"params": weights, "lr": float(config.require("train.lr"))},
         {"params": scales, "lr": float(config.require("train.scale_lr"))}],
        betas=(0.9, 0.95), weight_decay=0.0)

    start_step, seen = 0, 0
    existing = checkpoint.latest(run_dir)
    if resume and existing is not None:
        start_step = checkpoint.load_into(student, optimizer, existing)
        seen = start_step * per_step
        if lead:
            ui.good(f"resumed from {existing.name} at step {start_step}")

    trainable = student
    if distributed:
        from torch.nn.parallel import DistributedDataParallel as DDP

        local = rank % max(1, torch.cuda.device_count())
        trainable = DDP(student, device_ids=[local] if device == "cuda" else None)

    stream = data.TokenWindows(config, rank=rank, world=world)
    if resume and existing is not None:
        stream.load_state(checkpoint.read_meta(existing)["extra"].get("stream"))
    metrics = checkpoint.MetricsLog(run_dir / "metrics.jsonl")
    base_lr = float(config.require("train.lr"))
    base_scale_lr = float(config.require("train.scale_lr"))
    warmup = int(config.get("train.warmup", 200))
    log_every = int(config.get("train.log_every", 10))
    eval_every = int(config.get("train.eval_every", 500))
    save_every = int(config.get("train.checkpoint_every", 250))
    keep = int(config.get("train.keep_checkpoints", 3))

    student.train()
    # Only CUDA gets mixed precision here: the student's master weights are
    # float32, and a float32 matmul on a GPU is both slower and larger than it
    # needs to be. On cpu and mps the cast buys nothing and costs correctness.
    autocast_on = device == "cuda" and dtype is not torch.float32
    started = time.time()
    progress = ui.Progress("training", total_steps) if lead else None

    for step in range(start_step, total_steps):
        factor = learning_rate_factor(step, warmup, total_steps)
        optimizer.param_groups[0]["lr"] = base_lr * factor
        optimizer.param_groups[1]["lr"] = base_scale_lr * factor
        optimizer.zero_grad(set_to_none=True)

        kl_total = ce_total = 0.0
        for micro_step in range(accum):
            ids, gold = stream.batch(micro)
            ids, gold = ids.to(device), gold.to(device)
            with torch.no_grad():
                reference = teacher(ids).logits
            # Only the last micro-batch needs the gradient all-reduce.
            last = micro_step == accum - 1
            sync = trainable.no_sync() if (distributed and not last) \
                else _nullcontext()
            amp = torch.autocast(device_type=device, dtype=dtype) \
                if autocast_on else _nullcontext()
            with sync:
                with amp:
                    logits = trainable(ids).logits
                kl, ce = distil_backward(logits, reference, gold, alpha,
                                         temperature, chunk)
            kl_total += kl / accum
            ce_total += ce / accum
            del logits, reference

        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(student.parameters(), grad_clip)
        optimizer.step()
        seen += per_step

        if lead and (step % log_every == 0 or step == total_steps - 1):
            record = {"step": step, "tokens": seen, "kl": kl_total,
                      "ce": ce_total, "ppl": math.exp(min(20.0, ce_total)),
                      "lr": base_lr * factor,
                      "elapsed": round(time.time() - started, 1)}
            metrics.write(record)
            progress.update(step - start_step)
            if step % (log_every * 10) == 0:
                ui.say()
                ui.step(f"step {step:>6}  kl {kl_total:7.4f}  "
                        f"ce {ce_total:6.3f}  ppl {record['ppl']:9.2f}  "
                        f"{seen / 1e6:.0f}M tokens")

        if lead and eval_every and step and step % eval_every == 0:
            score = quick_eval(config, student, teacher, device, chunk)
            metrics.write({"step": step, "tokens": seen, **score})
            ui.say()
            ui.field("held-out perplexity", f"{score['perplexity']:.2f}", "peach")
            ui.field("agreement with teacher", f"{score['agreement']:.3f}", "peach")

        if lead and save_every and step and step % save_every == 0:
            checkpoint.save(run_dir, student, optimizer, step, seen, keep,
                            extra={"preset": config.get("preset"),
                                   "stream": stream.state()})

    final = None
    if lead:
        if progress is not None:
            progress.done(f"{seen / 1e9:.3f}B tokens")
        final = checkpoint.save(run_dir, student, optimizer, total_steps, seen,
                                keep, extra={"preset": config.get("preset"),
                                             "stream": stream.state(),
                                             "final": True})
        ui.good(f"final checkpoint: {final}")

    if distributed:
        import torch.distributed as dist

        dist.destroy_process_group()
    return final or run_dir


def quick_eval(config, student, teacher, device: str, chunk: int) -> dict:
    """A cheap held-out check during training; the full table is `make eval`."""
    from . import evaluate

    windows = min(8, int(config.get("eval.windows", 32)))
    inputs, targets = data.held_out_windows(config, windows)
    was_training = student.training
    scores = evaluate.measure(student, teacher, inputs, targets, device, chunk)
    student.train(was_training)
    return scores

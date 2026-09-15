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
from .config import device_type, resolve_device, resolve_dtype


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
    """With `teacher_logits` of None this is plain next-token training."""
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
    # Accumulated on the device. Reading them per chunk with .item() stalls the
    # pipeline: the card finishes the chunk and then waits for the host before
    # starting the next one.
    kl_sum = torch.zeros((), device=detached.device, dtype=torch.float32)
    ce_sum = torch.zeros((), device=detached.device, dtype=torch.float32)

    for start in range(0, length, chunk):
        stop = min(length, start + chunk)
        # A slice of the detached logits shares its storage, so taking a leaf
        # here costs nothing; cloning it used to cost a full copy per chunk.
        piece = detached[:, start:stop].detach().requires_grad_(True)
        gold = targets[:, start:stop]
        rows = (stop - start) * detached.shape[0]
        share = rows / total

        # One float32 cast, reused. At a vocabulary this size each cast is
        # larger than the model, and the naive version made six of them.
        piece_float = piece.float()

        if teacher_logits is None:
            ce = F.cross_entropy(piece_float.flatten(0, 1), gold.flatten())
            kl = torch.zeros((), device=piece_float.device)
            student_log = teacher_log = None
            loss = ce * share
        else:
            student_log = F.log_softmax(piece_float / temperature, dim=-1)
            with torch.no_grad():
                teacher_log = F.log_softmax(
                    teacher_logits[:, start:stop].float() / temperature, dim=-1)

            kl = F.kl_div(student_log, teacher_log, reduction="sum",
                          log_target=True) / rows
            kl = kl * temperature * temperature
            if temperature == 1.0:
                # log_softmax is already there; cross_entropy would redo it.
                ce = F.nll_loss(student_log.flatten(0, 1), gold.flatten())
            else:
                ce = F.cross_entropy(piece_float.flatten(0, 1), gold.flatten())

            loss = (alpha * kl + (1.0 - alpha) * ce) * share
        loss.backward()
        grad[:, start:stop] = piece.grad
        kl_sum += kl.detach() * share
        ce_sum += ce.detach() * share
        del piece, piece_float, student_log, teacher_log

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

def distributed_setup() -> tuple[int, int, int, bool]:
    """torchrun sets these; without it the run is plain single-process."""
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    # LOCAL_RANK, not RANK: on one machine they agree, across machines only
    # the local one names a card on this box.
    local = int(os.environ.get("LOCAL_RANK", rank))
    if world > 1:
        import torch
        import torch.distributed as dist

        dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo")
        if torch.cuda.is_available():
            local %= torch.cuda.device_count()
            torch.cuda.set_device(local)
    return rank, local, world, world > 1


def rank_device(requested: str, local: int, distributed: bool) -> str:
    """Which card this process owns.

    Under torchrun every process must take a different card. A configured
    index says which card a single run should use, and honouring it here would
    put every process on that one card: two copies of the model on one card,
    the other cards idle, and most likely out of memory.
    """
    if not distributed or device_type(requested) != "cuda":
        return requested
    return f"cuda:{local}"


def cuda_unavailable_reason() -> str:
    """Why torch sees no card. The driver message only appears on first use."""
    import torch

    if torch.cuda.is_available():
        return ""
    if not getattr(torch.version, "cuda", None):
        return "this torch build has no CUDA support (cpu-only wheel)"
    try:
        torch.cuda.init()
    except Exception as problem:  # noqa: BLE001 - the text is the diagnosis
        return f"torch built for CUDA {torch.version.cuda}, but: {problem}"
    return f"torch built for CUDA {torch.version.cuda}, no card visible"


def token_count(tokens: float) -> str:
    """Billions once there are billions, millions before that: a smoke run
    reporting "0.000B tokens" reads as though it trained on nothing.
    """
    if tokens >= 1e9:
        return f"{tokens / 1e9:.3f}B"
    if tokens >= 1e6:
        return f"{tokens / 1e6:.1f}M"
    return f"{tokens / 1e3:.0f}k"


def improvement_per_hour(history: list[tuple[float, float]],
                        window: float) -> float | None:
    """Perplexity points per hour over the last `window` hours. None while
    there is not yet a full window to judge by.

    The anchor is the most recent check that is still at least a window old.
    Judging against the previous check instead would read noise: two checks
    minutes apart differ by less than the run-to-run wobble. Reaching all the
    way back to the first check would be worse still - early training falls
    steeply, so the average since the start stays high long after the run has
    flattened out, and the threshold would never fire.
    """
    if len(history) < 2:
        return None
    latest_at, latest = history[-1]
    for stamp, score in reversed(history[:-1]):
        if latest_at - stamp >= window * 3600:
            elapsed_hours = (latest_at - stamp) / 3600
            return (score - latest) / elapsed_hours
    return None


def slow_streak_after(streak: int, gain: float | None,
                      threshold: float) -> int:
    """How many checks in a row have come in under the threshold.

    A check with no rate yet (too early to have a window) leaves the count
    alone rather than breaking a streak that is already running.
    """
    if gain is None:
        return streak
    return streak + 1 if gain < threshold else 0


def streak_is_enough(streak: int, patience: int) -> bool:
    """Whether a run of slow checks is long enough to stop on.

    `patience` below one is treated as one: zero would make an empty streak
    sufficient and end the run at its first evaluation, while still climbing.
    """
    return streak > 0 and streak >= max(1, patience)


def fit_micro_batch(trial, wanted: int, budget: int, device: str) -> int:
    """Largest micro-batch that survives one real step, at most `budget`.

    A conservative default leaves the card idle; a guess that is too large dies
    hours in. One trial step costs seconds and settles it on the actual machine.
    """
    import torch

    candidates = [n for n in (32, 24, 16, 12, 8, 6, 4, 3, 2, 1)
                  if wanted <= n <= budget] or [wanted]

    for size in candidates:
        try:
            trial(size)
        except torch.OutOfMemoryError:
            if device.startswith("cuda"):
                torch.cuda.empty_cache()
            continue
        except RuntimeError as problem:
            if "out of memory" not in str(problem).lower():
                raise
            if device.startswith("cuda"):
                torch.cuda.empty_cache()
            continue
        return size
    return 1


# ==============================
# ===  Loop                  ===
# ==============================

def run(config, resume: bool = True) -> Path:
    import torch

    rank, local_rank, world, distributed = distributed_setup()
    lead = rank == 0

    asked = resolve_device(str(config.get("run.device", "auto")))
    device = rank_device(asked, local_rank, distributed)
    # Only worth saying when a specific card was named and is being ignored.
    # "cuda" with no index is not a request for card zero, it is no request.
    if lead and device != asked and ":" in asked:
        ui.warn(f"{asked} was asked for, but this run spans {world} processes "
                f"- each takes its own card, starting at cuda:0")
    kind = device_type(device)
    dtype = resolve_dtype(str(config.get("train.dtype", "bfloat16")), device)
    torch.manual_seed(int(config.get("run.seed", 0)) + rank)
    if kind == "cuda" and not distributed:
        # Pin the default device, so scratch allocations land on the card the
        # config asked for rather than on card zero.
        torch.cuda.set_device(device)

    seq = int(config.require("train.seq"))
    raw_micro = config.require("train.micro_batch")
    auto_micro = str(raw_micro).lower() == "auto"
    micro = 1 if auto_micro else int(raw_micro)
    accum = int(config.require("train.accum"))
    chunk = int(config.get("train.loss_chunk", 256))
    alpha = float(config.get("train.alpha", 0.9))
    temperature = float(config.get("train.temperature", 1.0))
    grad_clip = float(config.get("train.grad_clip", 1.0))
    per_step = micro * accum * seq * world
    budget_tokens = int(float(config.require("train.max_tokens")))
    total_steps = max(1, budget_tokens // per_step)

    objective = str(config.get("train.objective", "distill")).lower()
    if objective not in ("distill", "language"):
        raise ValueError(f"train.objective must be distill or language, "
                         f"got '{objective}'")

    # These fallbacks are the values in config.toml: a config file that
    # predates the setting must behave the way the documentation describes.
    min_gain = float(config.get("train.min_improvement_per_hour", 0.05))
    gain_window = float(config.get("train.improvement_window_hours", 3.0))
    gain_patience = max(1, int(config.get("train.stop_patience", 2)))
    if min_gain > 0 and gain_window <= 0:
        raise ValueError("train.improvement_window_hours must be above zero; "
                         f"got {gain_window}")

    run_dir = config.run_dir()
    if lead:
        run_dir.mkdir(parents=True, exist_ok=True)
        ui.head("Training")
        ui.field("preset", config.get("preset", "-"))
        ui.field("device", f"{device} x{world}" if world > 1 else device)
        ui.field("objective", objective)
        ui.field("teacher" if objective == "distill" else "model",
                 config.require("model.teacher"))
        ui.field("token budget", token_count(budget_tokens))
        ui.field("run directory", run_dir)

    if lead and kind == "cpu":
        # A run that quietly lands on the processor looks identical to a slow
        # one for the first hour, and then for the next twenty.
        ui.fail("no accelerator: this run is on the processor and will take "
                "orders of magnitude longer")
        why = cuda_unavailable_reason()
        if why:
            ui.say(f"      {why}", "overlay")
        ui.say("      check `make status`; set run.device to force a card",
               "overlay")

    teacher = models.load_teacher(config, device, dtype) \
        if objective == "distill" else None
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
        stored_step = checkpoint.load_into(student, optimizer, existing)
        # Continue by tokens, not by step number: the micro-batch is probed per
        # machine, so a step here and a step in the previous run need not be the
        # same size, and counting steps would silently skip or repeat work.
        meta = checkpoint.read_meta(existing)
        seen = int(meta.get("tokens") or stored_step * per_step)
        start_step = seen // per_step
        if lead and meta.get("extra", {}).get("stopped_early"):
            # Continuing one of these is not wrong - more tokens still help -
            # but it costs a window plus the patience checks to reach the same
            # verdict, and the user should know that is what they are buying.
            ui.warn("the previous run stopped itself: perplexity was falling "
                    "slower than train.min_improvement_per_hour")
            ui.say("      it will train at least "
                   f"{gain_window * (gain_patience + 1):.0f}h before it can "
                   "decide again; STOP=0 trains the whole budget instead",
                   "overlay")
        if lead:
            ui.good(f"resumed from {existing.name}: {token_count(seen)} tokens "
                    f"already seen, continuing at step {start_step}")
            if seen >= budget_tokens:
                ui.warn(f"train.max_tokens is {token_count(budget_tokens)} and "
                        f"{token_count(seen)} are done — raise it to train further")

    trainable = student
    if distributed:
        from torch.nn.parallel import DistributedDataParallel as DDP

        trainable = DDP(student,
                        device_ids=[local_rank] if kind == "cuda" else None)

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

    history: list[tuple[float, float]] = []
    slow_streak = 0
    stopped_early = False
    last_step = total_steps
    if min_gain > 0 and not eval_every:
        # Asked for on the command line: refuse, rather than quietly spend the
        # whole budget the user wanted cut short. Left at its default: warn and
        # carry on, since a run should not fail over a setting nobody typed.
        if config.was_set("train.min_improvement_per_hour"):
            raise ValueError("train.min_improvement_per_hour needs held-out "
                             "checks to judge by, but train.eval_every is 0")
        if lead:
            ui.warn("train.eval_every is 0, so there is nothing to judge "
                    "progress by - training will run to train.max_tokens")
        min_gain = 0.0

    student.train()

    if auto_micro:
        def trial(size: int) -> None:
            ids, gold = stream.batch(size)
            ids, gold = ids.to(device), gold.to(device)
            reference = None
            if teacher is not None:
                with torch.no_grad():
                    reference = teacher(ids).logits
            amp = torch.autocast(device_type=kind, dtype=dtype) \
                if (kind == "cuda" and dtype is not torch.float32) \
                else _nullcontext()
            with amp:
                logits = student(ids).logits
            distil_backward(logits, reference, gold, alpha, temperature, chunk)
            optimizer.zero_grad(set_to_none=True)
            del logits, reference, ids, gold

        # `accum` stays as configured: it is how many micro-batches make a step.
        # Only their size is decided here, by what the card actually holds.
        micro = fit_micro_batch(trial, 1, 32, kind)
        per_step = micro * accum * seq * world
        total_steps = max(1, budget_tokens // per_step)
        # Both ends of the loop are counted in steps but decided in tokens, so
        # both have to be redone once the step size is known. Leaving the start
        # behind was a budget of ten billion tokens finishing after four
        # hundred million: the resumed position had been worked out when a
        # step was thirty two times smaller.
        start_step = seen // per_step
        if lead:
            ui.good(f"micro-batch fitted to {micro}, resuming at step "
                    f"{start_step} of {total_steps}")

    if lead:
        ui.field("tokens per step", f"{per_step / 1e3:.1f}k")
        ui.field("steps planned", total_steps)

    # Only CUDA gets mixed precision here: the student's master weights are
    # float32, and a float32 matmul on a GPU is both slower and larger than it
    # needs to be. On cpu and mps the cast buys nothing and costs correctness.
    autocast_on = kind == "cuda" and dtype is not torch.float32
    # Monotonic, not wall clock: an NTP step or a suspended machine would
    # otherwise inflate the denominator and read as a stalled run.
    started = time.monotonic()
    marker = (started, seen)
    progress = ui.Progress("training", total_steps) if lead else None

    for step in range(start_step, total_steps):
        factor = learning_rate_factor(step, warmup, total_steps)
        optimizer.param_groups[0]["lr"] = base_lr * factor
        optimizer.param_groups[1]["lr"] = base_scale_lr * factor
        optimizer.zero_grad(set_to_none=True)

        kl_total = ce_total = None
        for micro_step in range(accum):
            ids, gold = stream.batch(micro)
            ids = ids.to(device, non_blocking=True)
            gold = gold.to(device, non_blocking=True)
            reference = None
            if teacher is not None:
                with torch.no_grad():
                    reference = teacher(ids).logits
            # Only the last micro-batch needs the gradient all-reduce.
            last = micro_step == accum - 1
            sync = trainable.no_sync() if (distributed and not last) \
                else _nullcontext()
            amp = torch.autocast(device_type=kind, dtype=dtype) \
                if autocast_on else _nullcontext()
            with sync:
                with amp:
                    logits = trainable(ids).logits
                kl, ce = distil_backward(logits, reference, gold, alpha,
                                         temperature, chunk)
            kl_total = kl / accum if kl_total is None else kl_total + kl / accum
            ce_total = ce / accum if ce_total is None else ce_total + ce / accum
            del logits, reference

        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(student.parameters(), grad_clip)
        optimizer.step()
        seen += per_step

        reporting = lead and (step % log_every == 0 or step == total_steps - 1)
        if reporting:
            kl_total = float(kl_total)
            ce_total = float(ce_total)

        if reporting:
            now = time.monotonic()
            rate = (seen - marker[1]) / max(1e-6, now - marker[0])
            marker = (now, seen)
            record = {"step": step, "tokens": seen, "kl": kl_total,
                      "ce": ce_total, "ppl": math.exp(min(20.0, ce_total)),
                      "lr": base_lr * factor,
                      "tokens_per_second": round(rate, 1),
                      "elapsed": round(now - started, 1)}
            if kind == "cuda":
                record["gpu_gb"] = round(
                    torch.cuda.max_memory_allocated() / 2 ** 30, 2)
            metrics.write(record)
            progress.update(step - start_step)
            if step % (log_every * 10) == 0:
                ui.say()
                extra = f"  {record['gpu_gb']:.1f} GB" if "gpu_gb" in record else ""
                ui.step(f"step {step:>6}  kl {kl_total:7.4f}  "
                        f"ce {ce_total:6.3f}  ppl {record['ppl']:9.2f}  "
                        f"{token_count(seen)} tokens  "
                        f"{record['tokens_per_second'] / 1e3:.1f}k tok/s{extra}")

        if lead and eval_every and step and step % eval_every == 0:
            score = quick_eval(config, student, teacher, device, chunk)
            # `score` carries its own "tokens" — how many were scored — which
            # used to overwrite the training counter and flatten the x axis.
            score["scored_tokens"] = score.pop("tokens", None)
            history.append((time.monotonic(), score["perplexity"]))
            gain = improvement_per_hour(history, gain_window)

            record = {"step": step, "tokens": seen, "eval": True,
                      "elapsed": round(time.monotonic() - started, 1),
                      **score}
            if gain is not None:
                record["perplexity_gain_per_hour"] = round(gain, 4)
            metrics.write(record)

            ui.say()
            ui.field("held-out perplexity", f"{score['perplexity']:.2f}", "peach")
            if "agreement" in score:
                ui.field("agreement with teacher",
                         f"{score['agreement']:.3f}", "peach")
            if gain is not None:
                ui.field("gain per hour", f"{gain:+.3f} perplexity", "peach")

            if min_gain > 0:
                slow_streak = slow_streak_after(slow_streak, gain, min_gain)
                if 0 < slow_streak < gain_patience:
                    ui.field("below the threshold",
                             f"{slow_streak} of {gain_patience} checks",
                             "yellow")
                if streak_is_enough(slow_streak, gain_patience):
                    ui.say()
                    ui.warn(f"perplexity is improving by {gain:.3f} per hour, "
                            f"less than the {min_gain:.3f} asked for, "
                            f"{gain_patience} checks running - stopping here")
                    stopped_early = True

        if lead and save_every and step and step % save_every == 0:
            checkpoint.save(run_dir, student, optimizer, step, seen, keep,
                            extra={"preset": config.get("preset"),
                                   "stream": stream.state()})

        # Only the lead evaluates, so only the lead knows. Leaving the loop
        # without telling the others would hang them on the next all-reduce.
        if min_gain > 0 and eval_every and step and step % eval_every == 0:
            if distributed:
                import torch.distributed as dist

                flag = torch.tensor([float(stopped_early)], device=device)
                dist.broadcast(flag, src=0)
                stopped_early = bool(flag.item())
            if stopped_early:
                last_step = step
                break

    final = None
    if lead:
        if stopped_early:
            ui.field("stopped", "improvement fell below the threshold", "yellow")
        if progress is not None:
            progress.done(f"{token_count(seen)} tokens")
        final = checkpoint.save(run_dir, student, optimizer, last_step, seen,
                                keep, extra={"preset": config.get("preset"),
                                             "stream": stream.state(),
                                             "stopped_early": stopped_early,
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

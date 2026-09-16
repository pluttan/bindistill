"""Task benchmarks: every model answering the same questions.

Perplexity says how a distribution sits against a text. It does not say whether
the model is still good for anything, and it cannot be compared across models
with different tokenisers at all - the measure is per token, and a vocabulary of
32 thousand and one of 152 thousand do not produce comparable tokens. Multiple
choice accuracy has neither problem: the answer is picked from the same options
whatever the tokeniser, so a one-bit model here can be read against a published
number from somebody else's model.

Five rows are measured, in the order they earn their machine time:

    teacher     the full-precision model the student was distilled from
    student     the one-bit model, loaded from a checkpoint
    int4        the same teacher quantised to 4 bits, the method that works
    published   an open low-bit model, for the numbers its authors report
    naive       the sign rule with no training, the floor

Each row is written to its own file as soon as it finishes, so a run killed
half way through keeps what it measured and the next one continues rather than
starts over.

Signed: pluttan
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from . import ui

# The set published alongside small open models, so the rows can be read
# against numbers their authors already report. MMLU is kept apart: it is
# fourteen thousand questions and costs more than the other six together.
CORE_TASKS = ("arc_challenge", "arc_easy", "hellaswag", "openbookqa", "piqa",
              "winogrande")
WIDE_TASKS = ("mmlu",)

# Which of the harness's metrics to believe for each task. Normalised accuracy
# is the one reported for the multiple-choice sets whose options differ in
# length; where it is absent plain accuracy is the only thing offered.
PREFERRED = ("acc_norm,none", "acc,none", "acc_norm", "acc")


def task_list(config, wide: bool) -> list[str]:
    """Which tasks to run. `wide` adds the expensive one."""
    chosen = list(config.get("bench.tasks", CORE_TASKS))
    if wide:
        chosen += list(config.get("bench.wide_tasks", WIDE_TASKS))
    return chosen


def pick_score(entry: dict) -> float | None:
    """The one number out of a task's result block.

    The harness returns several metrics per task and names them with the filter
    that produced them, so the keys are `acc,none` rather than `acc`. Different
    versions have moved this around, which is why the lookup is by preference
    rather than by a fixed key.
    """
    for key in PREFERRED:
        if key in entry and isinstance(entry[key], (int, float)):
            return float(entry[key])
    for key, value in entry.items():
        # `acc_stderr` also starts with `acc` and is not a score: taking it
        # would report a confidence interval as the model's accuracy.
        if "stderr" in key or not isinstance(value, (int, float)):
            continue
        if key.startswith(("acc_norm", "acc")):
            return float(value)
    return None


def collect(results: dict) -> dict:
    """Task name to score, out of whatever shape the harness handed back."""
    table = results.get("results", results) or {}
    scores = {}
    for task, entry in table.items():
        if not isinstance(entry, dict):
            continue
        score = pick_score(entry)
        if score is not None:
            scores[task] = score
    return scores


def average(scores: dict) -> float:
    """Mean over the tasks that produced a number, or zero if none did.

    Subtasks are dropped first: asking for `mmlu` returns the aggregate and all
    fifty seven of its parts, and counting both would weigh it by fifty eight.
    """
    top = {k: v for k, v in scores.items() if "_" not in k or k in TOP_LEVEL}
    use = top or scores
    return sum(use.values()) / max(1, len(use))


TOP_LEVEL = set(CORE_TASKS) | set(WIDE_TASKS)


# ==============================
# ===  The models            ===
# ==============================

def build_teacher(config, device: str):
    """The full-precision model, as it came."""
    import torch

    from . import models

    model = models.load_teacher(config, device, torch.bfloat16)
    return model, models.load_tokenizer(config)


def build_student(config, device: str, checkpoint: Path):
    """The trained one-bit model, restored into a binarised skeleton."""
    from . import models
    from .checkpoint import load_into

    model, _ = models.load_student(config, device)
    step = load_into(model, None, checkpoint)
    ui.detail(f"student restored from step {step}")
    model.to(device)
    return model, models.load_tokenizer(config)


def build_naive(config, device: str):
    """The sign rule with no training: the floor every row has to clear."""
    import torch

    from . import models

    model, _ = models.load_naive_student(config, device, torch.bfloat16)
    return model, models.load_tokenizer(config)


def build_int4(config, device: str):
    """The same teacher at 4 bits - the discipline that is already solved.

    This is the row that makes the comparison mean something: 4-bit quantisation
    is what a practitioner would reach for today, and a one-bit model has to be
    read against it rather than against an untrained binarisation.
    """
    import torch
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    from . import models

    quantisation = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(
        models.model_source(config),
        quantization_config=quantisation,
        device_map={"": device},
        low_cpu_mem_usage=True)
    return model, models.load_tokenizer(config)


def restore_removed_names() -> list[str]:
    """Put back helper types that transformers has since dropped.

    A published model carries its own modelling code, written against the
    library as it was at the time, and that code is imported as-is. Names used
    only for type annotations get renamed or removed between versions, and the
    import then fails on a model that would otherwise run unchanged.

    Downgrading the library to suit one comparison row is the wrong trade, so
    the missing names are supplied instead. Each is an annotation helper with
    no behaviour, which is why a stand-in works at all; anything with logic in
    it would not be safe to fake, and is not faked here.
    """
    from typing import Optional, TypedDict

    import transformers.utils as utils

    restored = []
    if not hasattr(utils, "LossKwargs"):
        class LossKwargs(TypedDict, total=False):
            """Annotation-only, as it was when the model was published."""

            num_items_in_batch: Optional[int]

        utils.LossKwargs = LossKwargs
        restored.append("LossKwargs")
    return restored


def build_published(config, device: str):
    """An open low-bit model, measured here only to read its own numbers.

    Its weights are not inspected and nothing is inferred about how it was
    made: the point is one more row of accuracies on the same questions.
    """
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    name = str(config.get("bench.published", "deepgrove/Bonsai"))
    restored = restore_removed_names()
    if restored:
        ui.detail(f"transformers {transformers.__version__}: supplied "
                  f"{', '.join(restored)} for the published model's own code")
    model = AutoModelForCausalLM.from_pretrained(
        name, trust_remote_code=True, dtype=torch.bfloat16)
    model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    return model, tokenizer


BUILDERS = {
    "teacher": build_teacher,
    "student": build_student,
    "int4": build_int4,
    "published": build_published,
    "naive": build_naive,
}

# The order is by what the paper loses most if the night runs short.
ORDER = ("teacher", "student", "int4", "published", "naive")


# ==============================
# ===  Running one row       ===
# ==============================

def score_model(model, tokenizer, tasks: list[str], batch_size,
                limit: float | None) -> dict:
    """Hand an already-built model to the harness and take the accuracies."""
    import lm_eval
    from lm_eval.models.huggingface import HFLM

    wrapped = HFLM(pretrained=model, tokenizer=tokenizer,
                   batch_size=batch_size)
    results = lm_eval.simple_evaluate(model=wrapped, tasks=tasks,
                                      num_fewshot=0, limit=limit,
                                      bootstrap_iters=0)
    if results is None:
        return {}
    if not isinstance(results, dict):
        # Newer versions hand back an object holding the same table.
        results = getattr(results, "results", None) or {}
        return collect({"results": results})
    return collect(results)


def free(model) -> None:
    """Give the card back before the next model is built."""
    import torch

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run(config, checkpoint: Path | None = None, hours: float = 18.0,
        limit: float | None = None, wide: bool = True,
        only: list[str] | None = None) -> dict:
    """Every model over every task, within the time allowed.

    Rows already on disk are kept: a night that ran out of time is continued by
    running the same command again.
    """
    from .config import resolve_device

    device = resolve_device(str(config.get("run.device", "auto")))
    tasks = task_list(config, wide)
    batch_size = config.get("bench.batch_size", "auto")
    room = config.run_dir() / "bench"
    room.mkdir(parents=True, exist_ok=True)

    wanted = [name for name in ORDER if not only or name in only]
    if checkpoint is None and "student" in wanted:
        ui.warn("no checkpoint given, the student row is skipped")
        wanted.remove("student")

    ui.head("Benchmarks")
    ui.field("device", device)
    ui.field("tasks", ", ".join(tasks))
    ui.field("models", ", ".join(wanted))
    ui.field("time allowed", f"{hours:.1f} h")
    if limit:
        ui.field("examples per task", limit)

    deadline = time.time() + hours * 3600
    collected = {}

    for name in wanted:
        written = room / f"{name}.json"
        if written.exists() and limit is None:
            stored = json.loads(written.read_text())
            if stored.get("scores"):
                collected[name] = stored
                ui.detail(f"{name}: already measured, kept")
                continue

        left = deadline - time.time()
        if left <= 0:
            ui.warn(f"{name}: out of time, not started")
            continue

        ui.say()
        ui.field("measuring", f"{name} ({left / 3600:.1f} h left)")
        started = time.time()
        model = None
        try:
            builder = BUILDERS[name]
            if name == "student":
                model, tokenizer = builder(config, device, checkpoint)
            else:
                model, tokenizer = builder(config, device)
            scores = score_model(model, tokenizer, tasks, batch_size, limit)
            row = {"model": name, "scores": scores,
                   "average": average(scores),
                   "minutes": (time.time() - started) / 60}
        except Exception as problem:  # noqa: BLE001 - a row, not the night
            # One model failing to build is not a reason to lose the rest: the
            # 4-bit row needs a package that may not be installed, and the
            # published one needs the network.
            ui.warn(f"{name}: {problem}")
            row = {"model": name, "scores": {}, "error": str(problem),
                   "minutes": (time.time() - started) / 60}
        finally:
            if model is not None:
                free(model)

        if limit is None:
            written.write_text(json.dumps(row, indent=2))
        collected[name] = row
        if row["scores"]:
            ui.good(f"{name}: average {row['average']:.3f} "
                    f"in {row['minutes']:.0f} min")

    report(collected, tasks)
    return collected


def report(collected: dict, tasks: list[str]) -> None:
    """One table, models down the side and tasks across."""
    shown = [t for t in tasks if any(t in r.get("scores", {})
                                     for r in collected.values())]
    if not shown:
        ui.say()
        ui.warn("nothing was measured")
        return

    rows = []
    for name in ORDER:
        row = collected.get(name)
        if not row or not row.get("scores"):
            continue
        cells = [f"{row['scores'].get(t, float('nan')) * 100:.1f}"
                 if t in row["scores"] else "-" for t in shown]
        rows.append((name, *cells, f"{row['average'] * 100:.1f}"))

    ui.say()
    ui.table(rows, ("model", *shown, "avg"))
    ui.say()
    ui.detail("accuracy in per cent, zero-shot, same harness for every row")

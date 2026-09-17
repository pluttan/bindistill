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

# The set reported in the work this field is measured against: BitNet b1.58
# lists ARC-Easy, ARC-Challenge, HellaSwag, WinoGrande, PIQA, OpenbookQA and
# BoolQ, and no MMLU. Matching it is what makes a row comparable to a published
# one; a different set would have to be argued for.
CORE_TASKS = ("arc_easy", "arc_challenge", "hellaswag", "winogrande", "piqa",
              "openbookqa", "boolq")

# Knowledge and instruction following, reported by the later and much larger
# models of that line. Kept apart because MMLU alone is fourteen thousand
# questions and costs more than the seven above together.
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

    restored += restore_default_rope()
    return restored


def restore_default_rope() -> list[str]:
    """Put back the unscaled entry in the table of position encodings.

    A model without rope scaling asks this table for `default`, and the
    library no longer has that key: what is left is the scaled variants. The
    entry is restored rather than the model patched, because the model's file
    holds the table by reference and every layer reads it at construction.

    The library's own unscaled routine is reused if it is still there under its
    private name. Only if it is gone is one written out, and it is written out
    plainly: the inverse frequencies of ordinary rotary encoding, the formula
    that has not changed since it was introduced.
    """
    from transformers import modeling_rope_utils as rope

    if "default" in rope.ROPE_INIT_FUNCTIONS:
        return []

    routine = getattr(rope, "_compute_default_rope_parameters", None)
    if routine is None:
        def routine(config, device=None, seq_len=None, **kwargs):
            import torch

            base, turn = rope_settings(config)
            width = getattr(config, "head_dim", None) or (
                config.hidden_size // config.num_attention_heads)
            turning = int(width * turn)
            steps = torch.arange(0, turning, 2, dtype=torch.int64).to(
                device=device, dtype=torch.float)
            return 1.0 / (base ** (steps / turning)), 1.0

    rope.ROPE_INIT_FUNCTIONS["default"] = routine
    return ["ROPE_INIT_FUNCTIONS['default']"]


def report_rope(config) -> None:
    """Say out loud what the restored entry will use.

    A wrong base here does not raise: the model loads, generates, and answers
    at chance, which looks like a weak model rather than a broken one. The
    value published for the model is the only way to tell, so it gets printed
    next to the measurement.
    """
    base, turn = rope_settings(config)
    ui.field("rope base in use", f"{base:g}")
    if turn != 1.0:
        ui.field("partial rotary factor", turn)
    if base <= 10000.0:
        ui.warn("this is the library default, not a value read from the "
                "model: if the model was published with another base, its "
                "row is measuring a broken model and must not be reported")


def rope_settings(config) -> tuple[float, float]:
    """The base and the rotary factor, wherever this version keeps them.

    They used to be plain attributes of the config. Now they live in a
    dictionary, which may itself be keyed by layer type, and reading the old
    attribute raises rather than returning nothing. All three places are tried
    before falling back to the value that was the default when the model was
    written.
    """
    holder = getattr(config, "rope_parameters", None)
    if isinstance(holder, dict):
        if "rope_theta" in holder:
            return float(holder["rope_theta"]), float(
                holder.get("partial_rotary_factor", 1.0))
        for value in holder.values():
            if isinstance(value, dict) and "rope_theta" in value:
                return float(value["rope_theta"]), float(
                    value.get("partial_rotary_factor", 1.0))

    try:
        written = config.to_dict()
    except Exception:  # noqa: BLE001 - fall back to the raw namespace
        written = dict(getattr(config, "__dict__", {}))
    base = written.get("rope_theta") or written.get("theta") or 10000.0
    return float(base), float(written.get("partial_rotary_factor", 1.0))


def build_published(config, device: str):
    """An open low-bit model, measured here only to read its own numbers.

    Its weights are not inspected and nothing is inferred about how it was
    made: the point is one more row of accuracies on the same questions.
    """
    import torch
    import transformers
    from transformers import (AutoConfig, AutoModelForCausalLM,
                              AutoTokenizer)

    name = str(config.get("bench.published", "deepgrove/Bonsai"))
    restored = restore_removed_names()
    if restored:
        ui.detail(f"transformers {transformers.__version__}: supplied "
                  f"{', '.join(restored)} for the published model's own code")

    # The attention implementation has to be named explicitly. The model's own
    # code looks it up in a table of the library's functions, and the value the
    # library now defaults to is not a key in that table. Asking through the
    # loader is tried first, then through the config, because which of the two
    # wins has changed between versions.
    model = None
    problems = []
    for how, through_config in (("eager", False), ("eager", True),
                                ("sdpa", True)):
        try:
            if through_config:
                settings = AutoConfig.from_pretrained(name,
                                                      trust_remote_code=True)
                settings._attn_implementation = how
                model = AutoModelForCausalLM.from_pretrained(
                    name, config=settings, trust_remote_code=True,
                    dtype=torch.bfloat16)
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    name, trust_remote_code=True, dtype=torch.bfloat16,
                    attn_implementation=how)
            break
        except Exception as problem:  # noqa: BLE001 - try the next way in
            import traceback

            frames = traceback.extract_tb(problem.__traceback__)
            spot = ""
            if frames:
                spot = (f" raised at {Path(frames[-1].filename).name}:"
                        f"{frames[-1].lineno} in {frames[-1].name}")
            problems.append(f"[{how}"
                            f"{', via config' if through_config else ''}] "
                            f"{type(problem).__name__}: {problem}{spot}")
    if model is None:
        raise RuntimeError(" | ".join(problems))

    # Its layers read the setting during the forward pass, not at load time,
    # and each one keeps its own reference to a config object. Setting it in
    # one place is not enough if another holds a copy.
    settled = 0
    for module in model.modules():
        holder = getattr(module, "config", None)
        if holder is not None and getattr(
                holder, "_attn_implementation", None) == "default":
            holder._attn_implementation = "eager"
            settled += 1
    if settled:
        ui.detail(f"attention set to eager on {settled} modules that still "
                  f"held the library's placeholder")

    report_rope(model.config)
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
ORDER = ("teacher", "student", "int4", "naive")

# Measured only when asked for by name. A published model carries its own
# modelling code, written against the library as it stood at the time, and
# running it here means patching around every name the library has since moved.
# Three such patches went in before the model loaded, and it then answered at
# chance - far below what its authors report - so something else is broken too.
# A row measured that way says nothing true about somebody else's work, and the
# figures they published are the honest thing to cite instead.
ON_REQUEST = ("published",)


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

    # Named explicitly: measure exactly those. Named nothing: the default set,
    # which leaves out the rows that only make sense when asked for.
    known = ORDER + ON_REQUEST
    wanted = [name for name in known if name in only] if only else list(ORDER)
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
            #
            # The type and the frame are printed with the message because some
            # of these say very little on their own: a KeyError prints as the
            # missing key and nothing else, which names neither the dictionary
            # nor the file it was read in.
            import traceback

            trace = traceback.format_exc()
            frames = traceback.extract_tb(problem.__traceback__)
            where = ""
            if frames:
                last = frames[-1]
                where = f" at {Path(last.filename).name}:{last.lineno}"
            ui.warn(f"{name}: {type(problem).__name__}: {problem}{where}")
            if frames:
                ui.detail(f"    {frames[-1].line}")
            row = {"model": name, "scores": {},
                   "error": f"{type(problem).__name__}: {problem}",
                   "traceback": trace,
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

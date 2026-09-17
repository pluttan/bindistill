"""Perplexity on the corpora everyone else reports it on.

The perplexity this package measures during training is taken on a held-out
slice of its own corpus. That is the right number for watching a run, and it
is worthless for comparison: it depends on which corpus was collected, on the
tokeniser, and on where the slice fell. Published work on low-bit models
reports validation perplexity on WikiText-2 and C4, and a paper without those
two numbers cannot be placed next to any of it.

The procedure is the one those papers use: concatenate the split, tokenise it
whole, cut it into windows of a fixed length, and average the negative log
likelihood over every token. Window length matters - a longer window gives a
lower number - so it is reported alongside.

Signed: pluttan
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from . import ui

# What the field reports on. The C4 shard is the one named in the quantisation
# papers; the whole validation split is far larger than anyone measures on.
CORPORA = {
    "wikitext2": {
        "path": "wikitext",
        "name": "wikitext-2-raw-v1",
        "split": "test",
        "field": "text",
        "joiner": "\n\n",
    },
    "c4": {
        "path": "allenai/c4",
        "files": {"validation": "en/c4-validation.00000-of-00008.json.gz"},
        "split": "validation",
        "field": "text",
        "joiner": " ",
        "documents": 1100,
    },
}


def load_text(which: str) -> str:
    """One corpus, joined into a single string, as the papers prepare it."""
    from datasets import load_dataset

    spec = CORPORA[which]
    if "files" in spec:
        data = load_dataset(spec["path"], data_files=spec["files"],
                            split=spec["split"])
        # The validation shard holds far more than is ever measured on; the
        # papers take a few hundred documents and stop.
        data = data.select(range(min(len(data), spec["documents"])))
    else:
        data = load_dataset(spec["path"], spec["name"], split=spec["split"])
    return spec["joiner"].join(data[spec["field"]])


def perplexity(model, tokenizer, text: str, device: str, window: int = 2048,
               limit: int | None = None) -> dict:
    """Average negative log likelihood over fixed windows, exponentiated.

    Windows do not overlap and the text is not re-tokenised per window: the
    whole split is encoded once and cut, which is what makes the number
    comparable to published ones.
    """
    import torch

    encoded = tokenizer(text, return_tensors="pt").input_ids
    windows = encoded.numel() // window
    if limit:
        windows = min(windows, limit)
    if windows < 1:
        raise ValueError(f"text is shorter than one window of {window}")

    total = 0.0
    counted = 0
    model.eval()
    with torch.no_grad():
        for index in range(windows):
            piece = encoded[:, index * window:(index + 1) * window].to(device)
            out = model(piece, labels=piece)
            # `labels` shifts internally, so a window of n tokens scores n-1.
            total += float(out.loss) * (window - 1)
            counted += window - 1

    return {"perplexity": math.exp(min(30.0, total / max(1, counted))),
            "windows": windows, "window": window, "tokens": counted}


def run(config, checkpoint: Path | None = None, window: int = 2048,
        limit: int | None = None, only: list[str] | None = None) -> dict:
    """Every model on every reference corpus, in one table."""
    import torch

    from . import bench
    from .config import resolve_device

    device = resolve_device(str(config.get("run.device", "auto")))
    wanted = [n for n in bench.ORDER if not only or n in only]
    corpora = list(config.get("reference.corpora", tuple(CORPORA)))

    ui.head("Reference perplexity")
    ui.field("device", device)
    ui.field("corpora", ", ".join(corpora))
    ui.field("window", window)
    if checkpoint is None and "student" in wanted:
        ui.warn("no checkpoint given, the student row is skipped")
        wanted.remove("student")

    texts = {}
    for which in corpora:
        try:
            texts[which] = load_text(which)
            ui.detail(f"{which}: {len(texts[which]) / 1e6:.1f}M characters")
        except Exception as problem:  # noqa: BLE001 - one corpus, not the run
            ui.warn(f"{which}: {problem}")

    collected = {}
    for name in wanted:
        model = None
        try:
            builder = bench.BUILDERS[name]
            if name == "student":
                model, tokenizer = builder(config, device, checkpoint)
            else:
                model, tokenizer = builder(config, device)
            scores = {}
            for which, text in texts.items():
                scores[which] = perplexity(model, tokenizer, text, device,
                                           window, limit)
                ui.detail(f"{name} on {which}: "
                          f"{scores[which]['perplexity']:.2f}")
            collected[name] = scores
        except Exception as problem:  # noqa: BLE001 - a row, not the run
            ui.warn(f"{name}: {type(problem).__name__}: {problem}")
        finally:
            if model is not None:
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    rows = []
    for name in bench.ORDER:
        if name not in collected:
            continue
        cells = [f"{collected[name][w]['perplexity']:.2f}"
                 if w in collected[name] else "-" for w in corpora]
        rows.append((name, *cells))
    if rows:
        ui.say()
        ui.table(rows, ("model", *corpora))
        ui.say()
        ui.detail(f"windows of {window} tokens, no overlap, whole split "
                  f"encoded once - the procedure published work uses")

    room = config.run_dir()
    room.mkdir(parents=True, exist_ok=True)
    (room / "reference.json").write_text(json.dumps(collected, indent=2))
    ui.good(f"written to {room / 'reference.json'}")
    return collected

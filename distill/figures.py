"""Figures for the paper, drawn from what was measured.

Every number here either comes out of a file this package wrote or is quoted
from a published table with the source named next to it. Nothing is redrawn
from the paper's prose: the prose and the figures read the same measurements,
so when a number changes there is one place to change it.

Figures are sized for a column of the IEEE template - 3.4 inches wide - and
saved at 300 dpi, because a plot that has to be shrunk to fit loses its labels
first and its credibility second.

Signed: pluttan
"""

from __future__ import annotations

import json
from pathlib import Path

from . import ui

# ==============================
# ===  Measurements          ===
# ==============================

# Run 1, FineWeb-Edu, 300M tokens: perplexity on the held-out text at each
# evaluation point. The teacher is flat at 25.02 on the same text.
# Run 1, FineWeb-Edu, 300M tokens. These are five evaluations spaced along the
# run; the budget each one fell at was not recorded, so the axis counts
# evaluations rather than inventing token counts for them. Only the last point
# is pinned: it is the end of the 300M-token budget.
CURVE = {
    "student": (81.9, 42.7, 34.6, 30.2, 27.85),
    "teacher": 25.02,
    "budget": 300,          # million tokens at the last point
}

# Scale sweep: every group scale of the finished checkpoint multiplied by a
# common factor, perplexity measured at each one (12 windows of 1024 tokens of
# literary text, outside the training corpus).
SWEEP = {
    "ratio": (0.76, 0.92, 1.08, 1.25, 1.46, 1.73, 2.00, 2.38),
    "perplexity": (49.93, 39.89, 36.86, 36.94, 40.03, 48.49, 62.90, 99.24),
    "trained": 1.08,        # where training put it
}

# Closeness of the student's choice to the teacher's, run 2, held-out olmo-mix.
AGREEMENT = {
    "student": (0.722, 0.935, 0.964),
    "naive": (0.000, 0.001, 0.000),
    "labels": ("совпадение\nвыбора", "удержание\nв пятёрке", "на уверенных\nпозициях"),
}

# Reference perplexity, 2048-token windows, whole split encoded once.
REFERENCE = {
    "teacher": {"wikitext2": 20.97, "c4": 30.38, "bits": 16},
    "int4":    {"wikitext2": 23.80, "c4": 34.62, "bits": 4},
    "student": {"wikitext2": 30.20, "c4": 39.19, "bits": 1.125},
    "naive":   {"wikitext2": 4720537.90, "c4": 3684890.16, "bits": 1.125},
}

# Published low-bit results on WikiText-2, as ratios to the full-precision
# model each one starts from. Absolute perplexity is not comparable across
# tokenisers; the ratio is. Source: table 3 of arXiv:2402.04291 (BiLLM).
PUBLISHED = (
    # label,                       bits,  ratio,      family
    ("округление, 1 бит",          1.00,  168388.00 / 5.68, "LLaMA-7B"),
    ("GPTQ, 1 бит",                1.00,  267001.72 / 5.68, "LLaMA-7B"),
    ("GPTQ, 2 бита",               2.00,     152.31 / 5.68, "LLaMA-7B"),
    ("PB-LLM, 1,70 бита",          1.70,     102.36 / 5.68, "LLaMA-7B"),
    ("BiLLM, 1,09 бита",           1.09,      35.04 / 5.68, "LLaMA-7B"),
    ("BiLLM, 1,08 бита",           1.08,      32.48 / 5.47, "LLaMA2-7B"),
)

# ==============================
# ===  Look                  ===
# ==============================

# Catppuccin Mocha. The accents carry the data; the dark neutrals carry the
# axes, so the figure stays legible printed in grey.
MOCHA = {
    "blue": "#89b4fa", "red": "#f38ba8", "green": "#a6e3a1",
    "peach": "#fab387", "mauve": "#cba6f7", "teal": "#94e2d5",
    "yellow": "#f9e2af", "overlay": "#6c7086", "ink": "#11111b",
    "surface": "#585b70",
}

WIDTH = 3.4          # one column of the IEEE template, inches
DPI = 300


def style():
    """Matplotlib defaults matching the paper: serif, 8 pt, no chrome."""
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import font_manager, rcParams

    have = {f.name for f in font_manager.fontManager.ttflist}
    serif = next((n for n in ("Times New Roman", "Liberation Serif",
                              "Nimbus Roman", "DejaVu Serif") if n in have),
                 "serif")
    rcParams.update({
        "font.family": "serif", "font.serif": [serif], "font.size": 8,
        "axes.labelsize": 8, "axes.titlesize": 8, "legend.fontsize": 7,
        "xtick.labelsize": 7, "ytick.labelsize": 7,
        "axes.edgecolor": MOCHA["ink"], "axes.labelcolor": MOCHA["ink"],
        "text.color": MOCHA["ink"], "xtick.color": MOCHA["ink"],
        "ytick.color": MOCHA["ink"], "axes.linewidth": 0.6,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "figure.dpi": DPI, "savefig.dpi": DPI, "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02, "legend.frameon": False,
        "axes.spines.top": False, "axes.spines.right": False,
    })
    return serif


def ru(value: float, digits: int = 2) -> str:
    """A number the way the paper writes it: comma decimal, spaced thousands."""
    text = f"{value:,.{digits}f}".replace(",", "\u2009").replace(".", ",")
    return text.rstrip("0").rstrip(",") if "," in text and digits > 2 else text


def comma_axis(ax, which: str = "both"):
    """Replace the decimal points matplotlib puts on the ticks."""
    from matplotlib.ticker import FuncFormatter

    def tick(v, _):
        # matplotlib switches to 1e+06 past five digits, which is not how the
        # paper writes numbers.
        if abs(v) >= 10000:
            return f"{v:,.0f}".replace(",", "\u2009")
        return f"{v:g}".replace(".", ",")

    fmt = FuncFormatter(tick)
    if which in ("x", "both"):
        ax.xaxis.set_major_formatter(fmt)
    if which in ("y", "both"):
        ax.yaxis.set_major_formatter(fmt)


def finish(fig, path: Path):
    fig.savefig(path)
    import matplotlib.pyplot as plt
    plt.close(fig)
    ui.detail(f"{path.name}")


# ==============================
# ===  The figures           ===
# ==============================

def fig_deciles(signature: dict, out: Path):
    """Sign inversions by decile of the original weight - the training signature.

    Log scale: the point is that the two ends differ by three orders of
    magnitude, and on a linear axis the last four deciles are invisible.
    """
    import matplotlib.pyplot as plt

    share = [v * 100 for v in signature["deciles"]]
    fig, ax = plt.subplots(figsize=(WIDTH, 2.1))
    ax.bar(range(1, 11), share, color=MOCHA["mauve"],
           edgecolor=MOCHA["surface"], linewidth=0.5, width=0.72)
    ax.set_yscale("log")
    ax.set_xticks(range(1, 11))
    ax.set_xlabel("дециль модуля исходного веса")
    ax.set_ylabel("знаков перевёрнуто, %")
    ax.axhline(signature["flipped_share"] * 100, color=MOCHA["red"],
               linewidth=0.8, linestyle=(0, (4, 2)))
    ax.set_ylim(top=max(share) * 3.2)
    ax.annotate(f"в среднем {ru(signature['flipped_share'] * 100, 1)} %",
                xy=(10.4, signature["flipped_share"] * 100), ha="right",
                va="bottom", color=MOCHA["red"], fontsize=7)
    ax.grid(axis="y", color=MOCHA["overlay"], alpha=0.25, linewidth=0.4)
    ax.set_axisbelow(True)
    finish(fig, out / "deciles.png")


def fig_sweep(out: Path):
    """Perplexity against the group scale: the trained value sits in the minimum."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(WIDTH, 2.1))
    ax.plot(SWEEP["ratio"], SWEEP["perplexity"], marker="o", markersize=3,
            color=MOCHA["blue"], linewidth=1.1, markerfacecolor="white",
            markeredgewidth=0.8)
    best = SWEEP["perplexity"][SWEEP["ratio"].index(SWEEP["trained"])]
    ax.scatter([SWEEP["trained"]], [best], s=34, color=MOCHA["red"], zorder=5)
    ax.annotate("обученное\nзначение", xy=(SWEEP["trained"], best),
                xytext=(SWEEP["trained"] + 0.06, best + 17),
                color=MOCHA["red"], fontsize=7,
                arrowprops=dict(arrowstyle="-", color=MOCHA["red"], lw=0.6))
    ax.set_xlabel("масштаб группы к среднему модулю весов")
    ax.set_ylabel("перплексия")
    comma_axis(ax)
    ax.grid(color=MOCHA["overlay"], alpha=0.25, linewidth=0.4)
    ax.set_axisbelow(True)
    finish(fig, out / "scale-sweep.png")


def fig_curve(out: Path):
    """Training curve against the teacher it is aiming at."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(WIDTH, 2.1))
    spot = range(1, len(CURVE["student"]) + 1)
    ax.plot(list(spot), CURVE["student"], marker="o", markersize=3,
            color=MOCHA["teal"], linewidth=1.1, markerfacecolor="white",
            markeredgewidth=0.8, label="однобитный студент")
    ax.axhline(CURVE["teacher"], color=MOCHA["overlay"], linewidth=0.8,
               linestyle=(0, (4, 2)), label=f"учитель, {ru(CURVE['teacher'])}")
    ax.set_xticks(list(spot))
    ax.set_xlabel("замер на отложенном тексте по ходу прогона")
    ax.set_ylabel("перплексия")
    ax.annotate(f"{ru(CURVE['student'][-1])}\nконец бюджета\n{CURVE['budget']} млн токенов",
                xy=(len(CURVE["student"]), CURVE["student"][-1]),
                xytext=(-4, 12), textcoords="offset points",
                ha="right", fontsize=6.5, color=MOCHA["ink"])
    comma_axis(ax, "y")
    ax.legend(loc="upper right")
    ax.grid(color=MOCHA["overlay"], alpha=0.25, linewidth=0.4)
    ax.set_axisbelow(True)
    finish(fig, out / "training-curve.png")


def fig_agreement(out: Path):
    """How much of the teacher's behaviour survived, against a no-training control."""
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(WIDTH, 2.1))
    spot = np.arange(len(AGREEMENT["labels"]))
    ax.bar(spot - 0.19, [v * 100 for v in AGREEMENT["student"]], width=0.36,
           color=MOCHA["green"], edgecolor=MOCHA["surface"], linewidth=0.5,
           label="однобитный студент")
    ax.bar(spot + 0.19, [v * 100 for v in AGREEMENT["naive"]], width=0.36,
           color=MOCHA["red"], edgecolor=MOCHA["surface"], linewidth=0.5,
           label="наивное округление")
    for x, v in zip(spot, AGREEMENT["student"]):
        ax.annotate(ru(v * 100, 1), xy=(x - 0.19, v * 100), xytext=(0, 2),
                    textcoords="offset points", ha="center", fontsize=7)
    # The control is flat zero; without a label it reads as missing data.
    for x, v in zip(spot, AGREEMENT["naive"]):
        ax.annotate(ru(v * 100, 1), xy=(x + 0.19, v * 100), xytext=(0, 2),
                    textcoords="offset points", ha="center", fontsize=7,
                    color=MOCHA["surface"])
    ax.set_xticks(spot)
    ax.set_xticklabels(AGREEMENT["labels"])
    ax.set_ylabel("доля токенов, %")
    ax.set_ylim(0, 118)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.22), ncol=2)
    ax.grid(axis="y", color=MOCHA["overlay"], alpha=0.25, linewidth=0.4)
    ax.set_axisbelow(True)
    finish(fig, out / "agreement.png")


def fig_ratios(reference: dict, out: Path):
    """This work against published low-bit results, as ratio to full precision.

    Two columns wide: the labels are long and the axis spans six decades, and
    squeezing that into one column makes it unreadable.
    """
    import matplotlib.pyplot as plt

    teacher = reference["teacher"]["wikitext2"]
    rows = [(label, bits, ratio, MOCHA["overlay"])
            for label, bits, ratio, _ in PUBLISHED]
    rows.append(("наивное округление, 1,125 бита", 1.125,
                 reference["naive"]["wikitext2"] / teacher, MOCHA["red"]))
    rows.append(("настоящая работа, 1,125 бита", 1.125,
                 reference["student"]["wikitext2"] / teacher, MOCHA["green"]))
    rows.append(("квантование в 4 бита, NF4", 4.0,
                 reference["int4"]["wikitext2"] / teacher, MOCHA["blue"]))
    rows.sort(key=lambda r: r[2], reverse=True)

    fig, ax = plt.subplots(figsize=(WIDTH * 2.1, 2.7))
    spot = range(len(rows))
    ax.barh(list(spot), [r[2] for r in rows], color=[r[3] for r in rows],
            edgecolor=MOCHA["surface"], linewidth=0.5, height=0.62)
    ax.set_xscale("log")
    ax.set_yticks(list(spot))
    ax.set_yticklabels([r[0] for r in rows])
    ax.invert_yaxis()
    ax.axvline(1.0, color=MOCHA["ink"], linewidth=0.7)
    ax.set_xlabel("перплексия относительно своей полноточной модели, раз "
                  "(логарифмическая шкала)")
    for y, row in zip(spot, rows):
        ax.annotate(ru(row[2], 2 if row[2] < 100 else 0), xy=(row[2], y),
                    xytext=(4, 0), textcoords="offset points",
                    va="center", fontsize=7)
    ax.set_xlim(0.8, max(r[2] for r in rows) * 12)
    comma_axis(ax, "x")
    ax.grid(axis="x", color=MOCHA["overlay"], alpha=0.25, linewidth=0.4)
    ax.set_axisbelow(True)
    finish(fig, out / "ratios.png")


def fig_reference(reference: dict, out: Path):
    """Reference perplexity by bit width, both corpora, without the dead row."""
    import matplotlib.pyplot as plt
    import numpy as np

    order = ("teacher", "int4", "student")
    names = {"teacher": "учитель\n16 бит", "int4": "NF4\n4 бита",
             "student": "студент\n1,125 бита"}
    fig, ax = plt.subplots(figsize=(WIDTH, 2.1))
    spot = np.arange(len(order))
    ax.bar(spot - 0.19, [reference[k]["wikitext2"] for k in order], width=0.36,
           color=MOCHA["blue"], edgecolor=MOCHA["surface"], linewidth=0.5,
           label="WikiText-2")
    ax.bar(spot + 0.19, [reference[k]["c4"] for k in order], width=0.36,
           color=MOCHA["peach"], edgecolor=MOCHA["surface"], linewidth=0.5,
           label="C4")
    for x, key in zip(spot, order):
        for shift, corpus in ((-0.19, "wikitext2"), (0.19, "c4")):
            value = reference[key][corpus]
            ax.annotate(ru(value), xy=(x + shift, value),
                        xytext=(0, 2), textcoords="offset points",
                        ha="center", fontsize=6.5)
    ax.set_xticks(spot)
    ax.set_xticklabels([names[k] for k in order])
    ax.set_ylabel("перплексия")
    ax.set_ylim(0, reference["student"]["c4"] * 1.34)
    comma_axis(ax, "y")
    ax.legend(loc="upper left", ncol=2)
    ax.grid(axis="y", color=MOCHA["overlay"], alpha=0.25, linewidth=0.4)
    ax.set_axisbelow(True)
    finish(fig, out / "reference.png")


# ==============================
# ===  Entry point           ===
# ==============================

def load_signature(root: Path) -> dict:
    """The weight-analysis file, or the published numbers if it is not here."""
    for path in sorted(root.glob("docs/signature-*.json")):
        return json.loads(path.read_text())
    raise FileNotFoundError("docs/signature-*.json")


def load_reference(config) -> dict:
    """Measured reference perplexity, preferring the file the run wrote."""
    if config is not None:
        path = config.run_dir() / "reference.json"
        if path.exists():
            live = json.loads(path.read_text())
            out = {}
            for name, scores in live.items():
                out[name] = {c: s["perplexity"] for c, s in scores.items()}
                out[name]["bits"] = REFERENCE.get(name, {}).get("bits", 0)
            if all(k in out for k in REFERENCE):
                ui.detail(f"reference perplexity read from {path}")
                return out
    return REFERENCE


def run(config=None, out: Path | None = None) -> Path:
    """Draw every figure into docs/figures."""
    root = Path(__file__).resolve().parent.parent
    out = Path(out) if out else root / "docs" / "figures"
    out.mkdir(parents=True, exist_ok=True)

    ui.head("Figures")
    serif = style()
    ui.field("font", serif)
    ui.field("into", str(out))

    reference = load_reference(config)
    fig_deciles(load_signature(root), out)
    fig_sweep(out)
    fig_curve(out)
    fig_agreement(out)
    fig_reference(reference, out)
    fig_ratios(reference, out)

    ui.good(f"six figures written to {out}")
    return out


if __name__ == "__main__":       # usable without the rest of the package
    run()


# ==============================
# ===  Figures for the paper ===
# ==============================

def fig_bits_quality(reference: dict, out: Path):
    """Quality against bit width: where each procedure actually lands.

    The bar chart of ratios says which method is better. This says something
    the bars cannot: what a given bit width buys, and that the frontier is not
    a smooth curve - at one bit the field splits into procedures that survive
    and procedures that collapse by five orders of magnitude.
    """
    import matplotlib.pyplot as plt

    teacher = reference["teacher"]["wikitext2"]
    ours = [
        ("наивное округление", 1.125, reference["naive"]["wikitext2"] / teacher),
        ("настоящая работа", 1.125, reference["student"]["wikitext2"] / teacher),
        ("NF4", 4.0, reference["int4"]["wikitext2"] / teacher),
    ]
    theirs = [(label, bits, ratio) for label, bits, ratio, _ in PUBLISHED]

    fig, ax = plt.subplots(figsize=(WIDTH * 2.0, 3.0))
    ax.scatter([b for _, b, _ in theirs], [r for _, _, r in theirs], s=42,
               color=MOCHA["overlay"], edgecolor=MOCHA["ink"], linewidth=0.5,
               zorder=4, label="опубликованные процедуры квантования после обучения")
    ax.scatter([b for _, b, _ in ours], [r for _, _, r in ours], s=58,
               color=[MOCHA["red"], MOCHA["green"], MOCHA["blue"]],
               edgecolor=MOCHA["ink"], linewidth=0.6, zorder=5, marker="D",
               label="настоящая работа и контроли")
    ax.scatter([16], [1.0], s=58, color="white", edgecolor=MOCHA["ink"],
               linewidth=0.8, zorder=5, marker="*")
    ax.annotate("полная точность", xy=(16, 1.0), xytext=(-6, 8),
                textcoords="offset points", ha="right", fontsize=7)

    # BiLLM contributes two nearly identical points; labelling both collides.
    OFFSET = {"BiLLM, 1,09 бита": None, "BiLLM, 1,08 бита": (8, -2),
              "GPTQ, 2 бита": (8, 4), "PB-LLM, 1,70 бита": (8, -8),
              "округление, 1 бит": (8, -7), "GPTQ, 1 бит": (8, 2)}
    for label, bits, ratio in theirs + ours:
        shift = OFFSET.get(label, (8, -2))
        if shift is None:
            continue
        text = "BiLLM, 1,08-1,09 бита" if label.startswith("BiLLM") else label
        ax.annotate(text, xy=(bits, ratio), xytext=shift,
                    textcoords="offset points", fontsize=6.5,
                    color=MOCHA["ink"])
    ax.axhline(1.0, color=MOCHA["ink"], linewidth=0.6, linestyle=(0, (4, 2)))
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_xticks([1, 1.125, 1.7, 2, 4, 16])
    ax.set_xticklabels(["1", "1,125", "1,70", "2", "4", "16"])
    ax.set_xlabel("бит на вес")
    ax.set_ylabel("перплексия относительно\nсвоей полноточной модели, раз")
    ax.set_xlim(0.88, 22)
    ax.set_ylim(0.7, 10 ** 6.2)
    comma_axis(ax, "y")
    ax.grid(color=MOCHA["overlay"], alpha=0.22, linewidth=0.4)
    ax.set_axisbelow(True)
    # Away from the top right: legend markers there read as data points.
    ax.legend(loc="center left", bbox_to_anchor=(0.30, 0.62), fontsize=6.5)
    finish(fig, out / "bits-quality.png")


def fig_memory(out: Path):
    """Where the model's bytes are, before and after.

    Explains the gap between the two compression numbers: the blocks shrink
    14.22-fold, the rest does not shrink at all, and after binarisation the
    unshrinkable part is five sixths of what ships. Totals match table IV.
    """
    import matplotlib.pyplot as plt

    # MB, as measured: blocks, and everything that stays in 16 bits.
    ROWS = [("16 бит\nвезде", 840.0, 296.9),
            ("1,125 бита\nв блоках", 59.1, 296.8)]

    fig, ax = plt.subplots(figsize=(WIDTH, 2.0))
    spot = [0, 1]
    ax.barh(spot, [r[1] for r in ROWS], height=0.55, color=MOCHA["blue"],
            edgecolor=MOCHA["surface"], linewidth=0.5,
            label="матрицы блоков (бинаризуются)")
    ax.barh(spot, [r[2] for r in ROWS], left=[r[1] for r in ROWS], height=0.55,
            color=MOCHA["peach"], edgecolor=MOCHA["surface"], linewidth=0.5,
            label="остальное: векторные представления,\nвыходной слой, нормализации")

    for y, (_, blocks, rest) in zip(spot, ROWS):
        # A slice under ~12% of the axis cannot hold its own label.
        if blocks > 150:
            ax.annotate(ru(blocks, 1), xy=(blocks / 2, y), ha="center",
                        va="center", fontsize=7)
        else:
            ax.annotate(ru(blocks, 1), xy=(blocks, y), xytext=(0, 13),
                        textcoords="offset points", ha="center", fontsize=7,
                        arrowprops=dict(arrowstyle="-", lw=0.5,
                                        color=MOCHA["surface"]))
        ax.annotate(ru(rest, 1), xy=(blocks + rest / 2, y), ha="center",
                    va="center", fontsize=7)
        ax.annotate(f"{ru(blocks + rest, 1)} МБ", xy=(blocks + rest, y),
                    xytext=(5, 0), textcoords="offset points", va="center",
                    fontsize=7.5)

    ax.set_yticks(spot)
    ax.set_yticklabels([r[0] for r in ROWS])
    ax.set_ylim(1.75, -0.75)
    ax.set_xlabel("объём хранения, МБ")
    ax.set_xlim(0, 1400)
    comma_axis(ax, "x")
    ax.legend(loc="lower right", fontsize=6.5, borderpad=0.2,
              labelspacing=0.35)
    ax.grid(axis="x", color=MOCHA["overlay"], alpha=0.22, linewidth=0.4)
    ax.set_axisbelow(True)
    finish(fig, out / "memory.png")

"""Checks that run in seconds on any machine, with no network and no download.

The expensive failures in this kind of code are the quiet ones: a
straight-through estimator that passes no gradient, a chunked loss whose
gradient does not match the loss it claims to compute, a resume that silently
starts from scratch. None of those announce themselves — the run simply spends a
day and produces a model no better than the untrained baseline.

So they are checked directly here, against small tensors, before anything long
is started.
"""

from __future__ import annotations

import contextlib
import io
import tempfile
from pathlib import Path

from . import ui

CASES: list[tuple[str, callable]] = []


def case(name: str):
    def wrap(fn):
        CASES.append((name, fn))
        return fn
    return wrap


# ==============================
# ===  Binary layer          ===
# ==============================

@case("group size divides every row")
def _groups():
    from .binary import pick_group

    for width, wanted in ((1024, 128), (3072, 128), (1536, 128), (2048, 128),
                          (576, 64), (100, 4), (7, 1), (1, 1)):
        got = pick_group(width)
        assert got == wanted, f"pick_group({width}) gave {got}, wanted {wanted}"

    # The contract is three things at once, and a width that happens to divide
    # itself can satisfy the first while quietly breaking the other two.
    for width in list(range(1, 600)) + [1024, 1536, 2048, 3072, 4096, 8192]:
        group = pick_group(width)
        assert width % group == 0, f"group {group} does not divide {width}"
        assert group & (group - 1) == 0, f"group {group} is not a power of two"
        assert group <= 128, f"group {group} exceeds the preferred 128"


@case("forward really is one bit per weight")
def _binary_forward():
    import torch
    import torch.nn as nn

    from .binary import BinaryLinear

    layer = BinaryLinear(nn.Linear(64, 32, bias=False), group=16)
    weight = layer.quantised()
    grouped = weight.view(32, -1, 16)
    magnitudes = grouped.abs()
    # Every weight in a group has the same magnitude, and only the sign varies.
    spread = (magnitudes - magnitudes[:, :, :1]).abs().max().item()
    assert spread < 1e-6, spread
    assert set(torch.sign(grouped).unique().tolist()) <= {-1.0, 1.0}


@case("gradients reach the master weight and the scale")
def _gradients():
    import torch
    import torch.nn as nn

    from .binary import BinaryLinear

    layer = BinaryLinear(nn.Linear(32, 16, bias=False), group=8, clip=1.0)
    out = layer(torch.randn(4, 32))
    out.square().mean().backward()
    assert layer.master.grad is not None and layer.master.grad.abs().sum() > 0
    assert layer.log_scale.grad is not None and layer.log_scale.grad.abs().sum() > 0


@case("straight-through window stops runaway masters")
def _window():
    import torch
    import torch.nn as nn

    from .binary import BinaryLinear

    linear = nn.Linear(16, 8, bias=False)
    layer = BinaryLinear(linear, group=8, clip=1.0)
    with torch.no_grad():
        layer.master[0, 0] = 1e4          # far outside the window
        layer.master[0, 1] = 1e-6         # comfortably inside
    layer(torch.randn(8, 16)).square().mean().backward()
    assert layer.master.grad[0, 0].item() == 0.0
    assert layer.master.grad[0, 1].item() != 0.0


@case("a binary layer can actually learn")
def _learns():
    import torch
    import torch.nn as nn

    from .binary import BinaryLinear

    torch.manual_seed(0)
    target = nn.Linear(32, 32, bias=False)
    layer = BinaryLinear(nn.Linear(32, 32, bias=False), group=8)
    optimiser = torch.optim.AdamW(layer.parameters(), lr=1e-2)
    inputs = torch.randn(128, 32)
    with torch.no_grad():
        wanted = target(inputs)

    first = last = None
    for step in range(200):
        optimiser.zero_grad()
        loss = (layer(inputs) - wanted).square().mean()
        loss.backward()
        optimiser.step()
        if step == 0:
            first = loss.item()
        last = loss.item()
    assert last < first * 0.9, f"loss went {first:.4f} -> {last:.4f}"


@case("naive baseline is sign times mean magnitude")
def _naive():
    import torch

    from .binary import naive_quantise

    weight = torch.randn(8, 32)
    quantised = naive_quantise(weight, group=32)
    assert torch.equal(torch.sign(quantised), torch.sign(weight))
    expected = weight.abs().mean(dim=1)
    assert torch.allclose(quantised.abs()[:, 0], expected, atol=1e-6)


@case("model surgery finds the blocks")
def _surgery():
    import torch.nn as nn

    from .binary import BinaryLinear, binarise_model

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn = nn.Module()
            self.self_attn.q_proj = nn.Linear(32, 32, bias=False)
            self.mlp = nn.Module()
            self.mlp.up_proj = nn.Linear(32, 64, bias=False)

    class Toy(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = nn.Module()
            self.model.layers = nn.ModuleList([Block() for _ in range(3)])
            self.lm_head = nn.Linear(32, 100, bias=False)

    toy = Toy()
    replaced = binarise_model(toy, group=32)
    assert len(replaced) == 6, replaced
    assert isinstance(toy.model.layers[0].self_attn.q_proj, BinaryLinear)
    assert isinstance(toy.lm_head, nn.Linear), "the head must stay full precision"


# ==============================
# ===  Loss                  ===
# ==============================

@case("chunked loss gradient equals the whole-sequence one")
def _chunked_loss():
    for temperature in (1.3, 1.0):
        _chunked_loss_at(temperature)


def _chunked_loss_at(temperature: float):
    """Temperature 1.0 takes a different branch for the cross-entropy term."""
    import torch
    import torch.nn.functional as F

    from .train import distil_backward

    torch.manual_seed(0)
    batch, length, vocab = 2, 24, 50
    alpha, chunk = 0.7, 7

    teacher = torch.randn(batch, length, vocab)
    gold = torch.randint(0, vocab, (batch, length))
    logits = torch.randn(batch, length, vocab, requires_grad=True)

    # Reference: one loss over the whole window, no chunking anywhere.
    s = F.log_softmax(logits / temperature, dim=-1)
    t = F.log_softmax(teacher / temperature, dim=-1)
    kl = (t.exp() * (t - s)).sum(-1).mean() * temperature * temperature
    ce = F.cross_entropy(logits.flatten(0, 1), gold.flatten())
    (alpha * kl + (1 - alpha) * ce).backward()
    reference = logits.grad.clone()

    logits.grad = None
    other = logits.detach().clone().requires_grad_(True)
    kl_value, ce_value = distil_backward(
        other, teacher, gold, alpha, temperature, chunk)

    assert torch.allclose(other.grad, reference, atol=1e-6), \
        (other.grad - reference).abs().max().item()
    assert abs(kl_value - kl.item()) < 1e-4, (kl_value, kl.item())
    assert abs(ce_value - ce.item()) < 1e-4, (ce_value, ce.item())


@case("micro-batch probe backs off to what fits")
def _micro_batch():
    import torch

    from .train import fit_micro_batch

    for limit in (1, 3, 7, 16, 100):
        def trial(size, limit=limit):
            if size > limit:
                raise torch.OutOfMemoryError("CUDA out of memory")

        got = fit_micro_batch(trial, 1, 32, "cuda")
        assert got <= limit, f"probe chose {got} where only {limit} fits"
        assert got >= min(limit, 1)

    # A failure that is not about memory must not be mistaken for one.
    def broken(size):
        raise RuntimeError("shapes do not match")

    try:
        fit_micro_batch(broken, 1, 32, "cuda")
    except RuntimeError as problem:
        assert "shapes" in str(problem)
    else:
        raise AssertionError("an unrelated error was swallowed")


@case("loss works with no teacher at all")
def _no_teacher():
    import torch
    import torch.nn.functional as F

    from .train import distil_backward

    torch.manual_seed(0)
    batch, length, vocab = 2, 16, 40
    gold = torch.randint(0, vocab, (batch, length))
    logits = torch.randn(batch, length, vocab, requires_grad=True)

    reference = F.cross_entropy(logits.flatten(0, 1), gold.flatten())
    reference.backward()
    wanted = logits.grad.clone()

    logits.grad = None
    other = logits.detach().clone().requires_grad_(True)
    kl, ce = distil_backward(other, None, gold, 0.9, 1.0, 5)

    assert kl == 0.0, "there is no teacher to diverge from"
    assert abs(float(ce) - reference.item()) < 1e-4
    assert torch.allclose(other.grad, wanted, atol=1e-6)


@case("perplexity is measurable with no teacher")
def _measure_alone():
    import math

    import torch
    import torch.nn as nn

    from . import evaluate

    class Tiny(nn.Module):
        """Enough of a language model for `measure`: ids in, logits out."""

        def __init__(self, vocab: int):
            super().__init__()
            self.table = nn.Embedding(vocab, vocab)

        def forward(self, ids):
            return type("Out", (), {"logits": self.table(ids)})()

    torch.manual_seed(0)
    vocab, windows, length = 32, 2, 16
    model = Tiny(vocab)
    inputs = torch.randint(0, vocab, (windows, length))
    targets = torch.randint(0, vocab, (windows, length))

    alone = evaluate.measure(model, None, inputs, targets, "cpu", chunk=8)
    assert alone["tokens"] == windows * length
    assert math.isfinite(alone["perplexity"]) and alone["perplexity"] > 1.0
    # Columns that need a teacher are absent, not zero: a zero would read as a
    # real measurement in metrics.jsonl.
    for missing in ("teacher_perplexity", "agreement", "kl"):
        assert missing not in alone, missing

    # With a teacher the same call still fills the comparison in, and the
    # model's own perplexity does not change because of it.
    paired = evaluate.measure(model, Tiny(vocab), inputs, targets, "cpu",
                              chunk=8)
    assert abs(paired["perplexity"] - alone["perplexity"]) < 1e-6
    assert 0.0 <= paired["agreement"] <= 1.0
    assert paired["kl"] > 0

    # A model measured against itself agrees everywhere and diverges nowhere.
    same = evaluate.measure(model, model, inputs, targets, "cpu", chunk=8)
    assert same["agreement"] == 1.0
    assert abs(same["kl"]) < 1e-6


@case("dolma subsets are interleaved, unknown ones refused")
def _dolma_subsets():
    from . import config as config_module
    from .data import _dolma_urls

    urls = ["https://x/dolma-v1_7/books/books-0000.json.gz",
            "https://x/dolma-v1_7/books/books-0001.json.gz",
            "https://x/dolma-v1_7/c4-filtered/c4-0000.json.gz",
            "https://x/dolma-v1_7/c4-filtered/c4-0001.json.gz"]

    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)
        (root / "dolma-v1_7-urls.txt").write_text("\n".join(urls))
        loaded = config_module.load(preset="smoke", overrides=[
            f"paths.corpus={root}", "data.kind=dolma"])

        loaded.set("data.subsets", ["books", "c4-filtered"])
        mixed = _dolma_urls(loaded)
        assert [u.split("/")[-2] for u in mixed] == \
            ["books", "c4-filtered", "books", "c4-filtered"], mixed

        loaded.set("data.subsets", ["nope"])
        try:
            _dolma_urls(loaded)
        except ValueError as problem:
            assert "nope" in str(problem) and "books" in str(problem)
        else:
            raise AssertionError("an unknown subset was accepted")


@case("closeness to the teacher is measured where the teacher was sure")
def _closeness_metrics():
    """A single top-1 rate hides the difference that matters.

    Missing the teacher's choice on a token it was unsure about is cheap;
    missing it where the teacher was certain is the damage. The same goes for
    near misses: a choice pushed to second place and one dropped out of the
    ranking entirely both count as one disagreement.
    """
    import torch

    from . import evaluate

    class Fixed:
        """Stands in for a model: hands back the logits it was built with."""

        def __init__(self, logits):
            self.logits = logits

        def __call__(self, ids):
            return self

        def eval(self):
            return self

        def train(self, mode=True):
            return self

        training = False

    # Four positions. The teacher is certain about three of them; the student
    # repeats one of those, misses one but keeps it within reach, and drops
    # the last out of its top five altogether.
    teacher = torch.tensor([[[9.0, 0, 0, 0, 0, 0],
                             [8.0, 0, 0, 0, 0, 0],
                             [0.6, 0.5, 0.4, 0, 0, 0],
                             [0, 0, 0, 0, 0, 5.0]]])
    student = torch.tensor([[[9.0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 9.0],
                             [0, 9.0, 0, 0, 0, 0],
                             [0, 1.0, 2.0, 3.0, 4.0, -9.0]]])
    ids = torch.zeros((1, 4), dtype=torch.long)
    gold = torch.zeros((1, 4), dtype=torch.long)

    scores = evaluate.measure(Fixed(student), Fixed(teacher), ids, gold, "cpu",
                              chunk=4)

    assert abs(scores["agreement"] - 0.25) < 1e-6, scores["agreement"]
    assert abs(scores["agreement_top5"] - 0.75) < 1e-6, scores["agreement_top5"]
    assert abs(scores["certain_tokens"] - 0.75) < 1e-6, scores["certain_tokens"]
    # One of the three certain positions was repeated.
    assert abs(scores["agreement_when_certain"] - 1 / 3) < 1e-6, \
        scores["agreement_when_certain"]
    assert 0.24 < scores["teacher_choice_probability"] < 0.26, \
        scores["teacher_choice_probability"]


@case("the compression ratio is quoted over the whole model")
def _footprint_arithmetic():
    """Sixteen to one holds for a weight inside a block, not for the model.

    The embedding table, the output head and the norms are never binarised, and
    in a small model they are a quarter of the weights. Quoting the per-weight
    ratio for the model overstates it by four times, which is the kind of claim
    a reviewer checks first.
    """
    from . import footprint

    # One bit per weight plus one half-precision scale per group of 128.
    assert footprint.group_bytes(128, 128) == 16 + 2
    # A partial group still costs a whole scale, a partial byte a whole byte.
    assert footprint.group_bytes(1, 128) == 1 + 2
    assert footprint.group_bytes(129, 128) == 17 + 4

    # The real shape of Qwen3-0.6B: 440.4M weights in blocks out of 599.5M.
    split = {"binary_weights": 440_400_000,
             "full_weights": 599_500_000 - 440_400_000,
             "scale_values": 440_400_000 // 128, "group": 128, "parts": {}}
    priced = footprint.price(split)

    # 1 + 16/128 bits per weight is 14.22 times smaller than 16 bits.
    assert 14.1 < priced["blocks_only_ratio"] < 14.3, priced
    # End to end it is a little over three, because a quarter of the weights
    # never shrank.
    assert 3.0 < priced["whole_model_ratio"] < 3.3, priced
    assert 0.26 < priced["full_precision_share"] < 0.27, priced
    assert priced["whole_model_ratio"] < priced["blocks_only_ratio"] / 4


@case("benchmark scores are read out of whatever the harness returns")
def _bench_parsing():
    """The harness names its metrics by the filter that produced them.

    Keys come back as `acc_norm,none` rather than `acc_norm`, and which of the
    two a task offers depends on the task. Reading a fixed key gives an empty
    table and a night of machine time with nothing to show for it.
    """
    from . import bench

    block = {
        "arc_challenge": {"acc,none": 0.30, "acc_norm,none": 0.33,
                          "acc_stderr,none": 0.01},
        "piqa": {"acc,none": 0.70},
        "winogrande": {"acc": 0.55},
        "broken": {"alias": "broken"},
    }
    scores = bench.collect({"results": block})

    # Normalised accuracy wins where a task reports both.
    assert scores["arc_challenge"] == 0.33, scores
    assert scores["piqa"] == 0.70, scores
    assert scores["winogrande"] == 0.55, scores
    assert "broken" not in scores, scores
    assert bench.pick_score({"acc_stderr,none": 0.01}) is None

    # Asking for one wide task returns its aggregate and all of its parts;
    # averaging over both would weigh that task by the number of its parts.
    wide = bench.collect({"results": {
        "mmlu": {"acc,none": 0.30},
        "mmlu_anatomy": {"acc,none": 0.90},
        "mmlu_astronomy": {"acc,none": 0.90},
        "piqa": {"acc,none": 0.70},
    }})
    assert abs(bench.average(wide) - 0.50) < 1e-9, bench.average(wide)


@case("the newest checkpoint is the one written last")
def _checkpoint_order():
    """Step numbers are not comparable across runs of the same directory.

    A run resumed with a bigger micro-batch renumbers its steps from a smaller
    figure, so a file left by an earlier run can hold a larger number than
    anything the new run reaches. Picked by name, that stale file stays the
    newest for ever: `eval` measures it, `export` ships it, and a resumed run
    rewinds to it.
    """
    import time as clock

    from . import checkpoint

    with tempfile.TemporaryDirectory() as room:
        run_dir = Path(room)
        old = run_dir / "step610351.pt"
        old.write_bytes(b"older run, larger number")
        clock.sleep(0.01)
        fresh = run_dir / "step19000.pt"
        fresh.write_bytes(b"this run, smaller number")

        assert checkpoint.latest(run_dir) == fresh, checkpoint.latest(run_dir)

        # And the same order decides what gets deleted: keeping one must keep
        # the file just written, not the one with the biggest name.
        kept = checkpoint._slots(run_dir)[-1:]
        assert kept == [fresh], kept

    # Several checkpoints written in one moment share a modification time to
    # whatever precision the filesystem keeps, and ordering on time alone
    # leaves them in no particular order - enough to delete the wrong one. The
    # step number settles it, which is correct inside a single run.
    import os

    with tempfile.TemporaryDirectory() as room:
        run_dir = Path(room)
        stamp = clock.time()
        for step in (10, 20, 30, 40):
            written = run_dir / f"step{step}.pt"
            written.write_bytes(b"same moment")
            os.utime(written, (stamp, stamp))

        assert checkpoint.latest(run_dir).name == "step40.pt"
        keeping_two = [p.name for p in checkpoint._slots(run_dir)[-2:]]
        assert keeping_two == ["step30.pt", "step40.pt"], keeping_two


@case("resuming counts tokens, at whatever the step size turns out to be")
def _resume_arithmetic():
    """A step is worth different numbers of tokens on different machines,
    because the micro-batch is probed per card. Both ends of the loop are
    counted in steps, so both have to be derived from tokens after that probe
    - and getting only one of them right is worse than getting neither.
    """
    def plan(seen, micro, accum=8, seq=1024, world=2, budget=10_000_000_000):
        per_step = micro * accum * seq * world
        return seen // per_step, max(1, budget // per_step), per_step

    # The real case: 300M tokens done, probed at 32 where the last machine
    # managed 1. Two thirds of the budget must still be ahead.
    start, total, per_step = plan(300_000_000, 32)
    assert start == 572, start
    assert total == 19073, total
    left = (total - start) * per_step
    assert left > 9_000_000_000, left

    # Taking the start from a step size of 1 - the value before the probe -
    # is what ended a ten billion token run after four hundred million.
    stale, _, _ = plan(300_000_000, 1)
    assert stale == 18310, stale
    assert (total - stale) * per_step < 500_000_000

    # A fresh run starts at zero however the probe lands.
    for micro in (1, 4, 32):
        assert plan(0, micro)[0] == 0

    # And a budget already spent leaves nothing to do, rather than wrapping.
    start, total, _ = plan(10_000_000_000, 32)
    assert start >= total


@case("learning rate warms up then decays")
def _schedule():
    from .train import learning_rate_factor

    assert learning_rate_factor(0, 100, 1000) < 0.02
    assert abs(learning_rate_factor(99, 100, 1000) - 1.0) < 1e-6
    middle = learning_rate_factor(550, 100, 1000)
    assert 0.4 < middle < 0.7, middle
    assert learning_rate_factor(999, 100, 1000) < 0.15


@case("improvement per hour is judged over a whole window")
def _gain():
    from .train import improvement_per_hour

    hour = 3600.0
    # Nothing to compare against yet, and nothing a window old yet.
    assert improvement_per_hour([(0.0, 40.0)], 1.0) is None
    assert improvement_per_hour([(0.0, 40.0), (600.0, 39.0)], 1.0) is None

    # Two points an hour apart: four points of perplexity gained.
    steady = [(0.0, 40.0), (hour, 36.0)]
    assert abs(improvement_per_hour(steady, 1.0) - 4.0) < 1e-9

    # A jump in the last ten minutes must not be read as an hourly rate: the
    # oldest point outside the window is the anchor, not the previous check.
    noisy = [(0.0, 40.0), (hour, 39.8), (hour + 600, 36.0)]
    rate = improvement_per_hour(noisy, 1.0)
    assert 3.0 < rate < 3.5, rate

    # Perplexity going the wrong way reads as a negative rate, which is below
    # any positive threshold, so the run stops.
    worse = [(0.0, 36.0), (hour, 38.0)]
    assert improvement_per_hour(worse, 1.0) < 0

    # A half-hour window sees the same run sooner.
    assert improvement_per_hour(steady, 0.5) is not None

    # The anchor is the most recent check a window old, not the first one.
    # Training falls steeply at the start, so averaging since the beginning
    # would report a run as improving long after it had flattened out.
    settled = [(0.0, 100.0), (hour, 40.0), (2 * hour, 39.0), (3 * hour, 38.0)]
    rate = improvement_per_hour(settled, 1.0)
    assert abs(rate - 1.0) < 1e-9, rate


@case("a lone slow check does not end the run")
def _patience():
    """The streak rule as the loop applies it, not a copy of it: these are the
    two functions train.run() calls.
    """
    from .train import slow_streak_after, streak_is_enough

    threshold, patience = 0.05, 2

    def stops_at(gains, keep=patience):
        streak = 0
        for i, gain in enumerate(gains):
            streak = slow_streak_after(streak, gain, threshold)
            if streak_is_enough(streak, keep):
                return i
        return None

    # A real run's tail with one unlucky measurement in the middle of it.
    assert stops_at([0.4, 0.5, 0.01, 0.45, 0.3]) is None
    # Genuinely flat: two in a row, and it ends on the second.
    assert stops_at([0.4, 0.3, 0.02, 0.01, 0.3]) == 3
    # Perplexity moving the wrong way twice counts as flat too.
    assert stops_at([0.4, -0.1, -0.2]) == 2
    # Nothing below the threshold, nothing happens.
    assert stops_at([0.4, 0.3, 0.06, 0.51]) is None

    # Too early for a rate: the streak is neither advanced nor broken.
    assert slow_streak_after(1, None, threshold) == 1
    assert stops_at([0.01, None, 0.01]) == 2

    # Patience below one would otherwise make an empty streak sufficient and
    # end a healthy run at its first check.
    assert not streak_is_enough(0, 0)
    assert not streak_is_enough(0, -3)
    assert stops_at([0.4, 0.5, 0.6], keep=0) is None
    assert stops_at([0.01], keep=0) == 0


@case("each process of a multi-card run takes its own card")
def _rank_device():
    from .train import rank_device

    # One process: whatever was asked for, including an explicit index.
    assert rank_device("cuda", 0, False) == "cuda"
    assert rank_device("cuda:1", 0, False) == "cuda:1"
    assert rank_device("cpu", 0, False) == "cpu"

    # Several processes: one card each, by local rank. Honouring a configured
    # index here would put every process on that one card - two copies of the
    # model on it, the rest idle, and most likely out of memory.
    assert rank_device("cuda", 0, True) == "cuda:0"
    assert rank_device("cuda", 1, True) == "cuda:1"
    assert rank_device("cuda:1", 0, True) == "cuda:0"
    assert rank_device("cuda:1", 1, True) == "cuda:1"

    # Nothing to spread over on a processor or on mps.
    assert rank_device("cpu", 1, True) == "cpu"
    assert rank_device("mps", 1, True) == "mps"


@case("no name is read before it is assigned")
def _unbound():
    """A banner that prints a variable defined further down raises
    UnboundLocalError the moment the command runs, and nothing short of
    actually running it notices. This reads the syntax tree instead, so the
    check works on a machine with no card and no torch.
    """
    import ast

    scopes = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef,
              ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)

    def own(fn):
        # Everything in this function except nested scopes: a comprehension
        # binds its own names, and a nested def runs later, not here.
        stack = list(ast.iter_child_nodes(fn))
        while stack:
            node = stack.pop()
            if isinstance(node, scopes):
                continue
            yield node
            stack.extend(ast.iter_child_nodes(node))

    # Our own sources only. rglob from the project root would walk venv/ as
    # well: .gitignore keeps it out of git, not out of a directory walk, and
    # the standard library alone trips this heuristic fifteen times over
    # `except ... as exc` and `while True`, so `make all` would fail at the
    # selftest step on every machine that has actually installed the venv.
    root = Path(__file__).resolve().parent
    files = sorted(root.glob("*.py")) + [root.parent / "main.py"]
    problems = []
    for path in files:
        if not path.exists():
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), str(path))
        except (SyntaxError, UnicodeDecodeError) as problem:
            problems.append(f"{path.name}: cannot be parsed ({problem})")
            continue
        for fn in ast.walk(tree):
            if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            stored, loaded = {}, {}
            for node in own(fn):
                if isinstance(node, ast.Name) and isinstance(
                        node.ctx, (ast.Store, ast.Load)):
                    side = stored if isinstance(node.ctx, ast.Store) else loaded
                    side[node.id] = min(side.get(node.id, 1 << 30), node.lineno)
                elif isinstance(node, (ast.Global, ast.Nonlocal)):
                    for name in node.names:
                        stored[name] = 0
            names = {a.arg for a in fn.args.args + fn.args.kwonlyargs
                     + fn.args.posonlyargs}
            for extra in (fn.args.vararg, fn.args.kwarg):
                if extra:
                    names.add(extra.arg)
            for name, store in stored.items():
                first = loaded.get(name)
                if name not in names and first is not None and first < store:
                    problems.append(f"{path.name}:{first} {fn.name}(): "
                                    f"'{name}' read before line {store}")
    assert not problems, "; ".join(problems)


@case("config presets and overrides apply")
def _config():
    from . import config as config_module

    loaded = config_module.load(preset="smoke", overrides=[
        "train.lr=1e-5",
        'data.subsets=["books","c4-filtered"]',
        "eval.windows=4"])
    assert loaded.get("data.subsets") == ["books", "c4-filtered"], \
        f"a list on the command line arrived as {loaded.get('data.subsets')!r}"
    assert loaded.get("eval.windows") == 4
    assert config_module.load(preset="smoke",
                              overrides=["data.subsets=books,pes2o"]
                              ).get("data.subsets") == ["books", "pes2o"]
    assert loaded.get("model.teacher") == "HuggingFaceTB/SmolLM2-135M"
    assert loaded.get("train.lr") == 1e-5
    assert loaded.get("train.alpha") == 0.9, "defaults must survive a preset"
    assert loaded.path("paths.runs").is_absolute()


@case("checkpoint survives a round trip")
def _checkpoint():
    import torch
    import torch.nn as nn

    from . import checkpoint as ckpt
    from .binary import BinaryLinear

    layer = BinaryLinear(nn.Linear(32, 16, bias=False), group=8)
    optimiser = torch.optim.AdamW(layer.parameters(), lr=1e-3)
    layer(torch.randn(4, 32)).square().mean().backward()
    optimiser.step()

    with tempfile.TemporaryDirectory() as folder:
        run_dir = Path(folder)
        for step in (10, 20, 30, 40):
            ckpt.save(run_dir, layer, optimiser, step, step * 1000, keep=2)
        remaining = sorted(p.name for p in run_dir.glob("step*.pt"))
        assert remaining == ["step30.pt", "step40.pt"], remaining

        wanted = layer.master.detach().clone()
        with torch.no_grad():
            layer.master.zero_()
        restored = ckpt.load_into(layer, optimiser, ckpt.latest(run_dir))
        assert restored == 40
        assert torch.allclose(layer.master, wanted)


@case("the hub cache lives inside the project")
def _cache_location():
    """A home directory owned by someone else cannot hold a lock file, and a
    download that cannot take its lock waits rather than failing.
    """
    import os

    from . import config as config_module
    from . import models

    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)
        (root / "config.toml").write_text(
            '[paths]\ncache = "assets/cache"\nmodels = "assets/models"\n'
            'corpus = "assets/corpus"\nruns = "runs"\n'
            '[model]\nteacher = "x/y"\n')
        config = config_module.load(root / "config.toml", None, [])

        keep = {name: os.environ.get(name)
                for name in ("HF_HOME", "HF_HUB_CACHE", "TRANSFORMERS_CACHE")}
        try:
            for name in keep:
                os.environ.pop(name, None)
            chosen = models.use_local_cache(config)
            assert chosen is not None
            assert chosen.is_dir(), chosen
            # Relative paths resolve against the project, not against wherever
            # the config file happens to sit, so the cache travels with the
            # folder - the point of the whole arrangement.
            project = Path(__file__).resolve().parent.parent
            assert project in chosen.parents, chosen
            assert os.environ["HF_HOME"] == str(chosen)

            # An HF_HOME the user set is a deliberate choice; leave it alone.
            os.environ["HF_HOME"] = "/somewhere/of/my/own"
            assert models.use_local_cache(config) is None
            assert os.environ["HF_HOME"] == "/somewhere/of/my/own"
        finally:
            for name, value in keep.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value


@case("an unreachable source is reported, not waited on")
def _unreachable():
    """Two hundred files behind a blocked host is a day of silent timeouts;
    it has to be one message instead.
    """
    from . import data

    class Config:
        def __init__(self):
            self.values = {"data.timeout": 1, "data.retries": 1}

        def get(self, key, default=None):
            return self.values.get(key, default)

    urls = [f"https://blocked.example/dolma/books/books-{i:04d}.json.gz"
            for i in range(200)]
    original_urls, original_check = data._dolma_urls, data.check_reachable
    data._dolma_urls = lambda config: urls
    data.check_reachable = lambda url, timeout=20.0: "TimeoutError: timed out"
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            try:
                next(iter(data._dolma_documents(Config(), 0, 0)))
            except ConnectionError as problem:
                message = str(problem)
            else:
                raise AssertionError("an unreachable host must be reported")
    finally:
        data._dolma_urls, data.check_reachable = original_urls, original_check

    # The message has to carry the way out, not just the failure.
    assert "blocked.example" in message
    assert "HTTPS_PROXY" in message
    assert "fineweb" in message


@case("a dropped connection resumes instead of ending the fetch")
def _shard_retry():
    """Reading hundreds of gigabytes, a read will time out sooner or later.
    The fetch has to survive it without losing or repeating documents.
    """
    import json

    from . import data

    lines = [json.dumps({"text": f"document {i}"}).encode() for i in range(10)]
    slept = []

    def breaks_once(_url, state={"tries": 0}):
        state["tries"] += 1
        for i, line in enumerate(lines):
            # First attempt dies halfway, as a timed-out socket would.
            if state["tries"] == 1 and i == 6:
                raise TimeoutError("the read operation timed out")
            yield line

    with contextlib.redirect_stdout(io.StringIO()):
        got = list(data._shard_batches(breaks_once, "u/f.gz", 0, 5,
                                       slept.append, batch_size=2))

    texts = [t for _, batch in got for t in batch]
    assert texts == [f"document {i}" for i in range(10)], texts
    assert slept, "a retry should wait before reconnecting"

    # Every attempt failing gives up on this file rather than the whole run.
    def always_breaks(_url):
        raise TimeoutError("down")
        yield  # pragma: no cover - generator marker

    with contextlib.redirect_stdout(io.StringIO()):
        assert list(data._shard_batches(always_breaks, "u/f.gz", 0, 3,
                                        slept.append)) == []

    # Resuming a shard skips what was already handed out.
    def clean(_url):
        yield from lines

    with contextlib.redirect_stdout(io.StringIO()):
        rest = list(data._shard_batches(clean, "u/f.gz", 7, 3, slept.append,
                                        batch_size=2))
    assert [t for _, b in rest for t in b] == ["document 7", "document 8",
                                               "document 9"]


@case("time left is measured on this run, not on the whole file")
def _progress_eta():
    """Resuming a download that is already most of the way through, the bar
    divided everything ever written by the seconds since start-up and
    reported minutes where there were hours.
    """
    bar = ui.Progress("tokenising", 100.0, done=90.0)
    bar.started -= 10.0            # ten seconds of this run
    with contextlib.redirect_stdout(io.StringIO()) as printed:
        bar.update(92.0, force=True)
    text = printed.getvalue()

    # Two units in ten seconds, eight to go: forty seconds, not none.
    assert "1m left" in text, text

    # Without the carried-over figure the same call is wildly optimistic,
    # which is exactly the bug.
    naive = ui.Progress("tokenising", 100.0)
    naive.started -= 10.0
    with contextlib.redirect_stdout(io.StringIO()) as printed:
        naive.update(92.0, force=True)
    assert "0m left" in printed.getvalue()


@case("workers really are separate processes")
def _parallel_fill():
    """The filler end to end, with the network and the tokeniser replaced.

    Threads could not use this machine: the tokeniser's pool stops scaling
    past a few cores and everything around it holds the GIL. This checks the
    process version writes the right tokens, in the right amount, and records
    where each shard was left.
    """
    import json
    import os

    import numpy as np

    from . import data, parallel

    if parallel.start_method() != "fork":
        return  # the stand-ins below are inherited, which needs fork

    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)
        (root / "config.toml").write_text(
            '[paths]\ncorpus = "c"\nmodels = "m"\ncache = "k"\nruns = "r"\n'
            '[model]\nteacher = "x/y"\n[data]\nkind = "olmo"\n'
            'fetch_workers = 3\ncpu_cores = 3\nbatch_documents = 2\n')
        from . import config as config_module
        config = config_module.load(root / "config.toml", None, [])

        # Four shards of four documents; document N of shard S is one token.
        def lines(url):
            shard = int(url.rsplit("-", 1)[1])
            for i in range(4):
                yield json.dumps({"text": f"{shard}:{i}"}).encode()

        class Tokeniser:
            def __call__(self, texts, add_special_tokens=False):
                return {"input_ids": [[int(t.split(":")[0]) + 1]
                                      for t in texts]}

        original_lines = data._shard_lines
        from . import models
        original_loader = models.load_tokenizer
        data._shard_lines = lambda url, timeout: lines(url)
        models.load_tokenizer = lambda config: Tokeniser()
        try:
            urls = [f"https://example/data/dclm/shard-{i}" for i in range(4)]
            target = 40
            array_path = root / "corpus.npy"
            tokens = np.lib.format.open_memmap(
                array_path, mode="w+", dtype=np.uint32, shape=(target,))
            note = {"written": 0, "shard": 0, "row": 0, "target": target}
            bar = ui.Progress("t", target)
            with contextlib.redirect_stdout(io.StringIO()):
                written = parallel.fill(tokens, note, root / "note.json", urls,
                                        config, Tokeniser(), 0, target, bar)
        finally:
            data._shard_lines = original_lines
            models.load_tokenizer = original_loader

    # Four documents and a separator each, four shards: thirty two tokens.
    assert written == 32, written
    values = np.asarray(tokens[:written]).tolist()
    # Every shard contributed all four of its documents, none of them twice.
    for shard in range(4):
        assert values.count(shard + 1) == 4, (shard, values)
    assert values.count(0) == 16, values

    # And every shard is recorded as finished, or it would be read again on
    # the next run: the end-of-shard mark travels through a feeder thread and
    # is easy to lose at shutdown.
    assert sorted(note["finished"]) == [0, 1, 2, 3], note["finished"]
    assert not note["cursors"], note["cursors"]
    assert note["written"] == 32


@case("a half-finished download continues where it stopped")
def _resume_cursors():
    """A note written by the one-shard-at-a-time filler has to mean the same
    thing to the parallel one: hundreds of gigabytes already on disk must not
    be fetched again.
    """
    from .parallel import restore_cursors

    # The old shape: everything before shard 221 is done, and 221 is part way.
    finished, cursors = restore_cursors(
        {"written": 114_000_000_000, "shard": 221, "row": 3400})
    assert finished == set(range(221))
    assert cursors == {221: 3400}
    assert 220 in finished and 221 not in finished

    # A shard boundary: nothing part way through.
    finished, cursors = restore_cursors({"shard": 7, "row": 0})
    assert finished == set(range(7))
    assert cursors == {}

    # The new shape is read as written, gaps and all - a shard that failed
    # stays unfinished and is picked up again.
    finished, cursors = restore_cursors(
        {"finished": [0, 1, 3], "cursors": {"2": 500, "4": 10}})
    assert finished == {0, 1, 3}
    assert cursors == {2: 500, 4: 10}

    # Nothing at all means start from the beginning.
    assert restore_cursors({}) == (set(), {})


@case("each corpus gets its own file")
def _corpus_names():
    """Teacher names carry dots, and with_suffix() truncates at the first one:
    every corpus would share a file and rebuild over the last one.
    """
    from . import config as config_module
    from . import data

    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)
        (root / "config.toml").write_text(
            '[paths]\ncorpus = "assets/corpus"\nmodels = "assets/models"\n'
            'cache = "assets/cache"\nruns = "runs"\n'
            '[model]\nteacher = "Qwen/Qwen3-0.6B"\n'
            '[data]\nkind = "olmo"\ntrain_tokens = 101000000\n'
            'eval_tokens = 1000000\n')

        def files(overrides):
            config = config_module.load(root / "config.toml", None, overrides)
            return data.corpus_files(config)

        array, note = files([])
        assert array.name == "Qwen_Qwen3-0.6B-olmo-102M.npy", array.name
        assert note.name == "Qwen_Qwen3-0.6B-olmo-102M.json", note.name

        # A different source, size or teacher is a different corpus.
        others = [files(["data.kind=fineweb"])[0],
                  files(["data.train_tokens=500000000"])[0],
                  files(["model.teacher=HuggingFaceTB/SmolLM2-135M"])[0]]
        names = {array.name} | {o.name for o in others}
        assert len(names) == 4, sorted(names)


@case("corpus requests carry the hub token")
def _hub_token():
    """Shards are fetched with plain urllib, which knows nothing about the
    login the rest of the tooling uses - and the hub throttles anonymous
    downloads, which is invisible except as a slow fetch.
    """
    import os

    from . import data

    keep = {name: os.environ.get(name) for name in
            ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HUGGING_FACE_HUB_TOKEN")}
    try:
        for name in keep:
            os.environ.pop(name, None)
        os.environ["HF_TOKEN"] = "hf_secret"

        hub = data.request_headers(
            "https://huggingface.co/datasets/allenai/olmo-mix-1124/x.gz")
        assert hub["Authorization"] == "Bearer hf_secret"

        # Only to the hub: a token must not be handed to any other host.
        other = data.request_headers("https://olmo-data.org/dolma/x.gz")
        assert "Authorization" not in other, other
        assert other["User-Agent"] == "bindistill"

        os.environ["HF_TOKEN"] = "  hf_padded  "
        assert data.request_headers("https://huggingface.co/x")[
            "Authorization"] == "Bearer hf_padded"
    finally:
        for name, value in keep.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


@case("a lost counter is recovered from the corpus itself")
def _recover_counter():
    """The note is small and rewritten constantly; an interrupt during the
    write leaves it empty. The tokens are still there, and finding where they
    end beats fetching hundreds of gigabytes a second time.
    """
    import numpy as np

    from .data import read_note, write_note, written_tokens

    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)
        array_path = root / "corpus.npy"
        for real in (0, 1, 5_000_000):
            array = np.zeros(8_000_000, dtype=np.uint32)
            array[:real] = np.arange(1, real + 1, dtype=np.uint32)
            np.save(array_path, array)
            assert written_tokens(array_path) == real, real

        # A note cut short reads as absent, not as a crash.
        note_path = root / "corpus.json"
        note_path.write_text("")
        with contextlib.redirect_stdout(io.StringIO()):
            assert read_note(note_path) is None
        note_path.write_text('{"written": 5, "tar')
        with contextlib.redirect_stdout(io.StringIO()):
            assert read_note(note_path) is None

        # And writing it leaves no half-written state behind.
        write_note(note_path, {"written": 7})
        assert read_note(note_path) == {"written": 7}
        assert not list(root.glob("*.new"))


@case("an unfinished corpus hands back only what was written")
def _partial_corpus():
    """The file is created at full length up front, so an interrupted fetch
    leaves zeros at the end - exactly where the held-out set is taken from.
    """
    import json

    import numpy as np

    from . import data

    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)
        written, target = 4000, 10000
        array = np.zeros(target, dtype=np.uint16)
        array[:written] = np.arange(1, written + 1, dtype=np.uint16)
        array_path = root / "corpus.npy"
        np.save(array_path, array)
        (root / "corpus.json").write_text(json.dumps(
            {"written": written, "target": target, "shard": 0, "row": 0}))

        original = data.corpus_files
        data.corpus_files = lambda config: (array_path, root / "corpus.json")
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                tokens = data.open_corpus(None)
        finally:
            data.corpus_files = original

        assert len(tokens) == written, len(tokens)
        # Nothing from the unwritten tail, so no window of zeros can be drawn.
        assert int(np.asarray(tokens).min()) > 0


@case("held-out text is never drawn for training")
def _split():
    import numpy as np

    from . import config as config_module
    from . import data

    class FakeTokenizer:
        eos_token_id = 0

        def __len__(self):
            return 300

        def __call__(self, texts, add_special_tokens=False):
            return {"input_ids": [[(ord(c) % 299) + 1 for c in t] for t in texts]}

    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)
        sample = root / "sample.txt"
        sample.write_text("\n\n".join("word " * 200 for _ in range(400)))

        loaded = config_module.load(preset="smoke", overrides=[
            "data.kind=text", f"data.text_file={sample}",
            "data.train_tokens=40000", "data.eval_tokens=8000",
            "train.seq=64", f"paths.corpus={root}"])
        with contextlib.redirect_stdout(io.StringIO()):
            data.build_corpus(loaded, FakeTokenizer())

        tokens = data.open_corpus(loaded)
        assert len(tokens) == 48000, len(tokens)

        stream = data.TokenWindows(loaded)
        assert stream.limit == 48000 - 8000 - 64 - 1
        starts = stream.rng.integers(0, stream.limit, 200)
        assert int(starts.max()) + 64 < 48000 - 8000

        inputs, targets = data.held_out_windows(loaded, 4)
        assert inputs.shape == (4, 64) and targets.shape == (4, 64)
        assert np.array_equal(inputs[0, 1:].numpy(), targets[0, :-1].numpy())


# ==============================
# ===  Runner                ===
# ==============================

def run(verbose: bool = True) -> int:
    ui.head("Self-test")
    failures = 0
    missing_dependency = None
    for name, check in CASES:
        try:
            check()
        except AssertionError as problem:
            failures += 1
            ui.fail(f"FAIL  {name}")
            if verbose:
                ui.say(f"        {problem}", "red")
        except ImportError as problem:
            failures += 1
            missing_dependency = problem.name or str(problem)
            ui.fail(f"ERROR {name}: {problem}")
        except Exception as problem:  # noqa: BLE001 - report, do not hide
            failures += 1
            ui.fail(f"ERROR {name}: {type(problem).__name__}: {problem}")
        else:
            ui.good(f"ok    {name}")

    ui.say()
    if missing_dependency:
        ui.fail(f"'{missing_dependency}' is not installed — run `make install` "
                f"before reading anything else here")
    if failures:
        ui.fail(f"{failures} of {len(CASES)} checks failed — do not start a run")
    else:
        ui.good(f"all {len(CASES)} checks passed")
    return failures

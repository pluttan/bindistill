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

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
    import torch
    import torch.nn.functional as F

    from .train import distil_backward

    torch.manual_seed(0)
    batch, length, vocab = 2, 24, 50
    alpha, temperature, chunk = 0.7, 1.3, 7

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


@case("learning rate warms up then decays")
def _schedule():
    from .train import learning_rate_factor

    assert learning_rate_factor(0, 100, 1000) < 0.02
    assert abs(learning_rate_factor(99, 100, 1000) - 1.0) < 1e-6
    middle = learning_rate_factor(550, 100, 1000)
    assert 0.4 < middle < 0.7, middle
    assert learning_rate_factor(999, 100, 1000) < 0.15


# ==============================
# ===  Plumbing              ===
# ==============================

@case("config presets and overrides apply")
def _config():
    from . import config as config_module

    loaded = config_module.load(preset="smoke", overrides=["train.lr=1e-5"])
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

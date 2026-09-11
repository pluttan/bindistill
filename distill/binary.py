"""One-bit linear layers, and the naive baseline they have to beat.

The parameterisation follows what the measurements on Bonsai pointed at: the
sign of a full-precision master weight is what reaches the matmul, the master
itself keeps receiving gradients, and the magnitude is a learned parameter per
group of 128 rather than the group's mean. Bonsai's scales sit at roughly twice
the naive value, which no closed-form formula produces, so it has to be learned.

The baseline in `naive_quantise` is the same sign rule with the mean magnitude
and no training at all. Every number this package reports is printed next to it,
because a perplexity on its own does not say whether the training did anything.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

DEFAULT_GROUP = 128


# ==============================
# ===  Grouping              ===
# ==============================

def pick_group(width: int, preferred: int = DEFAULT_GROUP) -> int:
    """Largest power-of-two group that divides the row, at most `preferred`.

    Not every model has widths divisible by 128 — SmolLM2's hidden size is 576,
    for instance — and padding a matrix to make it fit changes the model. Taking
    a smaller group instead costs a little in bits per weight and nothing in
    correctness.
    """
    ceiling = min(preferred, width)
    # Start at a power of two: min(preferred, width) is not necessarily one, and
    # starting there returns the width itself whenever it divides itself — a
    # single group per row, which is not what the caller asked for.
    group = 1 << (ceiling.bit_length() - 1)
    while group > 1 and width % group != 0:
        group //= 2
    return max(1, group)


def bits_per_weight(group: int, scale_bits: int = 16) -> float:
    """One sign bit per weight, one scale per group."""
    return 1.0 + scale_bits / group


# ==============================
# ===  Binary linear layer   ===
# ==============================

class BinaryLinear(nn.Module):
    """Linear layer that is one bit per weight at every forward pass.

    Forward uses `sign(master) * exp(log_scale)`. Backward reaches `master`
    through a straight-through estimator, and `log_scale` through the real
    multiplication, so both train. The straight-through window is relative to
    the group scale: a master weight further than `clip` scales from zero has
    already made its mind up, and letting it keep accumulating gradient only
    drives it to infinity without ever changing the sign it contributes.
    """

    def __init__(self, linear: nn.Linear, group: int = DEFAULT_GROUP,
                 clip: float = 1.0):
        super().__init__()
        weight = linear.weight.data.float()
        self.out_features, self.in_features = weight.shape
        self.group = pick_group(self.in_features, group)
        self.clip = float(clip)

        self.master = nn.Parameter(weight.clone())
        grouped = weight.view(self.out_features, -1, self.group)
        self.log_scale = nn.Parameter(
            grouped.abs().mean(dim=2).clamp_min(1e-8).log())
        self.bias = linear.bias

    def quantised(self) -> torch.Tensor:
        """The weight the matmul actually sees, with gradients wired up."""
        grouped = self.master.view(self.out_features, -1, self.group)
        signs = torch.where(grouped >= 0, 1.0, -1.0).detach()
        scale = self.log_scale.exp().unsqueeze(-1)

        quant = signs * scale
        if self.clip > 0:
            bound = (self.clip * scale).detach().expand_as(grouped)
            passthrough = torch.clamp(grouped, -bound, bound)
        else:
            passthrough = grouped
        quant = quant + (passthrough - passthrough.detach())
        return quant.view(self.out_features, self.in_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.quantised().to(x.dtype), self.bias)

    @torch.no_grad()
    def packed(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Signs as bool and one scale per group — what a real export stores."""
        grouped = self.master.view(self.out_features, -1, self.group)
        return (grouped >= 0).view(self.out_features, self.in_features), \
            self.log_scale.exp()

    def extra_repr(self) -> str:
        return (f"in={self.in_features}, out={self.out_features}, "
                f"group={self.group}, clip={self.clip}")


# ==============================
# ===  Model surgery         ===
# ==============================

def transformer_blocks(model: nn.Module) -> nn.ModuleList:
    """Find the block list without assuming a particular architecture."""
    for path in ("model.layers", "transformer.h", "model.decoder.layers"):
        node = model
        for part in path.split("."):
            node = getattr(node, part, None)
            if node is None:
                break
        if isinstance(node, nn.ModuleList) and len(node) > 0:
            return node
    for module in model.modules():
        if isinstance(module, nn.ModuleList) and len(module) > 4:
            return module
    raise RuntimeError("could not locate the transformer blocks in this model")


def binarise_model(model: nn.Module, group: int = DEFAULT_GROUP,
                   clip: float = 1.0) -> list[str]:
    """Replace every linear layer inside the blocks. Returns the names touched.

    Embeddings, the output head and the norms stay in full precision. Bonsai
    binarises its embedding table too, but that needs a custom lookup and is not
    where the quality gap lives — the blocks are.
    """
    replaced: list[str] = []
    blocks = transformer_blocks(model)
    for index, block in enumerate(blocks):
        for parent_name, parent in list(block.named_modules()):
            for name, child in list(parent.named_children()):
                if isinstance(child, nn.Linear):
                    setattr(parent, name, BinaryLinear(child, group, clip))
                    prefix = f"block{index}"
                    tail = f"{parent_name}.{name}" if parent_name else name
                    replaced.append(f"{prefix}.{tail}")
    return replaced


@torch.no_grad()
def naive_quantise(weight: torch.Tensor,
                   group: int = DEFAULT_GROUP) -> torch.Tensor:
    """Sign times the group's mean magnitude — the thing training must beat."""
    out_features, width = weight.shape
    size = pick_group(width, group)
    grouped = weight.float().view(out_features, -1, size)
    scale = grouped.abs().mean(dim=2, keepdim=True)
    signs = torch.where(grouped >= 0, 1.0, -1.0)
    return (signs * scale).view(out_features, width).to(weight.dtype)


@torch.no_grad()
def naive_binarise_model(model: nn.Module,
                         group: int = DEFAULT_GROUP) -> int:
    """Apply the untrained baseline in place, so it can be measured directly."""
    count = 0
    for block in transformer_blocks(model):
        for _, parent in list(block.named_modules()):
            for _, child in list(parent.named_children()):
                if isinstance(child, nn.Linear):
                    child.weight.data.copy_(
                        naive_quantise(child.weight.data, group))
                    count += 1
    return count


def binary_parameters(model: nn.Module) -> tuple[int, float]:
    """Weight count under a binary layer, and the average bits each costs."""
    total, bits = 0, 0.0
    for module in model.modules():
        if isinstance(module, BinaryLinear):
            count = module.out_features * module.in_features
            total += count
            bits += count * bits_per_weight(module.group)
    return total, (bits / total if total else 0.0)


def split_parameters(model: nn.Module) -> tuple[list, list]:
    """Master weights and scales train at different rates; keep them apart."""
    scales, weights = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        (scales if name.endswith("log_scale") else weights).append(param)
    return weights, scales

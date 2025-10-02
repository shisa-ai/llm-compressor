"""Online rotation helpers for SpinQuant (R3/R4)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn

from .hadamard_utils import block_hadamard_, pick_block_size


@dataclass
class HadamardConfig:
    dim: int
    block_size: Optional[int] = None

    def resolved_block(self) -> int:
        return pick_block_size(self.dim, self.block_size)


class R3KVHadamard(nn.Module):
    """Apply the same Hadamard rotation to Q and K (and optionally V cache)."""

    def __init__(self, cfg: HadamardConfig):
        super().__init__()
        self.cfg = cfg

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: Optional[torch.Tensor] = None
    ):
        block = self.cfg.resolved_block()
        q = block_hadamard_(q, block)
        k = block_hadamard_(k, block)
        if v is not None:
            v = block_hadamard_(v, block)
        return q, k, v


class R4MLPHadamard(nn.Module):
    """Apply a Hadamard before/after the MLP down projection."""

    def __init__(self, cfg: HadamardConfig):
        super().__init__()
        self.cfg = cfg

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        block = self.cfg.resolved_block()
        return block_hadamard_(x, block)

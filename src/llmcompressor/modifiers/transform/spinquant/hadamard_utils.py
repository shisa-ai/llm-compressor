"""Hadamard and block-diagonal helpers for SpinQuant."""

from __future__ import annotations

import math
from typing import Optional

import torch


def largest_power_of_two_leq(n: int) -> int:
    if n < 1:
        raise ValueError("dimension must be positive")
    return 1 << (n.bit_length() - 1)


def pick_block_size(dim: int, prefer: Optional[int] = None) -> int:
    """Choose a power-of-two block that divides `dim`."""

    if prefer and prefer > 0 and prefer & (prefer - 1) == 0 and dim % prefer == 0:
        return prefer

    block = largest_power_of_two_leq(dim)
    while block > 1 and dim % block != 0:
        block //= 2
    return max(block, 1)


def fwht_inplace(x: torch.Tensor) -> torch.Tensor:
    """Fast Walsh-Hadamard Transform in-place over the last dimension."""

    if x.numel() == 0:
        return x

    n = x.shape[-1]
    if n & (n - 1) != 0:
        raise ValueError(f"Hadamard requires power-of-two dimension, got {n}")

    h = 1
    while h < n:
        x = x.view(*x.shape[:-1], -1, 2, h)
        a = x[..., 0, :].clone()
        b = x[..., 1, :]
        x[..., 0, :] = a + b
        x[..., 1, :] = a - b
        x = x.reshape(*x.shape[:-3], -1)
        h *= 2

    x /= math.sqrt(n)
    return x


def block_hadamard_(tensor: torch.Tensor, block_size: int) -> torch.Tensor:
    """Apply block-wise Hadamard transform along the last dimension."""

    if block_size <= 1:
        return tensor

    d = tensor.shape[-1]
    n_blocks = d // block_size
    if n_blocks == 0:
        return tensor

    head = tensor[..., : n_blocks * block_size]
    head = head.view(*tensor.shape[:-1], n_blocks, block_size)
    fwht_inplace(head)
    tensor[..., : n_blocks * block_size] = head.reshape(
        *tensor.shape[:-1], n_blocks * block_size
    )
    return tensor


def hadamard_matrix(dim: int, device=None, dtype=None) -> torch.Tensor:
    block = pick_block_size(dim)
    mat = torch.eye(dim, device=device, dtype=dtype)
    block_hadamard_(mat, block)
    return mat

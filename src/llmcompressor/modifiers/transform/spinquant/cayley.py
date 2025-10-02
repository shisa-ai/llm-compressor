"""Cayley-SGD helpers for SpinQuant rotations."""

from __future__ import annotations

import torch


def _hat(G: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
    """Compute Ĝ term from SpinQuant Eq. (3)."""

    # Ĝ = G R^T - 0.5 R R^T G R^T
    Rt = R.transpose(-1, -2)
    term = G @ Rt
    correction = 0.5 * (R @ (Rt @ term))
    return term - correction


def cayley_update(R: torch.Tensor, G: torch.Tensor, lr: float) -> torch.Tensor:
    """Perform a Cayley update that preserves orthonormality."""

    if lr == 0.0:
        return R

    G_hat = _hat(G, R)
    Y = G_hat - G_hat.transpose(-1, -2)

    eye = torch.eye(R.shape[-1], device=R.device, dtype=R.dtype)
    A = eye - (lr / 2.0) * Y
    B = eye + (lr / 2.0) * Y
    BR = B @ R
    return torch.linalg.solve(A, BR)


def enforce_orthonormal(R: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Project a matrix back to the Stiefel manifold via SVD."""

    U, _, Vh = torch.linalg.svd(R, full_matrices=False)
    return (U @ Vh).to(dtype=R.dtype).clamp(min=-1.0 - eps, max=1.0 + eps)


def cayley_step_(R: torch.Tensor, G: torch.Tensor, lr: float) -> None:
    """In-place Cayley step with optional fall-back projection."""

    R.copy_(cayley_update(R, G, lr))

    if not torch.allclose(
        R.transpose(-1, -2) @ R,
        torch.eye(R.shape[-1], device=R.device, dtype=R.dtype),
        atol=1e-5,
        rtol=1e-4,
    ):
        R.copy_(enforce_orthonormal(R))

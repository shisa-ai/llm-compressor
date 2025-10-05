"""Activation quantization with straight-through estimators for SpinQuant."""

from __future__ import annotations

import torch


class STEActivationQuantize(torch.autograd.Function):
    """
    Symmetric per-token activation quantization with straight-through estimator.

    Forward: quantize activations to int precision
    Backward: pass gradients through unchanged (STE)

    This matches the SpinQuant reference implementation's STEQuantize class
    (SpinQuant/utils/quant_utils.py:61-71).
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, num_bits: int = 4) -> torch.Tensor:
        """
        Quantize activation tensor with symmetric per-token quantization.

        Args:
            x: Input activation tensor
            num_bits: Number of bits for quantization (default: 4)

        Returns:
            Quantized tensor (dequantized to float for gradient flow)
        """
        if num_bits >= 16:
            return x

        # Symmetric quantization: range is [-maxq-1, maxq]
        maxq = 2 ** (num_bits - 1) - 1

        # Per-token scale: max absolute value per token (last dim)
        scale = x.detach().abs().amax(dim=-1, keepdim=True).clamp(min=1e-6) / maxq

        # Quantize and dequantize
        scaled = x / scale
        quantized = torch.clamp(scaled.round(), -(maxq + 1), maxq)
        dequantized = scale * quantized

        return dequantized

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        """Straight-through: pass gradients unchanged."""
        return grad_output, None


def fake_quantize_activation(
    x: torch.Tensor,
    num_bits: int = 4,
    enabled: bool = True,
) -> torch.Tensor:
    """
    Apply fake quantization to activation tensor during training.

    Args:
        x: Input activation tensor
        num_bits: Number of bits for quantization
        enabled: Whether quantization is enabled

    Returns:
        Quantized tensor (or original if disabled)
    """
    if not enabled or num_bits >= 16:
        return x

    return STEActivationQuantize.apply(x, num_bits)

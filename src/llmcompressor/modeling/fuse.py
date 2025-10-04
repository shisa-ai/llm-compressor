from typing import Iterable

import torch
from compressed_tensors import align_module_device, update_offload_parameter

__all__ = ["center_embeddings", "fuse_norm_linears"]


PRECISION = torch.float32


def center_embeddings(embedding: torch.nn.Module):
    """
    Shift each embedding to have a mean of zero

    :param embedding: embedding module containing embeddings to center
    """
    if not hasattr(embedding, "weight"):
        raise ValueError(f"Cannot fuse norm of type {type(embedding)}")

    with align_module_device(embedding):
        weight_dtype = embedding.weight.dtype
        weight = embedding.weight.to(PRECISION)
        new_weight = weight - weight.mean(dim=-1, keepdim=True)
        new_weight = new_weight.to(weight_dtype)

    update_offload_parameter(embedding, "weight", new_weight)


def fuse_norm_linears(norm: torch.nn.Module, linears: Iterable[torch.nn.Linear]):
    """
    Fuse the scaling operation of norm layer into subsequent linear layers.
    This useful for ensuring transform invariance between norm and linear layers.

    Note that unitary transforms (rotation) commute with normalization, but not scaling

    :param norm: norm layer whose weight will be fused into subsequent linears
    :param linears: linear layers which directly follow the norm layer
    """
    if not hasattr(norm, "weight"):
        raise ValueError(f"Cannot fuse norm of type {type(norm)}")

    compute_device = torch.device("cpu")

    for linear in linears:
        weight_dtype = linear.weight.dtype
        with align_module_device(norm, compute_device), align_module_device(
            linear, compute_device
        ):
            norm_weight = norm.weight.to(device=compute_device, dtype=PRECISION)
            fused_weight = linear.weight.to(device=compute_device, dtype=PRECISION)
            fused_weight = fused_weight * norm_weight

        update_offload_parameter(
            linear,
            "weight",
            fused_weight.to(dtype=weight_dtype, device=compute_device),
        )
        del fused_weight

    with align_module_device(norm, compute_device):
        new_norm_weight = torch.ones_like(norm.weight, device=compute_device)
    update_offload_parameter(norm, "weight", new_norm_weight)

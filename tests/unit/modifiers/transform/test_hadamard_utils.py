import torch

from llmcompressor.modifiers.transform.spinquant.hadamard_utils import (
    block_hadamard_,
    largest_power_of_two_leq,
    pick_block_size,
)


def test_largest_power_of_two_leq():
    assert largest_power_of_two_leq(1) == 1
    assert largest_power_of_two_leq(15) == 8
    assert largest_power_of_two_leq(64) == 64


def test_pick_block_size_prefers_valid_power_of_two():
    assert pick_block_size(80) == 16
    assert pick_block_size(80, prefer=32) == 16
    assert pick_block_size(80, prefer=8) == 8


def test_block_hadamard_is_self_inverse_on_blocks():
    tensor = torch.randn(3, 10)
    rotated = block_hadamard_(tensor.clone(), 4)
    restored = block_hadamard_(rotated.clone(), 4)
    assert torch.allclose(restored, tensor, atol=1e-5)
    assert torch.allclose(rotated[..., -2:], tensor[..., -2:])

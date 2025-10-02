import torch
import torch.nn.utils.parametrize as parametrize
from torch.utils.data import DataLoader, Dataset

from llmcompressor.core.state import State
from llmcompressor.modifiers.transform.spinquant import SpinQuantModifier
from transformers import LlamaConfig, LlamaForCausalLM


class _DummyDataset(Dataset):
    def __init__(self, vocab_size: int, length: int = 8, seq_len: int = 16):
        self.vocab_size = vocab_size
        self.length = length
        self.seq_len = seq_len

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index):
        input_ids = torch.randint(
            0, self.vocab_size, (self.seq_len,), dtype=torch.long
        )
        attention_mask = torch.ones_like(input_ids)
        return {"input_ids": input_ids, "attention_mask": attention_mask}


def _build_model() -> LlamaForCausalLM:
    config = LlamaConfig(
        hidden_size=32,
        intermediate_size=64,
        num_attention_heads=4,
        num_hidden_layers=1,
        vocab_size=256,
        max_position_embeddings=64,
    )
    return LlamaForCausalLM(config)


def test_spinquant_learnable_cayley_updates_rotation():
    model = _build_model()
    dataset = _DummyDataset(model.config.vocab_size)
    dataloader = DataLoader(dataset, batch_size=2)

    modifier = SpinQuantModifier(
        learn_rotations=True,
        rotations=["R1"],
        transform_type="spinquant-learnable",
        cayley_iters=2,
        cayley_lr=0.2,
    )

    state = State()
    state.update(model=model, calib_data=dataloader)

    assert modifier.on_initialize(state)

    modifier._center_embeddings(model)
    modifier._fuse_norms(model)
    modifier._apply_transforms(model)
    modifier._collect_rotation_structures(model)
    parent_modules = list(modifier._parent_modules)

    assert modifier._rotation_params, "Expected learnable rotation parameters"

    modifier._run_cayley(state)

    rotation_after = modifier._rotation_params[0].detach().clone()

    orthogonality = rotation_after.T @ rotation_after
    identity = torch.eye(orthogonality.shape[0], dtype=orthogonality.dtype)
    assert torch.allclose(orthogonality, identity, atol=1e-4, rtol=1e-3)

    parent_modules = list(modifier._parent_modules)
    modifier._finalize_transforms(model)

    for parent in parent_modules:
        assert not parametrize.is_parametrized(parent, "weight")


def test_spinquant_learnable_r2_shares_rotation():
    model = _build_model()
    dataset = _DummyDataset(model.config.vocab_size)
    dataloader = DataLoader(dataset, batch_size=2)

    modifier = SpinQuantModifier(
        learn_rotations=True,
        rotations=["R2"],
        transform_type="spinquant-learnable",
        cayley_iters=1,
        cayley_lr=0.1,
        cayley_samples=4,
    )

    state = State()
    state.update(model=model, calib_data=dataloader)

    assert modifier.on_initialize(state)

    modifier._center_embeddings(model)
    modifier._fuse_norms(model)
    modifier._apply_transforms(model)
    modifier._collect_rotation_structures(model)
    parent_modules = list(modifier._parent_modules)

    r2_transforms = [
        transform
        for transform in modifier._rotation_transforms
        if transform.rotation_name == "R2"
    ]
    assert len(r2_transforms) == 2, "Expected R2 transforms for v_proj and o_proj"
    assert len({t.rotation_key for t in r2_transforms}) == 1
    assert len(modifier._rotation_params) == 1

    modifier._run_cayley(state)

    rotation = modifier._rotation_params[0].detach()
    ortho = rotation.T @ rotation
    identity = torch.eye(ortho.shape[0], dtype=ortho.dtype)
    assert torch.allclose(ortho, identity, atol=1e-4, rtol=1e-3)

    modifier._finalize_transforms(model)

    for parent in parent_modules:
        assert not parametrize.is_parametrized(parent, "weight")


def test_spinquant_stores_and_reloads_rotations():
    model = _build_model()
    dataset = _DummyDataset(model.config.vocab_size)
    dataloader = DataLoader(dataset, batch_size=2)

    trainer_modifier = SpinQuantModifier(
        learn_rotations=True,
        rotations=["R1", "R2"],
        transform_type="spinquant-learnable",
        cayley_iters=1,
        cayley_lr=0.1,
        cayley_samples=4,
    )

    state = State()
    state.update(model=model, calib_data=dataloader)

    assert trainer_modifier.on_initialize(state)
    trainer_modifier._center_embeddings(model)
    trainer_modifier._fuse_norms(model)
    trainer_modifier._apply_transforms(model)
    trainer_modifier._collect_rotation_structures(model)
    trainer_modifier._run_cayley(state)
    trainer_modifier._finalize_transforms(model)

    assert trainer_modifier.stored_rotations
    reference_weight = (
        model.model.layers[0].self_attn.q_proj.weight.detach().clone()
    )

    reload_modifier = SpinQuantModifier(
        rotations=["R1", "R2"],
        transform_type="spinquant-learnable",
        stored_rotations=trainer_modifier.stored_rotations,
        stored_rotations_dtype=trainer_modifier.stored_rotations_dtype,
    )

    new_model = _build_model()
    reload_state = State()
    reload_state.update(model=new_model, calib_data=dataloader)

    assert reload_modifier.on_initialize(reload_state)
    reload_modifier._center_embeddings(new_model)
    reload_modifier._fuse_norms(new_model)
    reload_modifier._apply_transforms(new_model)
    reload_modifier._finalize_transforms(new_model)

    reloaded_weight = (
        new_model.model.layers[0].self_attn.q_proj.weight.detach().clone()
    )
    assert torch.allclose(reference_weight, reloaded_weight, atol=1e-5, rtol=1e-4)


def test_spinquant_r4_block_size_autopick():
    config = LlamaConfig(
        hidden_size=96,
        intermediate_size=192,
        num_attention_heads=4,
        num_hidden_layers=1,
        vocab_size=128,
        max_position_embeddings=32,
    )
    model = LlamaForCausalLM(config)
    dataset = _DummyDataset(model.config.vocab_size)
    dataloader = DataLoader(dataset, batch_size=2)

    modifier = SpinQuantModifier(rotations=["R4"], transform_type="hadamard")
    state = State()
    state.update(model=model, calib_data=dataloader)

    assert modifier.on_initialize(state)
    scheme = modifier.transform_config.config_groups["R4"]
    assert scheme.head_dim == 32

import copy

import pytest
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import LlamaConfig, LlamaForCausalLM

from llmcompressor.args.dataset_arguments import DatasetArguments
from llmcompressor.core import active_session
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.modifiers.smoothquant import SmoothQuantModifier
from llmcompressor.modifiers.transform.spinquant import SpinQuantModifier
from llmcompressor.pipelines.registry import CalibrationPipeline
from llmcompressor.recipe import Recipe


class _ToyDataset(Dataset):
    def __init__(self, vocab_size: int, length: int = 8, seq_len: int = 64):
        self.vocab_size = vocab_size
        self.length = length
        self.seq_len = seq_len

    def __len__(self):
        return self.length

    def __getitem__(self, _: int):
        input_ids = torch.randint(0, self.vocab_size, (self.seq_len,), dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        return {"input_ids": input_ids, "attention_mask": attention_mask}


def _build_model() -> LlamaForCausalLM:
    config = LlamaConfig(
        hidden_size=256,
        intermediate_size=512,
        num_attention_heads=4,
        num_key_value_heads=4,
        num_hidden_layers=2,
        vocab_size=512,
        max_position_embeddings=256,
    )
    model = LlamaForCausalLM(config)
    model.eval()
    return model


def _collect_logits(model: LlamaForCausalLM, dataset: Dataset) -> torch.Tensor:
    dataloader = DataLoader(dataset, batch_size=2)
    logits = []
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            logits.append(outputs.logits.cpu())
    return torch.cat(logits, dim=0)


def _kld(base_logits: torch.Tensor, quant_logits: torch.Tensor) -> float:
    base_log_probs = torch.log_softmax(base_logits, dim=-1)
    quant_log_probs = torch.log_softmax(quant_logits, dim=-1)
    kl = torch.nn.functional.kl_div(
        quant_log_probs,
        base_log_probs,
        reduction="batchmean",
        log_target=True,
    )
    return float(kl.item())


def _apply_recipe(model: LlamaForCausalLM, modifiers, dataset: Dataset) -> None:
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False)
    session = active_session()
    session.reset()

    recipe = Recipe.from_modifiers(modifiers)
    session.initialize(
        recipe=recipe,
        model=model,
        calib_data=dataloader,
        copy_data=False,
    )

    dataset_args = DatasetArguments(
        pipeline="sequential",
        num_calibration_samples=len(dataset),
        max_seq_length=dataset.seq_len,
    )

    pipeline = CalibrationPipeline.from_modifiers(
        session.lifecycle.recipe.modifiers,
        user=dataset_args.pipeline,
    )

    pipeline(model, dataloader, dataset_args)
    session.reset()


@pytest.mark.xfail(reason="SpinQuant Cayley currently diverges due to loss definition.", strict=True)
def test_spinquant_cayley_small_model_regression():
    torch.manual_seed(42)

    base_model = _build_model()
    dataset = _ToyDataset(base_model.config.vocab_size, length=8, seq_len=64)

    base_logits = _collect_logits(base_model, dataset)

    # SmoothQuant baseline
    smooth_model = copy.deepcopy(base_model)
    smooth_modifiers = [
        SmoothQuantModifier(smoothing_strength=0.8),
        QuantizationModifier(scheme={"W8A8": ["Linear"]}, ignore=["lm_head"]),
    ]
    _apply_recipe(smooth_model, smooth_modifiers, dataset)
    smooth_logits = _collect_logits(smooth_model, dataset)

    # Static SpinQuant baseline
    static_model = copy.deepcopy(base_model)
    static_modifiers = [
        SpinQuantModifier(
            rotations=["R1", "R2"],
            transform_type="hadamard",
            learn_rotations=False,
        ),
        QuantizationModifier(scheme={"W8A8": ["Linear"]}, ignore=["lm_head"]),
    ]
    _apply_recipe(static_model, static_modifiers, dataset)
    static_logits = _collect_logits(static_model, dataset)

    # SpinQuant with Cayley-learned rotations
    cayley_model = copy.deepcopy(base_model)
    cayley_modifiers = [
        SpinQuantModifier(
            rotations=["R1", "R2"],
            transform_type="spinquant-learnable",
            learn_rotations=True,
            cayley_iters=2,
            cayley_lr=1.5,
            cayley_samples=8,
            cayley_seed=1234,
            precision=torch.float32,
        ),
        QuantizationModifier(scheme={"W8A8": ["Linear"]}, ignore=["lm_head"]),
    ]
    _apply_recipe(cayley_model, cayley_modifiers, dataset)
    cayley_logits = _collect_logits(cayley_model, dataset)

    smooth_kl = _kld(base_logits, smooth_logits)
    static_kl = _kld(base_logits, static_logits)
    cayley_kl = _kld(base_logits, cayley_logits)

    print("SmoothQuant KL:", smooth_kl)
    print("SpinQuant static KL:", static_kl)
    print("SpinQuant Cayley KL:", cayley_kl)

    # Expected once fixed: Cayley KL should be comparable to static baseline.
    assert cayley_kl <= static_kl * 5

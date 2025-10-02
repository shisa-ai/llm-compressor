import copy

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import LlamaConfig, LlamaForCausalLM

from llmcompressor.core import Event, EventType, State
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.modifiers.smoothquant import SmoothQuantModifier
from llmcompressor.modifiers.transform.spinquant import SpinQuantModifier


class _ToyDataset(Dataset):
    def __init__(self, vocab_size: int, length: int = 8, seq_len: int = 16):
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
        hidden_size=64,
        intermediate_size=128,
        num_attention_heads=4,
        num_hidden_layers=2,
        vocab_size=256,
        max_position_embeddings=128,
    )
    model = LlamaForCausalLM(config)
    model.eval()
    return model


def _apply_recipe(model: LlamaForCausalLM, modifiers, dataset: Dataset) -> None:
    dataloader = DataLoader(dataset, batch_size=2)
    state = State()
    state.update(model=model, calib_data=dataloader)

    for modifier in modifiers:
        modifier.initialize(state)

    start_event = Event(type_=EventType.CALIBRATION_EPOCH_START)
    end_event = Event(type_=EventType.CALIBRATION_EPOCH_END)
    batch_start = Event(type_=EventType.BATCH_START)
    batch_end = Event(type_=EventType.BATCH_END)

    for modifier in modifiers:
        modifier.on_event(state, start_event)

    device = next(model.parameters()).device

    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        state.batch_data = batch
        for modifier in modifiers:
            modifier.on_event(state, batch_start)
        with torch.no_grad():
            model(**batch)
        for modifier in modifiers:
            modifier.on_event(state, batch_end)

    for modifier in modifiers:
        modifier.on_event(state, end_event)
        modifier.finalize(state)


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


def test_spinquant_pipeline_kld_improves_over_smoothquant():
    torch.manual_seed(42)

    base_model = _build_model()
    dataset = _ToyDataset(base_model.config.vocab_size)

    base_logits = _collect_logits(base_model, dataset)

    smooth_model = copy.deepcopy(base_model)
    smooth_modifiers = [
        SmoothQuantModifier(smoothing_strength=0.8),
        QuantizationModifier(scheme={"W8A8": ["Linear"]}, ignore=["lm_head"]),
    ]
    _apply_recipe(smooth_model, smooth_modifiers, dataset)
    smooth_logits = _collect_logits(smooth_model, dataset)

    spin_model = copy.deepcopy(base_model)
    spin_modifiers = [
        SpinQuantModifier(
            rotations=["R1", "R2"],
            learn_rotations=True,
            transform_type="spinquant-learnable",
            cayley_iters=2,
            cayley_lr=0.2,
            cayley_samples=8,
            transform_block_size=8,
        ),
        QuantizationModifier(scheme={"W8A8": ["Linear"]}, ignore=["lm_head"]),
    ]
    _apply_recipe(spin_model, spin_modifiers, dataset)
    spin_logits = _collect_logits(spin_model, dataset)

    smooth_kl = _kld(base_logits, smooth_logits)
    spin_kl = _kld(base_logits, spin_logits)

    assert smooth_kl > 0.0
    assert spin_kl > 0.0
    assert spin_kl <= smooth_kl + 1e-4

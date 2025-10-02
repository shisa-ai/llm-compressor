import logging
from enum import Enum
from typing import Dict, Iterable, List, Literal, Optional

import torch
import torch.nn.utils.parametrize as parametrize
from compressed_tensors import match_modules_set, match_named_modules
from compressed_tensors.transform import (
    TransformArgs,
    TransformConfig,
    TransformScheme,
    apply_transform_config,
)
from compressed_tensors.utils import TorchDtype
from pydantic import Field, ValidationInfo, field_validator, PrivateAttr
from transformers import PreTrainedModel

from llmcompressor.core import Event, EventType, State
from llmcompressor.modeling import center_embeddings, fuse_norm_linears
from llmcompressor.modifiers import Modifier

from .cayley import cayley_step_
from .hadamard_utils import block_hadamard_, pick_block_size
from .learnable import SpinQuantLearnableTransform
from .mappings import SpinQuantMapping, infer_mapping_from_model
from .norm_mappings import NormMapping, infer_norm_mapping_from_model


LOGGER = logging.getLogger(__name__)


class SpinquantRotation(str, Enum):
    R1 = "R1"
    R2 = "R2"
    R3 = "R3"
    R4 = "R4"


class SpinQuantModifier(Modifier, use_enum_values=True):
    """
    Implements the transforms according to "SpinQuant: LLM quantization
    with learned rotations" (https://arxiv.org/abs/2405.16406)

    Transforms (rotations) are extra layers added to a model which reduce the accuracy
    loss induced by quantization. This is achived through "rotating" weights and
    activations into a space with a smaller dynamic range of values, thus decreasing
    the range of scales required for quantization.

    The SpinQuant authors describe four different rotations which can be applied to a
    model. R1 and R2 are "offline" rotations, meaning that they can be fused into
    existing weights and therefore do not induce runtime cost. R3 and R4 are "online"
    rotations, meaning that they require additional computation at runtime.

    Lifecycle:
        - on_initialize
            - infer SpinQuantMappings & NormMappings
            - as needed, create transform schemes for R1, R2, R3, & R4
        - on_start
            - normalize embeddings
            - fuse norm layers into subsequent Linear layers
            - apply TransformConfig
                - fuse transforms into weights for mergeable transforms
                - add hooks for online transforms
        - on sequential epoch end
        - on_end
        - on_finalize

    :param rotations: A list containing the names of rotations to apply to the model.
        Possible rotations include R1, R2, R3, and R4
    :param transform_type: The type of transform to apply to the model.
        `"hadamard"` has the least performance cost but only supports sizes which are
        powers of power of two.
        `"random-matrix"` has more performance cost, but supports a much larger set of
            sizes.
        `"random-matrix"` has the greatest performance cost, but supports any size
    :param randomize: if True, create distinct transforms for each application
    :param precision: Precision at which all transforms should be applied. This applies
        to both weight fusing and online rotations
    :param transform_block_size: Block size to use for rotation matrices. The model's
        hidden_size and head_dim must be evenly divisible by transform_block_size.
        Layers will be transformed by a block-diagonal matrix where each block is a
        matrix of this size.
        If None is provided, model's hidden_size will be used for R1, R3, and R4
        and model's head_dim will be used for R2
    :param learn_rotations: enable Cayley-SGD stage for R1/R2 before fusing weights
    :param cayley_iters: number of Cayley steps to run during optimization
    :param cayley_lr: learning rate for Cayley-SGD
    :param cayley_samples: number of calibration samples to use for learning
    :param cayley_seed: random seed for rotation initialization
    :param r1_mode: rotation layout for R1 ("global" or "block")
    :param r1_block_size: optional override for block size when r1_mode="block"
    :param enable_r3: attach online Hadamard to attention Q/K/KV cache paths
    :param enable_r4: attach online Hadamard to MLP down projection
    :param mappings: Specifies layers within a model to target for transforms.
        A mapping will be inferred if None is provided
    :param norm_mappings: Specifies layers within a model to target for norm fusing.
        A mapping will be inferred if None is provided
    :param transform_config: Optional transform config for overriding provided arguments
    """

    rotations: List[SpinquantRotation] = Field(default_factory=lambda: ["R1", "R2"])
    transform_type: Literal[
        "hadamard",
        "random-hadamard",
        "random-matrix",
        "spinquant-learnable",
    ] = Field(default="hadamard")
    randomize: bool = Field(default=False)
    precision: TorchDtype = Field(default=torch.float64)
    transform_block_size: Optional[int] = Field(default=None)
    learn_rotations: bool = Field(default=False)
    cayley_iters: int = Field(default=100)
    cayley_lr: float = Field(default=1.5)
    cayley_samples: int = Field(default=128)
    cayley_seed: int = Field(default=1234)
    r1_mode: Literal["global", "block"] = Field(default="block")
    r1_block_size: Optional[int] = Field(default=None)
    enable_r3: bool = Field(default=False)
    enable_r4: bool = Field(default=False)

    # norm mappings separate from spinquant mappings to allow users to
    # override spinquant mappings with transform_config without overriding norms
    mappings: Optional[SpinQuantMapping] = Field(
        default=None,
        repr=False,
        exclude=True,
    )
    norm_mappings: Optional[List[NormMapping]] = Field(
        default=None,
        repr=False,
        exclude=True,
    )

    # optional override for more fine-grained control
    # also included in recipe serialization
    transform_config: Optional[TransformConfig] = Field(default=None, repr=False)

    _rotation_transforms: List[SpinQuantLearnableTransform] = PrivateAttr(
        default_factory=list
    )
    _rotation_params: List[torch.nn.Parameter] = PrivateAttr(default_factory=list)
    _original_requires_grad: List[tuple[torch.nn.Parameter, bool]] = PrivateAttr(
        default_factory=list
    )
    _calib_loader = PrivateAttr(default=None)
    _parent_modules: List[torch.nn.Module] = PrivateAttr(default_factory=list)
    _hidden_size: Optional[int] = PrivateAttr(default=None)
    _head_dim: Optional[int] = PrivateAttr(default=None)
    _r3_hooked: bool = PrivateAttr(default=False)
    _r3_block_size: Optional[int] = PrivateAttr(default=None)

    @field_validator("randomize", mode="before")
    def validate_not_implemented(cls, value, info: ValidationInfo):
        if value:
            raise NotImplementedError(f"{info.field_name} is not supported as of now")
        return value

    @field_validator("rotations", mode="before")
    def validate_rotations(cls, value):
        if isinstance(value, Iterable):
            return tuple(v.upper() for v in value)
        return value

    def on_initialize(self, state: State, **kwargs) -> bool:
        if self.transform_config is not None:
            return True

        self.mappings = infer_mapping_from_model(state.model)
        self.norm_mappings = infer_norm_mapping_from_model(state.model)
        self._calib_loader = state.data.calib
        model_config = getattr(state.model, "config", None)
        self._hidden_size = getattr(model_config, "hidden_size", None)

        if SpinquantRotation.R3 in self.rotations:
            self.enable_r3 = True
        if SpinquantRotation.R4 in self.rotations:
            self.enable_r4 = True

        if self.learn_rotations and self.transform_type != "spinquant-learnable":
            LOGGER.info(
                "Switching SpinQuant transform type to spinquant-learnable for training"
            )
            self.transform_type = "spinquant-learnable"

        head_dim = self._infer_head_dim(state.model)
        self._head_dim = head_dim

        if self._hidden_size is None:
            heads = getattr(model_config, "num_attention_heads", None)
            if heads is not None:
                self._hidden_size = head_dim * heads

        self._r3_hooked = False

        if SpinquantRotation.R3 in self.rotations or self.enable_r3:
            total_dim = self._hidden_size or head_dim
            prefer = self.transform_block_size or head_dim
            self._r3_block_size = pick_block_size(total_dim, prefer)
        else:
            self._r3_block_size = None

        config_groups = {}
        if SpinquantRotation.R1 in self.rotations:
            config_groups["R1"] = self._create_r1_scheme()

        if SpinquantRotation.R2 in self.rotations:
            config_groups["R2"] = self._create_r2_scheme(head_dim)

        if SpinquantRotation.R4 in self.rotations:
            config_groups["R4"] = self._create_r4_scheme()

        self.transform_config = TransformConfig(config_groups=config_groups)

        return True

    def on_start(self, state: State, event: Event, **kwargs):
        self.started_ = True

        self._center_embeddings(state.model)
        self._fuse_norms(state.model)

        self._apply_transforms(state.model)

        if self.learn_rotations:
            self._collect_rotation_structures(state.model)
            self._run_cayley(state)
            self._finalize_transforms(state.model)
        else:
            self._rotation_transforms.clear()
            self._rotation_params.clear()

    def on_event(self, state: State, event: Event, **kwargs):
        if event.type_ == EventType.CALIBRATION_EPOCH_START:
            if not self.started_:
                self.on_start(state, None)

        elif event.type_ == EventType.SEQUENTIAL_EPOCH_END:
            pass

        elif event.type_ == EventType.CALIBRATION_EPOCH_END:
            if not self.ended_:
                self.on_end(state, None)

    def on_end(self, state: State, event: Event, **kwargs):
        self.ended_ = True

    def on_finalize(self, state: State, **kwargs) -> bool:
        if not self.ended_:
            self.on_end(state, None)

        return True

    def _center_embeddings(self, model: PreTrainedModel):
        for _, embedding in match_named_modules(
            model, [self.mappings.embedding], warn_on_fail=True
        ):
            center_embeddings(embedding)

    def _fuse_norms(self, model: PreTrainedModel):
        for mapping in self.norm_mappings:
            for norm, *linears in match_modules_set(
                model, (mapping.norm, *mapping.linears)
            ):
                fuse_norm_linears(norm, linears)

    def _create_r1_scheme(self) -> TransformScheme:
        head_dim = None
        if self.r1_mode == "block":
            if self._hidden_size is None:
                raise ValueError(
                    "SpinQuant R1 block mode requires the model hidden size to be known"
                )
            prefer = self.r1_block_size or self.transform_block_size
            head_dim = pick_block_size(self._hidden_size, prefer)

        return TransformScheme(
            type=self.transform_type,
            randomize=self.randomize,
            requires_grad=self.learn_rotations,
            precision=self.precision,
            head_dim=head_dim,
            apply=[
                TransformArgs(
                    targets=[
                        self.mappings.embedding,
                        self.mappings.attn_o,
                        *self.mappings.mlp_out,
                    ],
                    location="weight_output",
                ),
                TransformArgs(
                    targets=[
                        self.mappings.attn_q,
                        self.mappings.attn_k,
                        self.mappings.attn_v,
                        *self.mappings.mlp_in,
                        self.mappings.lm_head,
                    ],
                    location="weight_input",
                    inverse=True,
                ),
            ],
        )

    def _create_r2_scheme(self, head_dim: int) -> TransformScheme:
        block = pick_block_size(head_dim, self.transform_block_size)

        return TransformScheme(
            type=self.transform_type,
            randomize=self.randomize,
            requires_grad=self.learn_rotations,
            precision=self.precision,
            head_dim=block,
            apply=[
                TransformArgs(targets=[self.mappings.attn_v], location="weight_output"),
                TransformArgs(
                    targets=[self.mappings.attn_o],
                    location="weight_input",
                    inverse=True,
                ),
            ],
        )

    def _create_r3_scheme(self, head_dim: int) -> TransformScheme:
        raise NotImplementedError(
            "SpinQuant R3 rotations will be added in a future release"
        )

    def _create_r4_scheme(self) -> TransformScheme:
        return TransformScheme(
            type=self.transform_type,
            randomize=self.randomize,
            requires_grad=self.learn_rotations,
            precision=self.precision,
            head_dim=self.transform_block_size,
            apply=[
                TransformArgs(
                    targets=[*self.mappings.mlp_out],
                    location="input",
                ),
                TransformArgs(
                    targets=[*self.mappings.mlp_out],
                    location="weight_input",
                    inverse=True,
                ),
            ],
        )

    def _infer_head_dim(self, model: PreTrainedModel) -> int:
        config = model.config

        if hasattr(config, "head_dim"):
            return config.head_dim
        if hasattr(config, "hidden_size") and hasattr(config, "num_attention_heads"):
            return config.hidden_size // config.num_attention_heads
        raise NotImplementedError()

    def _apply_transforms(self, model: PreTrainedModel) -> None:
        if self.transform_config is None:
            return
        self._prepare_rotation_keys(model)
        apply_transform_config(model, self.transform_config)
        self._attach_online_rotations(model)

    def _prepare_rotation_keys(self, model: PreTrainedModel) -> None:
        if self.transform_type != "spinquant-learnable":
            return

        if SpinquantRotation.R1 in self.rotations and self.r1_mode == "block":
            r1_targets = [
                self.mappings.embedding,
                self.mappings.attn_o,
                *self.mappings.mlp_out,
                self.mappings.attn_q,
                self.mappings.attn_k,
                self.mappings.attn_v,
                *self.mappings.mlp_in,
                self.mappings.lm_head,
            ]
            for target in r1_targets:
                for name, module in match_named_modules(
                    model, [target], warn_on_fail=True
                ):
                    setattr(module, "_spinquant_rotation_key", f"R1_{name}")

        if SpinquantRotation.R2 in self.rotations:
            for attn_name, attn_module in match_named_modules(
                model, [self.mappings.attn], warn_on_fail=True
            ):
                key = f"R2_{attn_name}"
                for target in (self.mappings.attn_v, self.mappings.attn_o):
                    for _, module in match_named_modules(
                        attn_module, [target], warn_on_fail=True
                    ):
                        setattr(module, "_spinquant_rotation_key", key)

    def _attach_online_rotations(self, model: PreTrainedModel) -> None:
        if self._r3_hooked:
            return
        if not (SpinquantRotation.R3 in self.rotations or self.enable_r3):
            return
        if self._r3_block_size is None or self._r3_block_size <= 1:
            LOGGER.debug("Skipping SpinQuant R3 hooks due to block size <= 1")
            self._r3_hooked = True
            return

        block_size = self._r3_block_size
        for _, attn in match_named_modules(model, [self.mappings.attn], warn_on_fail=True):
            for attr in ("q_proj", "k_proj"):
                module = getattr(attn, attr, None)
                if module is None or getattr(module, "_spinquant_r3_hooked", False):
                    continue

                def _make_hook(bs):
                    def _hook(_mod, _inp, output):
                        return block_hadamard_(output, bs)

                    return _hook

                module.register_forward_hook(_make_hook(block_size))
                setattr(module, "_spinquant_r3_hooked", True)

            if self.enable_r3:
                module = getattr(attn, "v_proj", None)
                if module is not None and not getattr(module, "_spinquant_r3_hooked", False):
                    module.register_forward_hook(
                        lambda _mod, _inp, output, bs=block_size: block_hadamard_(
                            output, bs
                        )
                    )
                    setattr(module, "_spinquant_r3_hooked", True)

        self._r3_hooked = True

    def _collect_rotation_structures(self, model: PreTrainedModel) -> None:
        self._rotation_transforms.clear()
        self._rotation_params.clear()
        self._parent_modules.clear()

        parent_lookup = {}
        for parent in model.modules():
            for child in parent.children():
                if isinstance(child, SpinQuantLearnableTransform):
                    parent_lookup[child] = parent

        seen = set()
        for transform in parent_lookup:
            if transform.rotation_name not in ("R1", "R2"):
                continue
            self._rotation_transforms.append(transform)
            parent = parent_lookup[transform]
            if parent not in self._parent_modules:
                self._parent_modules.append(parent)
            rotation = transform.weight
            if id(rotation) not in seen:
                self._rotation_params.append(rotation)
                seen.add(id(rotation))

    def _run_cayley(self, state: State) -> None:
        if not self._rotation_params:
            LOGGER.warning("No learnable rotations found; skipping Cayley optimization")
            return

        if self._calib_loader is None:
            raise ValueError("SpinQuant learn_rotations requires a calibration dataloader")

        model = state.model
        model_device = next(model.parameters()).device
        model.eval()

        self._original_requires_grad = [
            (param, param.requires_grad) for param in model.parameters()
        ]
        for param, _ in self._original_requires_grad:
            param.requires_grad_(False)
        for rotation in self._rotation_params:
            rotation.requires_grad_(True)

        iterator = iter(self._calib_loader)
        max_iters = max(self.cayley_iters, 1)
        samples_limit = self.cayley_samples if self.cayley_samples > 0 else None
        batches_run = 0
        samples_seen = 0

        if self.cayley_seed is not None:
            torch.manual_seed(self.cayley_seed)

        while batches_run < max_iters:
            if samples_limit is not None and samples_seen >= samples_limit:
                break

            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(self._calib_loader)
                batch = next(iterator)

            batch = {
                key: value.to(model_device)
                for key, value in batch.items()
                if isinstance(value, torch.Tensor)
            }

            batch_size = self._infer_batch_size(batch)
            samples_seen += batch_size

            model.zero_grad(set_to_none=True)
            with torch.enable_grad():
                outputs = model(**batch)

                if not hasattr(outputs, "logits"):
                    raise ValueError(
                        "SpinQuant Cayley optimization expects model outputs with logits"
                    )

                logits = outputs.logits
                quantized = self._ste_uniform_quant(logits)
                loss = torch.nn.functional.mse_loss(quantized, logits.detach())

            loss.backward()
            self._apply_cayley_updates()

            batches_run += 1

        for rotation in self._rotation_params:
            rotation.requires_grad_(False)
        for param, flag in self._original_requires_grad:
            param.requires_grad_(flag)
        self._original_requires_grad.clear()

    def _finalize_transforms(self, model: PreTrainedModel) -> None:
        if not self._rotation_transforms:
            return
        for module in self._parent_modules:
            if parametrize.is_parametrized(module, "weight"):
                parametrize.remove_parametrizations(
                    module, "weight", leave_parametrized=True
                )

        self._rotation_transforms.clear()
        self._rotation_params.clear()
        self._parent_modules.clear()

    def _apply_cayley_updates(self) -> None:
        lr = self.cayley_lr
        if lr <= 0:
            return

        with torch.no_grad():
            for rotation in self._rotation_params:
                grad = rotation.grad
                if grad is None:
                    continue
                cayley_step_(rotation, grad, lr)
                rotation.grad.zero_()

    @staticmethod
    def _infer_batch_size(batch: dict[str, torch.Tensor]) -> int:
        for value in batch.values():
            if isinstance(value, torch.Tensor) and value.ndim > 0:
                return value.shape[0]
        return 1

    @staticmethod
    def _ste_uniform_quant(tensor: torch.Tensor, num_bits: int = 4) -> torch.Tensor:
        if num_bits <= 0:
            return tensor

        qmax = 2 ** (num_bits - 1) - 1
        scale = tensor.detach().abs().amax(dim=-1, keepdim=True).clamp(min=1e-6) / qmax
        scaled = tensor / scale
        quantized = torch.clamp(scaled.round(), -qmax - 1, qmax)
        dequant = quantized * scale
        return tensor + (dequant - tensor).detach()

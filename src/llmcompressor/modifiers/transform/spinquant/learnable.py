"""Custom learnable transform factory for SpinQuant rotations."""

from __future__ import annotations

from typing import Dict, Hashable, Optional

import torch
from compressed_tensors.transform import apply_transform_weight
from compressed_tensors.transform.factory.base import TransformBase, TransformFactory
from compressed_tensors.transform.factory.hadamard import get_transform_size
from compressed_tensors.utils import get_offloaded_device
from torch import Tensor
from torch.nn import Module, Parameter, Linear

from llmcompressor.utils.accelerate import get_execution_device
from .quantization import fake_quantize_activation


class SpinQuantLearnableTransform(TransformBase):
    """Trainable rotation transform backed by a shared Parameter."""

    def __init__(
        self,
        name: str,
        key: Hashable,
        weight: Parameter,
        inverse: bool,
        scheme,
        args,
        module_type: type[Module],
    ) -> None:
        super().__init__()
        self.rotation_name = name
        self.rotation_key = key
        self.weight = weight
        self.inverse = inverse
        self.scheme = scheme
        self.args = args
        self.module_type = module_type
        # Activation quantization control (set during Cayley training)
        self.cayley_quant_enabled = False
        self.cayley_quant_bits = 4

    def forward(self, value: Tensor) -> Tensor:
        rotation = self.weight
        if self.inverse:
            rotation = rotation.transpose(-1, -2)

        rotated = apply_transform_weight(
            rotation.to(device=value.device),
            value.to(dtype=rotation.dtype),
            self.args.location,
            self.module_type,
        )

        # Apply activation quantization during Cayley training
        # This matches SpinQuant reference: ActQuantWrapper.forward() line 283-286
        if self.cayley_quant_enabled and self.training:
            rotated = fake_quantize_activation(
                rotated, num_bits=self.cayley_quant_bits, enabled=True
            )

        return rotated.to(value.dtype)


@TransformFactory.register("spinquant-learnable")
class SpinQuantLearnableFactory(TransformFactory):
    """Factory that produces learnable orthonormal transforms for SpinQuant."""

    def __init__(self, name: str, scheme, seed: Optional[int] = None) -> None:
        super().__init__(name=name, scheme=scheme, seed=seed)
        self._rotations: Dict[Hashable, Parameter] = {}

    def create_transform(self, module: Module, args) -> SpinQuantLearnableTransform:
        key = self._make_rotation_key(module)
        rotation = self._ensure_rotation(module, args, key)
        module_type = type(module)
        if module_type.__name__ == "ParametrizedLinear":
            module_type = Linear
        return SpinQuantLearnableTransform(
            name=self.name,
            key=key,
            weight=rotation,
            inverse=args.inverse,
            scheme=self.scheme,
            args=args,
            module_type=module_type,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _make_rotation_key(self, module: Module) -> Hashable:
        key = getattr(module, "_spinquant_rotation_key", None)
        if key is not None:
            return key

        if self.name == "R1":
            return "R1_global"

        return id(module)

    def _ensure_rotation(self, module: Module, args, key: Hashable) -> Parameter:
        if key in self._rotations:
            return self._rotations[key]

        size = get_transform_size(module, args.location, self.scheme.head_dim)
        device = get_offloaded_device(module)
        exec_device = get_execution_device(module)
        precision = self.scheme.precision
        if (
            not args.is_online()
            and precision in {torch.float16, torch.bfloat16}
            and not torch.cuda.is_available()
        ):
            precision = torch.float32

        preload = getattr(module, "_spinquant_rotation_preload", None)
        if isinstance(preload, dict) and key in preload:
            data = preload.pop(key)
            if not preload:
                delattr(module, "_spinquant_rotation_preload")
            data = data.to(device=device, dtype=precision)
            rotation = Parameter(data.clone(), requires_grad=self.scheme.requires_grad)
            self._rotations[key] = rotation
            return rotation

        eye = torch.eye(size, dtype=precision, device=exec_device)
        eye = eye.to(device=device)

        rotation = Parameter(eye, requires_grad=self.scheme.requires_grad)
        self._rotations[key] = rotation
        return rotation

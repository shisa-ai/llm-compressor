"""Utilities for interacting with Accelerate device hooks."""

from __future__ import annotations

from typing import Optional

import torch
from torch.nn import Module

try:  # pragma: no cover - accelerate may not be installed in minimal environments
    from accelerate.hooks import AlignDevicesHook, SequentialHook
except ImportError:  # pragma: no cover
    AlignDevicesHook = None  # type: ignore[assignment]
    SequentialHook = None  # type: ignore[assignment]

from compressed_tensors.utils import get_execution_device as _ct_get_execution_device


def _extract_align_hook(hook: Optional[object]) -> Optional[AlignDevicesHook]:
    """Return the first ``AlignDevicesHook`` contained within ``hook``."""
    if AlignDevicesHook is None or hook is None:
        return None

    if isinstance(hook, AlignDevicesHook):
        return hook

    # ``SequentialHook`` is used by accelerate to chain multiple hooks together.
    if SequentialHook is not None and isinstance(hook, SequentialHook):  # pragma: no branch
        for sub_hook in hook.hooks:
            found = _extract_align_hook(sub_hook)
            if found is not None:
                return found

    return None


def get_execution_device(module: Module) -> torch.device:
    """Resolve the execution device for a module that may have nested accelerate hooks.

    ``compressed_tensors.utils.get_execution_device`` only recognises hooks when the
    stored ``_hf_hook`` is an instance of ``AlignDevicesHook``. Newer versions of
    Accelerate wrap hooks inside ``SequentialHook`` containers. This helper unwraps
    such containers before delegating to the original utility, ensuring the resolved
    device does not default to ``meta`` tensors.
    """

    for submodule in module.modules():
        align_hook = _extract_align_hook(getattr(submodule, "_hf_hook", None))
        if align_hook is not None and align_hook.execution_device is not None:
            return torch.device(align_hook.execution_device)

        param = next(submodule.parameters(recurse=False), None)
        if param is not None and param.device.type != "meta":
            return param.device

    # Fall back to the compressed-tensors implementation as a last resort.
    device = _ct_get_execution_device(module)
    if device.type == "meta":
        return torch.device("cpu")
    return device


__all__ = ["get_execution_device"]

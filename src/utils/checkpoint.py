from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn


def safe_torch_load(path: str | Path, map_location: torch.device | str | None = None) -> Any:
    """Load checkpoint with safer defaults across torch versions."""
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        # Fallback for older torch versions without weights_only.
        return torch.load(path, map_location=map_location)


def _strip_compile_prefix(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    prefix = "_orig_mod."
    if state_dict and all(k.startswith(prefix) for k in state_dict):
        return {k[len(prefix):]: v for k, v in state_dict.items()}
    return state_dict


def extract_model_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint
    else:
        raise TypeError("Unsupported checkpoint format: expected a dict-like object")

    if not isinstance(state_dict, dict):
        raise TypeError("model_state_dict must be a dictionary")

    return _strip_compile_prefix(state_dict)


def get_model_to_save(model: nn.Module) -> nn.Module:
    # torch.compile wraps the original model under _orig_mod.
    return getattr(model, "_orig_mod", model)

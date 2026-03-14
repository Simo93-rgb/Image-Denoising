from __future__ import annotations

import torch

from src.models.autoencoder import DenoisingAutoencoder
from src.utils.checkpoint import extract_model_state_dict


def test_extract_model_state_dict_strips_compile_prefix() -> None:
    model = DenoisingAutoencoder()
    base_state = model.state_dict()
    prefixed_state = {f"_orig_mod.{k}": v.clone() for k, v in base_state.items()}
    checkpoint = {"model_state_dict": prefixed_state}

    extracted = extract_model_state_dict(checkpoint)
    assert "enc1.0.weight" in extracted
    assert "_orig_mod.enc1.0.weight" not in extracted


def test_extract_model_state_dict_accepts_plain_state_dict() -> None:
    model = DenoisingAutoencoder()
    base_state = model.state_dict()
    extracted = extract_model_state_dict(base_state)
    assert extracted.keys() == base_state.keys()

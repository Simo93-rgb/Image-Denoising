from __future__ import annotations

import torch

from src.models.autoencoder import DenoisingAutoencoder


def test_autoencoder_output_shape_matches_input() -> None:
    model = DenoisingAutoencoder()
    x = torch.rand(8, 1, 28, 28)
    y = model(x)
    assert y.shape == x.shape


def test_autoencoder_output_is_in_unit_range() -> None:
    model = DenoisingAutoencoder()
    x = torch.rand(4, 1, 28, 28)
    y = model(x)
    assert torch.min(y) >= 0.0
    assert torch.max(y) <= 1.0

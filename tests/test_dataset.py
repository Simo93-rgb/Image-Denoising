from __future__ import annotations

import torch

from src.data.dataset import DenoisingPairDataset, load_fashion_mnist_csv


def test_csv_loading_returns_expected_shapes() -> None:
    tensors = load_fashion_mnist_csv("data/raw/fashion-mnist_train.csv")
    assert tensors.images.ndim == 4
    assert tensors.images.shape[1:] == (1, 28, 28)
    assert tensors.labels.ndim == 1
    assert tensors.images.shape[0] == tensors.labels.shape[0]


def test_denoising_dataset_pair_shapes_and_range() -> None:
    tensors = load_fashion_mnist_csv("data/raw/fashion-mnist_train.csv")
    idx = torch.arange(0, 32, dtype=torch.long)
    ds = DenoisingPairDataset(
        tensors.images,
        tensors.labels,
        indices=idx,
        noise_std=0.3,
        fixed_noise=True,
        seed=42,
    )
    noisy, clean, label = ds[0]
    assert noisy.shape == clean.shape == (1, 28, 28)
    assert isinstance(label.item(), int)
    assert torch.min(noisy) >= 0.0
    assert torch.max(noisy) <= 1.0

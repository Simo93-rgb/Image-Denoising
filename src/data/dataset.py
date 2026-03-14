from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset

from src.data.transforms import AddGaussianNoise


@dataclass(slots=True)
class DatasetTensors:
    images: torch.Tensor
    labels: torch.Tensor


def load_fashion_mnist_csv(csv_path: str | Path) -> DatasetTensors:
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV dataset not found: {csv_path}")

    frame = pd.read_csv(csv_path)
    data = torch.tensor(frame.values, dtype=torch.float32)

    labels = data[:, 0].long()
    images = (data[:, 1:] / 255.0).reshape(-1, 1, 28, 28)
    return DatasetTensors(images=images, labels=labels)


class DenoisingPairDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    """Return (noisy, clean, label) pairs for denoising autoencoder training."""

    def __init__(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        indices: torch.Tensor,
        noise_std: float = 0.3,
        fixed_noise: bool = False,
        seed: int = 42,
    ) -> None:
        self.images = images[indices]
        self.labels = labels[indices]
        self.noise = AddGaussianNoise(std=noise_std)
        self.fixed_noise = fixed_noise
        self.generator = torch.Generator().manual_seed(seed)

        self.noisy_images: torch.Tensor | None = None
        if self.fixed_noise:
            self.noisy_images = self.noise(self.images, generator=self.generator)

    def __len__(self) -> int:
        return self.images.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        clean = self.images[idx]
        label = self.labels[idx]
        if self.noisy_images is not None:
            noisy = self.noisy_images[idx]
        else:
            noisy = self.noise(clean)
        return noisy, clean, label

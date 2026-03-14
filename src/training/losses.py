from __future__ import annotations

import torch
from torch import nn
from torchmetrics.functional.image import structural_similarity_index_measure


class CombinedDenoisingLoss(nn.Module):
    def __init__(self, alpha: float = 0.85) -> None:
        super().__init__()
        if not (0.0 <= alpha <= 1.0):
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        self.alpha = alpha
        self.mse = nn.MSELoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mse_term = self.mse(pred, target)
        ssim_term = 1.0 - structural_similarity_index_measure(pred, target, data_range=1.0)
        return self.alpha * mse_term + (1.0 - self.alpha) * ssim_term


def reconstruction_loss(alpha: float = 0.85) -> nn.Module:
    return CombinedDenoisingLoss(alpha=alpha)

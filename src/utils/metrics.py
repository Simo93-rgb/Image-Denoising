from __future__ import annotations

import torch
from torchmetrics.functional.image import peak_signal_noise_ratio, structural_similarity_index_measure


def batch_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean((pred - target) ** 2)


def batch_psnr(pred: torch.Tensor, target: torch.Tensor, data_range: float = 1.0) -> torch.Tensor:
    return peak_signal_noise_ratio(pred, target, data_range=data_range)


def batch_ssim(pred: torch.Tensor, target: torch.Tensor, data_range: float = 1.0) -> torch.Tensor:
    return structural_similarity_index_measure(pred, target, data_range=data_range)


@torch.no_grad()
def evaluate_metrics(pred: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    mse = float(batch_mse(pred, target).detach().cpu().item())
    psnr = float(batch_psnr(pred, target).detach().cpu().item())
    ssim = float(batch_ssim(pred, target).detach().cpu().item())
    return {"mse": mse, "psnr": psnr, "ssim": ssim}

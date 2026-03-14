from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import torch


def plot_loss_curves(train_losses: list[float], val_losses: list[float], out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(9, 5))
    plt.plot(epochs, train_losses, label="Train Loss", linewidth=2)
    plt.plot(epochs, val_losses, label="Val Loss", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Training and Validation Loss")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


@torch.no_grad()
def plot_denoising_samples(
    clean: torch.Tensor,
    noisy: torch.Tensor,
    denoised: torch.Tensor,
    out_path: str | Path,
    max_samples: int = 12,
) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n = min(max_samples, clean.shape[0])
    clean = clean[:n].detach().cpu()
    noisy = noisy[:n].detach().cpu()
    denoised = denoised[:n].detach().cpu()

    fig, axes = plt.subplots(nrows=n, ncols=3, figsize=(9, 2.2 * n))
    if n == 1:
        axes = axes.reshape(1, -1)

    for i in range(n):
        axes[i, 0].imshow(clean[i, 0], cmap="gray")
        axes[i, 0].set_title("Clean")
        axes[i, 1].imshow(noisy[i, 0], cmap="gray")
        axes[i, 1].set_title("Noisy")
        axes[i, 2].imshow(denoised[i, 0], cmap="gray")
        axes[i, 2].set_title("Denoised")

        for col in range(3):
            axes[i, col].axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)

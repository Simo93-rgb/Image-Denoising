from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


@dataclass(slots=True)
class TrainHistory:
    train_loss: list[float]
    val_loss: list[float]
    best_epoch: int
    best_val_loss: float
    best_checkpoint: Path


class DenoisingTrainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: torch.device,
        checkpoint_dir: str | Path,
        use_amp: bool = True,
        early_stopping_patience: int = 10,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.use_amp = use_amp and device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        self.early_stopping_patience = early_stopping_patience

    def fit(self, train_loader: DataLoader, val_loader: DataLoader, epochs: int) -> TrainHistory:
        train_losses: list[float] = []
        val_losses: list[float] = []
        best_val = float("inf")
        best_epoch = 0
        best_checkpoint = self.checkpoint_dir / "best_model.pth"
        epochs_without_improvement = 0

        for epoch in range(1, epochs + 1):
            start = perf_counter()
            train_loss = self._run_epoch(train_loader, training=True)
            val_loss = self._run_epoch(val_loader, training=False)
            train_losses.append(train_loss)
            val_losses.append(val_loss)

            if val_loss < best_val:
                best_val = val_loss
                best_epoch = epoch
                epochs_without_improvement = 0
                self._save_checkpoint(best_checkpoint, epoch, val_loss)
            else:
                epochs_without_improvement += 1

            elapsed = perf_counter() - start
            print(
                f"Epoch {epoch:03d}/{epochs:03d} | "
                f"train_loss={train_loss:.6f} | val_loss={val_loss:.6f} | "
                f"time={elapsed:.2f}s"
            )

            if epochs_without_improvement >= self.early_stopping_patience:
                print(
                    "Early stopping triggered: "
                    f"no val improvement for {self.early_stopping_patience} epochs."
                )
                break

        return TrainHistory(
            train_loss=train_losses,
            val_loss=val_losses,
            best_epoch=best_epoch,
            best_val_loss=best_val,
            best_checkpoint=best_checkpoint,
        )

    def _run_epoch(self, loader: DataLoader, training: bool) -> float:
        self.model.train(training)
        total_loss = 0.0
        total_samples = 0

        iterator = tqdm(loader, leave=False, desc="Train" if training else "Val")
        for noisy, clean, _ in iterator:
            noisy = noisy.to(self.device, non_blocking=True)
            clean = clean.to(self.device, non_blocking=True)

            if training:
                self.optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type="cuda", enabled=self.use_amp):
                output = self.model(noisy)
                loss = self.criterion(output, clean)

            if training:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

            batch_size = noisy.shape[0]
            total_loss += float(loss.detach().item()) * batch_size
            total_samples += batch_size
            iterator.set_postfix(loss=f"{loss.detach().item():.4f}")

        return total_loss / max(1, total_samples)

    def _save_checkpoint(self, path: Path, epoch: int, val_loss: float) -> None:
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "val_loss": val_loss,
            },
            path,
        )

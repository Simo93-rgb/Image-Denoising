from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import torch
from torch import nn
from torchmetrics.functional.image import peak_signal_noise_ratio, structural_similarity_index_measure
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.utils.checkpoint import get_model_to_save


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
        tb_writer: Any | None = None,
        use_amp: bool = True,
        early_stopping_patience: int = 10,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.tb_writer = tb_writer

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
            train_stats = self._run_epoch(train_loader, training=True)
            val_stats = self._run_epoch(val_loader, training=False)

            train_loss = train_stats["loss"]
            val_loss = val_stats["loss"]
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
            if self.tb_writer is not None:
                self.tb_writer.add_scalar("loss/train", train_loss, epoch)
                self.tb_writer.add_scalar("loss/val", val_loss, epoch)
                self.tb_writer.add_scalar("metrics/train_mse", train_stats["mse"], epoch)
                self.tb_writer.add_scalar("metrics/val_mse", val_stats["mse"], epoch)
                self.tb_writer.add_scalar("metrics/train_psnr", train_stats["psnr"], epoch)
                self.tb_writer.add_scalar("metrics/val_psnr", val_stats["psnr"], epoch)
                self.tb_writer.add_scalar("metrics/train_ssim", train_stats["ssim"], epoch)
                self.tb_writer.add_scalar("metrics/val_ssim", val_stats["ssim"], epoch)
                self.tb_writer.add_scalar("train/epoch_time_sec", elapsed, epoch)
                self.tb_writer.add_scalar("train/lr", self.optimizer.param_groups[0]["lr"], epoch)

            print(
                f"Epoch {epoch:03d}/{epochs:03d} | "
                f"train_loss={train_loss:.6f} | val_loss={val_loss:.6f} | "
                f"val_psnr={val_stats['psnr']:.3f} | val_ssim={val_stats['ssim']:.4f} | "
                f"time={elapsed:.2f}s"
            )

            if epochs_without_improvement >= self.early_stopping_patience:
                print(
                    "Early stopping triggered: "
                    f"no val improvement for {self.early_stopping_patience} epochs."
                )
                break

        if self.tb_writer is not None:
            self.tb_writer.add_scalar("train/best_epoch", best_epoch, 0)
            self.tb_writer.add_scalar("train/best_val_loss", best_val, 0)

        return TrainHistory(
            train_loss=train_losses,
            val_loss=val_losses,
            best_epoch=best_epoch,
            best_val_loss=best_val,
            best_checkpoint=best_checkpoint,
        )

    def _run_epoch(self, loader: DataLoader, training: bool) -> dict[str, float]:
        self.model.train(training)
        total_loss = 0.0
        total_mse = 0.0
        total_psnr = 0.0
        total_ssim = 0.0
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
            with torch.no_grad():
                pred = output.detach()
                target = clean.detach()
                mse = torch.mean((pred - target) ** 2)
                psnr = peak_signal_noise_ratio(pred, target, data_range=1.0)
                ssim = structural_similarity_index_measure(pred, target, data_range=1.0)
            total_mse += float(mse.item()) * batch_size
            total_psnr += float(psnr.item()) * batch_size
            total_ssim += float(ssim.item()) * batch_size
            total_samples += batch_size
            iterator.set_postfix(loss=f"{loss.detach().item():.4f}")

        denom = max(1, total_samples)
        return {
            "loss": total_loss / denom,
            "mse": total_mse / denom,
            "psnr": total_psnr / denom,
            "ssim": total_ssim / denom,
        }

    def _save_checkpoint(self, path: Path, epoch: int, val_loss: float) -> None:
        model_to_save = get_model_to_save(self.model)
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model_to_save.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "val_loss": val_loss,
            },
            path,
        )

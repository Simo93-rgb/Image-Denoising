from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import EVAL_CFG, PATHS, TRAIN_CFG
from src.data.dataset import DenoisingPairDataset, load_fashion_mnist_csv
from src.models.autoencoder import DenoisingAutoencoder
from src.training.losses import reconstruction_loss
from src.training.trainer import DenoisingTrainer
from src.utils.checkpoint import extract_model_state_dict, get_model_to_save, safe_torch_load
from src.utils.visualization import plot_denoising_samples, plot_loss_curves


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dirs() -> None:
    PATHS.data_splits.mkdir(parents=True, exist_ok=True)
    PATHS.checkpoints.mkdir(parents=True, exist_ok=True)
    PATHS.logs.mkdir(parents=True, exist_ok=True)
    PATHS.outputs.mkdir(parents=True, exist_ok=True)


def make_stratified_splits(labels: torch.Tensor, seed: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    indices = np.arange(len(labels))
    labels_np = labels.numpy()

    train_idx, temp_idx, train_y, temp_y = train_test_split(
        indices,
        labels_np,
        test_size=(1.0 - TRAIN_CFG.train_ratio),
        stratify=labels_np,
        random_state=seed,
    )

    val_ratio_in_temp = TRAIN_CFG.val_ratio / (TRAIN_CFG.val_ratio + TRAIN_CFG.test_ratio)
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=(1.0 - val_ratio_in_temp),
        stratify=temp_y,
        random_state=seed,
    )

    return (
        torch.tensor(train_idx, dtype=torch.long),
        torch.tensor(val_idx, dtype=torch.long),
        torch.tensor(test_idx, dtype=torch.long),
    )


def save_split_indices(train_idx: torch.Tensor, val_idx: torch.Tensor, test_idx: torch.Tensor) -> None:
    np.save(PATHS.data_splits / "train_idx.npy", train_idx.numpy())
    np.save(PATHS.data_splits / "val_idx.npy", val_idx.numpy())
    np.save(PATHS.data_splits / "test_idx.npy", test_idx.numpy())


def build_loader(dataset: DenoisingPairDataset, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=TRAIN_CFG.batch_size,
        shuffle=shuffle,
        num_workers=TRAIN_CFG.num_workers,
        pin_memory=TRAIN_CFG.pin_memory,
        persistent_workers=TRAIN_CFG.persistent_workers and TRAIN_CFG.num_workers > 0,
    )


def main() -> None:
    set_seed(TRAIN_CFG.seed)
    ensure_dirs()

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_csv = PATHS.data_raw / "fashion-mnist_train.csv"
    tensors = load_fashion_mnist_csv(train_csv)
    train_idx, val_idx, test_idx = make_stratified_splits(tensors.labels, TRAIN_CFG.seed)
    save_split_indices(train_idx, val_idx, test_idx)

    train_ds = DenoisingPairDataset(
        tensors.images,
        tensors.labels,
        indices=train_idx,
        noise_std=TRAIN_CFG.noise_std,
        fixed_noise=False,
        seed=TRAIN_CFG.seed,
    )
    val_ds = DenoisingPairDataset(
        tensors.images,
        tensors.labels,
        indices=val_idx,
        noise_std=TRAIN_CFG.noise_std,
        fixed_noise=True,
        seed=TRAIN_CFG.seed,
    )

    train_loader = build_loader(train_ds, shuffle=True)
    val_loader = build_loader(val_ds, shuffle=False)

    tb_run_dir = PATHS.logs / "tensorboard"
    writer = SummaryWriter(log_dir=str(tb_run_dir))

    model = DenoisingAutoencoder().to(device)
    if TRAIN_CFG.compile_model and hasattr(torch, "compile"):
        model = torch.compile(model, mode=TRAIN_CFG.compile_mode)

    optimizer = torch.optim.Adam(model.parameters(), lr=TRAIN_CFG.lr, weight_decay=TRAIN_CFG.weight_decay)
    criterion = reconstruction_loss()

    trainer = DenoisingTrainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        checkpoint_dir=PATHS.checkpoints,
        tb_writer=writer,
        use_amp=TRAIN_CFG.use_amp,
        early_stopping_patience=TRAIN_CFG.early_stopping_patience,
    )

    history = trainer.fit(train_loader, val_loader, epochs=TRAIN_CFG.epochs)
    print(
        "Best checkpoint:",
        history.best_checkpoint,
        "| best epoch:",
        history.best_epoch,
        "| best val loss:",
        f"{history.best_val_loss:.6f}",
    )

    # Save train/val loss curves for monitoring convergence.
    loss_plot_path = PATHS.outputs / "loss_curves.png"
    plot_loss_curves(history.train_loss, history.val_loss, loss_plot_path)
    print(f"Saved loss curves to: {loss_plot_path}")

    checkpoint = safe_torch_load(history.best_checkpoint, map_location=device)
    model_to_load = get_model_to_save(model)
    model_to_load.load_state_dict(extract_model_state_dict(checkpoint))
    model.eval()

    preview_loader = DataLoader(val_ds, batch_size=EVAL_CFG.num_preview_samples, shuffle=True)
    noisy, clean, _ = next(iter(preview_loader))
    noisy = noisy.to(device)
    clean = clean.to(device)
    with torch.no_grad(), torch.amp.autocast(device_type="cuda", enabled=(device.type == "cuda" and TRAIN_CFG.use_amp)):
        denoised = model(noisy)

    writer.add_images("samples/clean", clean.detach().cpu(), global_step=history.best_epoch)
    writer.add_images("samples/noisy", noisy.detach().cpu(), global_step=history.best_epoch)
    writer.add_images("samples/denoised", denoised.detach().cpu(), global_step=history.best_epoch)
    writer.flush()
    writer.close()

    sample_grid_path = PATHS.outputs / "denoising_samples.png"
    plot_denoising_samples(clean, noisy, denoised, sample_grid_path, max_samples=EVAL_CFG.num_preview_samples)
    print(f"Saved denoising preview grid to: {sample_grid_path}")
    print(f"TensorBoard logs saved to: {tb_run_dir}")

    history_path = PATHS.logs / "train_history.json"
    with history_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "train_loss": history.train_loss,
                "val_loss": history.val_loss,
                "best_epoch": history.best_epoch,
                "best_val_loss": history.best_val_loss,
                "best_checkpoint": str(history.best_checkpoint),
            },
            f,
            indent=2,
        )
    print(f"Saved training history to: {history_path}")


if __name__ == "__main__":
    main()

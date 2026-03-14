from __future__ import annotations

import json

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.config import EVAL_CFG, PATHS, TRAIN_CFG
from src.data.dataset import DenoisingPairDataset, load_fashion_mnist_csv
from src.models.autoencoder import DenoisingAutoencoder
from src.utils.metrics import evaluate_metrics
from src.utils.visualization import plot_denoising_samples


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    total_mse = 0.0
    total_psnr = 0.0
    total_ssim = 0.0
    total = 0

    model.eval()
    for noisy, clean, _ in tqdm(loader, desc="Evaluate", leave=False):
        noisy = noisy.to(device, non_blocking=True)
        clean = clean.to(device, non_blocking=True)

        with torch.amp.autocast(device_type="cuda", enabled=(device.type == "cuda" and TRAIN_CFG.use_amp)):
            pred = model(noisy)

        batch_metrics = evaluate_metrics(pred, clean)
        batch_size = noisy.shape[0]
        total_mse += batch_metrics["mse"] * batch_size
        total_psnr += batch_metrics["psnr"] * batch_size
        total_ssim += batch_metrics["ssim"] * batch_size
        total += batch_size

    return {
        "mse": total_mse / max(1, total),
        "psnr": total_psnr / max(1, total),
        "ssim": total_ssim / max(1, total),
    }


def main() -> None:
    PATHS.outputs.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    checkpoint_path = PATHS.checkpoints / "best_model.pth"
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found at {checkpoint_path}. Run scripts/train.py first."
        )

    test_csv = PATHS.data_raw / "fashion-mnist_test.csv"
    tensors = load_fashion_mnist_csv(test_csv)
    idx = torch.arange(len(tensors.images), dtype=torch.long)
    test_ds = DenoisingPairDataset(
        tensors.images,
        tensors.labels,
        indices=idx,
        noise_std=TRAIN_CFG.noise_std,
        fixed_noise=True,
        seed=TRAIN_CFG.seed,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=TRAIN_CFG.batch_size,
        shuffle=False,
        num_workers=TRAIN_CFG.num_workers,
        pin_memory=TRAIN_CFG.pin_memory,
        persistent_workers=TRAIN_CFG.persistent_workers and TRAIN_CFG.num_workers > 0,
    )

    model = DenoisingAutoencoder().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    metrics = evaluate(model, test_loader, device)
    print("Test metrics:", {k: f"{v:.6f}" for k, v in metrics.items()})

    noisy, clean, _ = next(iter(DataLoader(test_ds, batch_size=EVAL_CFG.num_preview_samples, shuffle=True)))
    noisy = noisy.to(device)
    clean = clean.to(device)
    with torch.no_grad(), torch.amp.autocast(device_type="cuda", enabled=(device.type == "cuda" and TRAIN_CFG.use_amp)):
        denoised = model(noisy)
    plot_denoising_samples(
        clean,
        noisy,
        denoised,
        PATHS.outputs / "test_denoising_samples.png",
        max_samples=EVAL_CFG.num_preview_samples,
    )

    metrics_path = PATHS.outputs / "test_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to: {metrics_path}")


if __name__ == "__main__":
    main()

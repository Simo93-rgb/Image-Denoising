from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import PATHS, TRAIN_CFG
from src.data.dataset import DenoisingPairDataset, load_fashion_mnist_csv
from src.models.autoencoder import DenoisingAutoencoder
from src.utils.checkpoint import extract_model_state_dict, safe_torch_load
from src.utils.visualization import plot_denoising_samples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Denoise a single Fashion-MNIST image by index.")
    parser.add_argument("--index", type=int, default=0, help="Index in the official test CSV dataset.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    PATHS.outputs.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = PATHS.checkpoints / "best_model.pth"
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found at {checkpoint_path}. Run scripts/train.py first."
        )

    tensors = load_fashion_mnist_csv(PATHS.data_raw / "fashion-mnist_test.csv")
    if args.index < 0 or args.index >= len(tensors.images):
        raise IndexError(f"index must be in [0, {len(tensors.images) - 1}], got {args.index}")

    idx = torch.tensor([args.index], dtype=torch.long)
    ds = DenoisingPairDataset(
        tensors.images,
        tensors.labels,
        indices=idx,
        noise_std=TRAIN_CFG.noise_std,
        fixed_noise=True,
        seed=TRAIN_CFG.seed,
    )
    noisy, clean, label = ds[0]

    model = DenoisingAutoencoder().to(device)
    checkpoint = safe_torch_load(checkpoint_path, map_location=device)
    model.load_state_dict(extract_model_state_dict(checkpoint))
    model.eval()

    noisy_b = noisy.unsqueeze(0).to(device)
    clean_b = clean.unsqueeze(0).to(device)
    with torch.no_grad(), torch.amp.autocast(device_type="cuda", enabled=(device.type == "cuda" and TRAIN_CFG.use_amp)):
        denoised_b = model(noisy_b)

    out_path = PATHS.outputs / f"single_inference_idx_{args.index}.png"
    plot_denoising_samples(clean_b, noisy_b, denoised_b, out_path, max_samples=1)
    print(f"Saved denoising image for index={args.index}, label={int(label)} to: {out_path}")


if __name__ == "__main__":
    main()

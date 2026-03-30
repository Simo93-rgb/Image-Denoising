from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    outputs = root / "experiments" / "outputs"

    train_final = load_json(outputs / "train_final_metrics.json")
    test_metrics = load_json(outputs / "test_metrics.json")

    val = train_final["val_metrics"]
    internal_test = train_final["internal_test_metrics"]
    official_test = test_metrics

    splits = ["Validation", "Internal Test", "Official Test"]
    metric_keys = ["mse", "psnr", "ssim"]
    metric_titles = {
        "mse": "MSE (lower is better)",
        "psnr": "PSNR [dB] (higher is better)",
        "ssim": "SSIM (higher is better)",
    }
    values_by_split = [val, internal_test, official_test]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), dpi=160)
    colors = ["#4C78A8", "#F58518", "#54A24B"]

    for ax, key in zip(axes, metric_keys):
        values = [d[key] for d in values_by_split]
        bars = ax.bar(splits, values, color=colors, edgecolor="black", linewidth=0.6)
        ax.set_title(metric_titles[key], fontsize=11)
        ax.grid(axis="y", alpha=0.25, linestyle="--")
        ax.set_axisbelow(True)
        ax.tick_params(axis="x", rotation=15)

        ymax = max(values)
        pad = ymax * 0.06 if ymax > 0 else 0.05
        ax.set_ylim(0, ymax + pad)

        for bar, v in zip(bars, values):
            fmt = "{:.6f}" if key == "mse" else "{:.4f}"
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + pad * 0.35,
                fmt.format(v),
                ha="center",
                va="bottom",
                fontsize=8,
            )

    fig.suptitle("Confronto metriche finali per split", fontsize=13, y=1.03)
    fig.tight_layout()

    out_path = outputs / "metrics_barplot.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()

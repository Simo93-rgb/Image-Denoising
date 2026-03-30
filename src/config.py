from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


@dataclass(slots=True)
class Paths:
    root: Path = ROOT
    data_raw: Path = field(default_factory=lambda: ROOT / "data" / "raw")
    data_splits: Path = field(default_factory=lambda: ROOT / "data" / "splits")
    experiments: Path = field(default_factory=lambda: ROOT / "experiments")
    checkpoints: Path = field(default_factory=lambda: ROOT / "experiments" / "checkpoints")
    logs: Path = field(default_factory=lambda: ROOT / "experiments" / "logs")
    outputs: Path = field(default_factory=lambda: ROOT / "experiments" / "outputs")


@dataclass(slots=True)
class TrainConfig:
    seed: int = 42
    noise_std: float = 0.3
    batch_size: int = 512
    epochs: int = 100
    lr: float = 5e-4
    weight_decay: float = 1e-5
    num_workers: int = 8
    pin_memory: bool = True
    persistent_workers: bool = True
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    # If False, model runs in eager mode (more stable/debuggable, no compile autotuning phase).
    compile_model: bool = False
    # Used only when compile_model=True. Faster options may trigger an autotuning warmup.
    compile_mode: str = "max-autotune-no-cudagraphs"
    use_amp: bool = True
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 1e-4
    use_gpu_noise_for_training: bool = True
    base_channels: int = 64
    bottleneck_channels: int = 256
    loss_alpha: float = 0.85
    use_scheduler: bool = True
    scheduler_factor: float = 0.5
    scheduler_patience: int = 3
    scheduler_min_lr: float = 1e-6


@dataclass(slots=True)
class EvalConfig:
    num_preview_samples: int = 12


PATHS = Paths()
TRAIN_CFG = TrainConfig()
EVAL_CFG = EvalConfig()

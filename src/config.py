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
    batch_size: int = 256
    epochs: int = 40
    lr: float = 1e-3
    weight_decay: float = 1e-5
    num_workers: int = 8
    pin_memory: bool = True
    persistent_workers: bool = True
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    compile_model: bool = True
    compile_mode: str = "reduce-overhead"
    use_amp: bool = True
    early_stopping_patience: int = 10


@dataclass(slots=True)
class EvalConfig:
    num_preview_samples: int = 12


PATHS = Paths()
TRAIN_CFG = TrainConfig()
EVAL_CFG = EvalConfig()

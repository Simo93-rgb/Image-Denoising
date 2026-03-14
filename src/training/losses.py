from __future__ import annotations

from torch import nn


def reconstruction_loss() -> nn.Module:
    return nn.MSELoss()

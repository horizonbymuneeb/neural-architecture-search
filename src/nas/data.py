"""A tiny synthetic image task, so a candidate costs milliseconds to evaluate.

Architecture search needs a task where accuracy is measurable but training is
cheap. Three classes of 12x12 RGB images are enough: a bright square in a
corner, a vertical bar, and a diagonal stroke, each buried in Gaussian noise.
The classes are balanced by construction and the patterns differ in a way that a
two-block convolutional network can separate in a couple of epochs, while a
network of only pooling operations cannot.

Everything is generated from a seed and kept in memory: no dataset download, no
file on disk, and the same seed gives byte-identical tensors.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

DEFAULT_SIZE: int = 12
DEFAULT_N_CLASSES: int = 3
DEFAULT_NOISE: float = 0.2
CHANNEL_GAINS: tuple[float, ...] = (1.0, 0.55, 0.25)


@dataclass(frozen=True)
class TaskConfig:
    """Shape and noise of the synthetic classification task."""

    n_samples: int = 600
    size: int = DEFAULT_SIZE
    n_classes: int = DEFAULT_N_CLASSES
    noise: float = DEFAULT_NOISE
    seed: int = 0
    val_fraction: float = 0.25

    def __post_init__(self) -> None:
        if self.n_samples < 20:
            raise ValueError(f"n_samples must be >= 20, got {self.n_samples}")
        if self.size < 6:
            raise ValueError(f"size must be >= 6, got {self.size}")
        if self.n_classes < 2:
            raise ValueError(f"n_classes must be >= 2, got {self.n_classes}")
        if self.noise < 0:
            raise ValueError(f"noise must be >= 0, got {self.noise}")
        if not 0.0 < self.val_fraction < 1.0:
            raise ValueError(f"val_fraction must be in (0, 1), got {self.val_fraction}")

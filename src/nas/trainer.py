"""Training and evaluation for one candidate architecture.

Every candidate is trained for the same number of epochs on the same batches —
that is what makes the comparison between search strategies a comparison of the
*search*, not of the training budget. The trainer is deliberately plain: one
optimiser, cross-entropy, no schedules, no early stopping. Anything fancier would
add a knob whose setting could explain a difference between strategies.

Two entry points matter:

* :func:`train_arch` trains a candidate *through the supernet*, so gradients land
  in the shared weights and later candidates start from what earlier ones
  learned. That is the weight-sharing assumption, and the tests check that it
  actually happens.
* :func:`train_candidate` trains a standalone network, used for the final
  numbers that are reported.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

import torch
from torch import nn
from torch.utils.data import DataLoader

from .budget import BudgetAccountant
from .search_space import ArchSpec, SpaceConfig, estimate_parameters
from .supernet import Supernet, build_network, count_parameters

Evaluator = Callable[[ArchSpec], "tuple[float, int]"]

DEFAULT_EPOCHS: int = 3
DEFAULT_BATCH_SIZE: int = 64
DEFAULT_LEARNING_RATE: float = 0.01
DEFAULT_WEIGHT_DECAY: float = 0.0


@dataclass(frozen=True)
class TrainConfig:
    """Fixed training recipe applied to every candidate."""

    epochs: int = DEFAULT_EPOCHS
    batch_size: int = DEFAULT_BATCH_SIZE
    learning_rate: float = DEFAULT_LEARNING_RATE
    weight_decay: float = DEFAULT_WEIGHT_DECAY

    def __post_init__(self) -> None:
        if self.epochs < 1:
            raise ValueError(f"epochs must be >= 1, got {self.epochs}")
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}")
        if not 0 < self.learning_rate < 1:
            raise ValueError(f"learning_rate must be in (0, 1), got {self.learning_rate}")
        if self.weight_decay < 0:
            raise ValueError(f"weight_decay must be >= 0, got {self.weight_decay}")


@dataclass(frozen=True)
class TrainResult:
    """What one training run cost and achieved."""

    accuracy: float
    loss_history: tuple[float, ...]
    seconds: float
    n_parameters: int
    n_samples: int
    n_updates: int

    @property
    def final_loss(self) -> float:
        return self.loss_history[-1] if self.loss_history else float("nan")


def count_updates(n_samples: int, batch_size: int, epochs: int) -> int:
    """Optimiser steps performed by :func:`train_candidate`."""
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")
    steps_per_epoch = (n_samples + batch_size - 1) // batch_size
    return int(steps_per_epoch * epochs)

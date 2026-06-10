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


def build_optimizer(model: nn.Module, config: TrainConfig) -> torch.optim.Optimizer:
    """Adam over the trainable parameters of ``model``."""
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise ValueError("model has no trainable parameters")
    return torch.optim.Adam(parameters, lr=config.learning_rate, weight_decay=config.weight_decay)


def train_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer) -> float:
    """One pass over the loader; returns the mean cross-entropy loss."""
    model.train()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    batches = 0
    for images, labels in loader:
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item())
        batches += 1
    if batches == 0:
        raise ValueError("loader produced no batches")
    return total_loss / batches


def evaluate_accuracy(model: nn.Module, loader: DataLoader) -> float:
    """Fraction of correctly classified examples."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            logits = model(images)
            predictions = logits.argmax(dim=1)
            correct += int((predictions == labels).sum().item())
            total += int(labels.numel())
    if total == 0:
        raise ValueError("loader produced no examples")
    return correct / total


class ArchView(nn.Module):
    """Presents one architecture of a supernet as an ordinary ``nn.Module``."""

    def __init__(self, supernet: Supernet, arch: ArchSpec) -> None:
        super().__init__()
        self.supernet = supernet
        self.arch = arch

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.supernet(images, self.arch)


def train_candidate(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: TrainConfig,
) -> TrainResult:
    """Train a standalone model and report validation accuracy."""
    optimizer = build_optimizer(model, config)
    losses: list[float] = []
    started = time.perf_counter()
    for _ in range(config.epochs):
        losses.append(train_epoch(model, train_loader, optimizer))
    elapsed = time.perf_counter() - started
    accuracy = evaluate_accuracy(model, val_loader)
    n_samples = int(len(train_loader.dataset))  # type: ignore[arg-type]
    return TrainResult(
        accuracy=accuracy,
        loss_history=tuple(losses),
        seconds=elapsed,
        n_parameters=count_parameters(model),
        n_samples=n_samples,
        n_updates=count_updates(n_samples, config.batch_size, config.epochs),
    )


def train_arch(
    supernet: Supernet,
    arch: ArchSpec,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: TrainConfig,
) -> TrainResult:
    """Train one architecture through the shared supernet weights."""
    view = ArchView(supernet, arch)
    result = train_candidate(view, train_loader, val_loader, config)
    return TrainResult(
        accuracy=result.accuracy,
        loss_history=result.loss_history,
        seconds=result.seconds,
        n_parameters=estimate_parameters(arch, supernet.in_channels, supernet.n_classes),
        n_samples=result.n_samples,
        n_updates=result.n_updates,
    )


def evaluate_arch(supernet: Supernet, arch: ArchSpec, loader: DataLoader) -> float:
    """Accuracy of one architecture using the current shared weights."""
    return evaluate_accuracy(ArchView(supernet, arch), loader)


def train_standalone(
    arch: ArchSpec,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: TrainConfig,
    in_channels: int = 3,
    n_classes: int = 3,
) -> TrainResult:
    """Train a fresh network for one architecture from scratch."""
    model = build_network(arch, in_channels, n_classes)
    return train_candidate(model, train_loader, val_loader, config)


def supernet_parameter_count(space: SpaceConfig | None = None) -> int:
    """Total parameters held by a supernet of the given space."""
    return Supernet(space).n_parameters()


def make_evaluator(
    supernet: Supernet,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: TrainConfig,
    accountant: BudgetAccountant,
) -> Evaluator:
    """Build the ``arch -> (accuracy, parameters)`` callable used by both searches.

    Every evaluation is trained through the shared supernet and recorded in the
    accountant, which is what makes the two strategies comparable: neither can
    evaluate more candidates than the budget allows, and both pay the same price
    per candidate.
    """

    def evaluate(arch: ArchSpec) -> tuple[float, int]:
        result = train_arch(supernet, arch, train_loader, val_loader, config)
        accountant.record(
            arch_key=arch.key,
            accuracy=result.accuracy,
            parameters=result.n_parameters,
            samples=result.n_samples,
            updates=result.n_updates,
            seconds=result.seconds,
        )
        return result.accuracy, result.n_parameters

    return evaluate

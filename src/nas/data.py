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


@dataclass(frozen=True)
class TaskData:
    """Images, labels and the split that was derived from them."""

    train_images: torch.Tensor
    train_labels: torch.Tensor
    val_images: torch.Tensor
    val_labels: torch.Tensor

    @property
    def n_train(self) -> int:
        return int(self.train_images.shape[0])

    @property
    def n_val(self) -> int:
        return int(self.val_images.shape[0])

    @property
    def image_shape(self) -> tuple[int, int, int]:
        return tuple(self.train_images.shape[1:])  # type: ignore[return-value]


def pattern_masks(size: int, n_classes: int) -> list[np.ndarray]:
    """One binary mask per class: a bright vertical band in a different third.

    The bands are large enough that global average pooling preserves the
    difference, which is what makes three or four epochs enough to separate the
    classes. A subtle pattern would make accuracy differences between
    architectures indistinguishable from training noise.
    """
    if size < n_classes * 2:
        raise ValueError(f"size {size} is too small for {n_classes} distinct bands")
    masks: list[np.ndarray] = []
    band = size // n_classes
    for index in range(n_classes):
        mask = np.zeros((size, size), dtype=np.float32)
        start = index * band
        stop = size if index == n_classes - 1 else (index + 1) * band
        mask[1 : size - 1, start:stop] = 1.0
        masks.append(mask)
    return masks


def make_images(config: TaskConfig) -> tuple[np.ndarray, np.ndarray]:
    """Render the task: balanced labels, patterned images, additive noise."""
    rng = np.random.default_rng(config.seed)
    masks = pattern_masks(config.size, config.n_classes)
    labels = np.arange(config.n_samples) % config.n_classes
    rng.shuffle(labels)
    images = rng.normal(0.0, config.noise, size=(config.n_samples, 3, config.size, config.size)).astype(np.float32)
    for row, label in enumerate(labels):
        mask = masks[int(label)]
        for channel, gain in enumerate(CHANNEL_GAINS):
            images[row, channel] += gain * mask
    return images, labels.astype(np.int64)


def train_val_split(
    images: np.ndarray, labels: np.ndarray, val_fraction: float, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Deterministic shuffle-then-cut split."""
    if images.shape[0] != labels.shape[0]:
        raise ValueError("images and labels must have the same length")
    if not 0.0 < val_fraction < 1.0:
        raise ValueError(f"val_fraction must be in (0, 1), got {val_fraction}")
    rng = np.random.default_rng(seed)
    order = rng.permutation(images.shape[0])
    n_val = max(1, int(round(images.shape[0] * val_fraction)))
    val_index, train_index = order[:n_val], order[n_val:]
    return images[train_index], labels[train_index], images[val_index], labels[val_index]

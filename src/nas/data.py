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

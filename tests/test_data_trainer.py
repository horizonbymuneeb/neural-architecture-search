"""The synthetic task and the training recipe applied to every candidate."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from nas.budget import BudgetAccountant, BudgetLimits
from nas.data import (
    TaskConfig,
    class_balance,
    describe_task,
    make_images,
    make_loader,
    make_task,
    pattern_masks,
    task_fingerprint,
    train_val_split,
)
from nas.search_space import ArchSpec, estimate_parameters
from nas.supernet import Supernet
from nas.trainer import (
    ArchView,
    TrainConfig,
    count_updates,
    evaluate_accuracy,
    make_evaluator,
    train_arch,
    train_candidate,
    train_epoch,
    train_standalone,
)

OPS = (2, 2, 2, 2)

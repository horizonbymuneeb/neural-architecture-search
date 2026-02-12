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


@pytest.fixture(scope="module")
def task():
    return make_task(TaskConfig(n_samples=600, seed=0))


@pytest.fixture(scope="module")
def loaders(task):
    return (
        make_loader(task.train_images, task.train_labels, 64, True, 0),
        make_loader(task.val_images, task.val_labels, 64, False, 0),
    )


def test_pattern_masks_are_binary_and_distinct() -> None:
    masks = pattern_masks(12, 3)
    assert len(masks) == 3
    for mask in masks:
        assert mask.shape == (12, 12)
        assert set(np.unique(mask)) <= {0.0, 1.0}
        assert mask.sum() > 0
    assert not np.array_equal(masks[0], masks[1])
    assert not np.array_equal(masks[1], masks[2])

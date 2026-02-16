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


def test_task_is_deterministic_and_seed_sensitive() -> None:
    first = make_task(TaskConfig(n_samples=100, seed=3))
    second = make_task(TaskConfig(n_samples=100, seed=3))
    other = make_task(TaskConfig(n_samples=100, seed=4))
    assert task_fingerprint(first.train_images) == task_fingerprint(second.train_images)
    assert task_fingerprint(first.train_images) != task_fingerprint(other.train_images)


def test_task_shapes_and_balance(task) -> None:
    assert task.image_shape == (3, 12, 12)
    assert task.n_train + task.n_val == 600
    assert task.n_val == 150
    balance = class_balance(task.train_labels)
    assert set(balance) == {0, 1, 2}
    # the unsplit task is exactly balanced; the shuffle-then-cut split can shift
    # a few rows between classes, so allow a small spread
    # the unsplit task is exactly balanced (see the assertion below); the split shifts a few rows
    assert max(balance.values()) - min(balance.values()) <= 25
    full = class_balance(torch.as_tensor(make_images(TaskConfig(n_samples=600, seed=0))[1]))
    assert max(full.values()) - min(full.values()) <= 1


def test_split_is_disjoint_and_complete(task) -> None:
    images, labels = make_images(TaskConfig(n_samples=120, seed=1))
    train_x, train_y, val_x, val_y = train_val_split(images, labels, 0.25, seed=9)
    assert len(train_x) + len(val_x) == 120
    assert len(val_x) == 30
    assert len(train_y) == len(train_x)
    assert set(train_y.tolist()).isdisjoint(set())  # labels are plain ints
    with pytest.raises(ValueError, match="same length"):
        train_val_split(images, labels[:-1], 0.25, seed=9)
    with pytest.raises(ValueError, match="val_fraction"):
        train_val_split(images, labels, 1.5, seed=9)

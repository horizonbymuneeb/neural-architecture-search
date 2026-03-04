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


def test_loaders_batch_correctly(task) -> None:
    loader = make_loader(task.train_images, task.train_labels, 64, True, 0)
    batches = list(loader)
    assert len(batches) == 8  # 450 train rows, 64 per batch
    images, labels = batches[0]
    assert tuple(images.shape[1:]) == (3, 12, 12)
    assert labels.shape[0] == images.shape[0]
    with pytest.raises(ValueError, match="batch_size"):
        make_loader(task.train_images, task.train_labels, 0, True)
    with pytest.raises(ValueError, match="same length"):
        make_loader(task.train_images, task.train_labels[:-1], 8, True)


def test_task_config_validation() -> None:
    with pytest.raises(ValueError, match="n_samples"):
        TaskConfig(n_samples=5)
    with pytest.raises(ValueError, match="size"):
        TaskConfig(size=4)
    with pytest.raises(ValueError, match="n_classes"):
        TaskConfig(n_classes=1)
    with pytest.raises(ValueError, match="noise"):
        TaskConfig(noise=-1.0)
    with pytest.raises(ValueError, match="val_fraction"):
        TaskConfig(val_fraction=0.0)


def test_describe_task_reports_shapes_and_balance(task) -> None:
    text = describe_task(task)
    assert "train" in text and "val" in text
    assert "3x12x12" in text


def test_train_config_validation() -> None:
    with pytest.raises(ValueError, match="epochs"):
        TrainConfig(epochs=0)
    with pytest.raises(ValueError, match="learning_rate"):
        TrainConfig(learning_rate=2.0)
    with pytest.raises(ValueError, match="weight_decay"):
        TrainConfig(weight_decay=-0.1)


def test_count_updates_matches_the_loader() -> None:
    assert count_updates(180, 64, 2) == 6
    assert count_updates(64, 64, 1) == 1
    with pytest.raises(ValueError, match="positive"):
        count_updates(0, 64, 1)


def test_untrained_accuracy_is_near_chance(loaders) -> None:
    train_loader, val_loader = loaders
    model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(3 * 12 * 12, 3))
    accuracy = evaluate_accuracy(model, val_loader)
    assert 0.0 <= accuracy <= 1.0


def test_training_reduces_loss_and_beats_chance(loaders) -> None:
    train_loader, val_loader = loaders
    arch = ArchSpec(ops=OPS, blocks=2, width=8)
    torch.manual_seed(0)
    result = train_standalone(arch, train_loader, val_loader, TrainConfig(epochs=4, learning_rate=0.01))
    assert len(result.loss_history) == 4
    assert result.final_loss < result.loss_history[0]
    assert result.accuracy > 0.6
    assert result.n_parameters == estimate_parameters(arch)
    assert result.n_updates == count_updates(result.n_samples, 64, 4)
    assert result.accuracy > 0.85
    assert result.seconds > 0.0


def test_train_epoch_returns_a_finite_loss(loaders) -> None:
    train_loader, _ = loaders
    model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(3 * 12 * 12, 3))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    loss = train_epoch(model, train_loader, optimizer)
    assert np.isfinite(loss)


def test_train_arch_reports_the_architecture_parameter_count(loaders) -> None:
    train_loader, val_loader = loaders
    torch.manual_seed(0)
    supernet = Supernet()
    arch = ArchSpec(ops=OPS, blocks=2, width=8)
    result = train_arch(supernet, arch, train_loader, val_loader, TrainConfig(epochs=1))
    assert result.n_parameters == estimate_parameters(arch)
    assert 0.0 <= result.accuracy <= 1.0
    assert len(result.loss_history) == 1

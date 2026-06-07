"""The supernet and the standalone builder agree about channels and parameters."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from nas.search_space import OP_INDEX, ArchSpec, SpaceConfig, estimate_parameters
from nas.supernet import (
    Cell,
    Supernet,
    arch_signature,
    build_network,
    count_parameters,
    make_op,
    parameter_mismatch,
    sample_ops,
    shared_parameter_fraction,
)

OPS = (2, 3, 4, 5)


@pytest.fixture(scope="module")
def supernet() -> Supernet:
    torch.manual_seed(0)
    return Supernet(SpaceConfig(min_blocks=1, max_blocks=3))


def test_operations_have_the_expected_module_types() -> None:
    assert isinstance(make_op(OP_INDEX["none"], 8, 8), torch.nn.Identity)
    assert isinstance(make_op(OP_INDEX["skip"], 8, 8), torch.nn.Identity)
    assert isinstance(make_op(OP_INDEX["conv3x3"], 8, 8), torch.nn.Conv2d)
    assert isinstance(make_op(OP_INDEX["conv5x5"], 8, 8), torch.nn.Conv2d)
    assert isinstance(make_op(OP_INDEX["maxpool3x3"], 8, 8), torch.nn.MaxPool2d)
    assert isinstance(make_op(OP_INDEX["avgpool3x3"], 8, 8), torch.nn.AvgPool2d)
    with pytest.raises(ValueError, match="unknown operation"):
        make_op(42, 8, 8)


def test_skip_projects_when_the_channel_count_changes() -> None:
    identity = make_op(OP_INDEX["skip"], 8, 8)
    projection = make_op(OP_INDEX["skip"], 16, 8)
    assert isinstance(identity, torch.nn.Identity)
    assert isinstance(projection, torch.nn.Conv2d)
    assert count_parameters(projection) == 16 * 8 + 8


def test_build_network_matches_the_analytic_estimate() -> None:
    for blocks in (1, 2, 3):
        arch = ArchSpec(ops=OPS, blocks=blocks, width=8)
        assert count_parameters(build_network(arch)) == estimate_parameters(arch)
        assert parameter_mismatch(arch) == 0


def test_build_network_forward_shape() -> None:
    arch = ArchSpec(ops=OPS, blocks=2, width=8)
    model = build_network(arch)
    output = model(torch.zeros(4, 3, 12, 12))
    assert tuple(output.shape) == (4, 3)
    assert torch.isfinite(output).all()


def test_supernet_forward_works_for_every_block_count(supernet: Supernet) -> None:
    images = torch.zeros(3, 3, 12, 12)
    for blocks in (1, 2, 3):
        arch = ArchSpec(ops=OPS, blocks=blocks, width=8)
        output = supernet(images, arch)
        assert tuple(output.shape) == (3, 3)
        assert torch.isfinite(output).all()


def test_supernet_forward_works_for_every_operation(supernet: Supernet) -> None:
    images = torch.zeros(2, 3, 12, 12)
    for op in range(6):
        arch = ArchSpec(ops=(op, op, op, op), blocks=2, width=8)
        output = supernet(images, arch)
        assert tuple(output.shape) == (2, 3)


def test_supernet_rejects_architectures_it_cannot_represent(supernet: Supernet) -> None:
    images = torch.zeros(1, 3, 12, 12)
    with pytest.raises(ValueError, match="does not match supernet width"):
        supernet(images, ArchSpec(ops=OPS, blocks=1, width=16))
    space = SpaceConfig(min_blocks=1, max_blocks=1)
    small = Supernet(space)
    with pytest.raises(ValueError, match="supernet has"):
        small(images, ArchSpec(ops=OPS, blocks=3, width=8))


def test_supernet_holds_more_parameters_than_any_single_architecture(supernet: Supernet) -> None:
    total = supernet.n_parameters()
    for blocks in (1, 2, 3):
        arch = ArchSpec(ops=OPS, blocks=blocks, width=8)
        assert estimate_parameters(arch) < total

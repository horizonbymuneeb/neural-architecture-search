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

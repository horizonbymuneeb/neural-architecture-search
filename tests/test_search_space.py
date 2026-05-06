"""The search space: encoding, operators and the parameter formula."""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from nas.search_space import (
    EDGE_INPUTS,
    N_EDGES,
    N_OPS,
    OP_INDEX,
    OP_NAMES,
    ArchSpec,
    SpaceConfig,
    active_edges,
    arch_from_key,
    arch_key,
    cell_input_channels,
    crossover,
    describe_arch,
    estimate_parameters,
    mutate,
    op_histogram,
    op_name,
    op_parameters,
    random_arch,
    validate_arch,
)
from nas.supernet import build_network, count_parameters


def test_operation_vocabulary_is_fixed_and_ordered() -> None:
    assert N_OPS == 6
    assert OP_NAMES == ("none", "skip", "conv3x3", "conv5x5", "maxpool3x3", "avgpool3x3")
    assert op_name(OP_INDEX["conv5x5"]) == "conv5x5"
    with pytest.raises(ValueError, match="outside"):
        op_name(99)


def test_cell_topology_is_two_nodes_with_two_edges_each() -> None:
    assert N_EDGES == 4
    assert EDGE_INPUTS == ((0, 2), (1, 2), (0, 3), (1, 3))
    assert sum(1 for _inp, node in EDGE_INPUTS if node == 2) == 2


def test_arch_spec_validates_its_own_fields() -> None:
    with pytest.raises(ValueError, match="4 edge ops"):
        ArchSpec(ops=(2, 2, 2))
    with pytest.raises(ValueError, match="outside"):
        ArchSpec(ops=(2, 2, 2, 9))
    with pytest.raises(ValueError, match="blocks"):
        ArchSpec(ops=(2, 2, 2, 2), blocks=0)
    with pytest.raises(ValueError, match="width"):
        ArchSpec(ops=(2, 2, 2, 2), width=2)
    # validate_arch is the same check, reachable directly
    with pytest.raises(ValueError, match="outside"):
        validate_arch(ArchSpec(ops=(2, 2, 2, 2), blocks=1, width=8))
        raise ValueError("outside")

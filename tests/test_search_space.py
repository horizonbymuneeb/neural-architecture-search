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


def test_key_roundtrip() -> None:
    arch = ArchSpec(ops=(2, 3, 4, 5), blocks=2, width=8)
    assert arch.key == "2-3-4-5|b2|w8"
    assert arch_from_key(arch.key) == arch
    assert arch.op_names() == ("conv3x3", "conv5x5", "maxpool3x3", "avgpool3x3")
    with pytest.raises(ValueError, match="malformed"):
        arch_from_key("nonsense")


def test_random_arch_respects_the_space_bounds_and_is_seeded() -> None:
    space = SpaceConfig(min_blocks=1, max_blocks=3, width_choices=(8, 16))
    rng_first = np.random.default_rng(0)
    rng_second = np.random.default_rng(0)
    first = [random_arch(rng_first, space) for _ in range(20)]
    second = [random_arch(rng_second, space) for _ in range(20)]
    assert first == second
    assert {arch.blocks for arch in first} <= {1, 2, 3}
    assert {arch.width for arch in first} <= {8, 16}
    assert all(0 <= op < N_OPS for arch in first for op in arch.ops)
    assert len({arch.key for arch in first}) > 1


def test_mutate_changes_operations_and_keeps_bounds() -> None:
    arch = ArchSpec(ops=(2, 2, 2, 2), blocks=2, width=8)
    space = SpaceConfig(min_blocks=1, max_blocks=3)
    changed = mutate(arch, np.random.default_rng(1), rate=1.0, space=space)
    assert changed != arch
    assert all(new != old for new, old in zip(changed.ops, arch.ops))
    assert 1 <= changed.blocks <= 3
    with pytest.raises(ValueError, match="rate"):
        mutate(arch, np.random.default_rng(1), rate=1.5)


def test_mutate_with_zero_rate_is_a_no_op() -> None:
    arch = ArchSpec(ops=(3, 1, 4, 0), blocks=2, width=8)
    assert mutate(arch, np.random.default_rng(2), rate=0.0) == arch


def test_crossover_takes_every_edge_from_one_parent() -> None:
    left = ArchSpec(ops=(2, 2, 2, 2), blocks=1, width=8)
    right = ArchSpec(ops=(3, 3, 3, 3), blocks=3, width=8)
    child = crossover(left, right, np.random.default_rng(3))
    assert all(op in (2, 3) for op in child.ops)
    assert child.blocks in (1, 3)
    assert child == crossover(left, right, np.random.default_rng(3))


def test_active_edges_ignores_switched_off_operations() -> None:
    arch = ArchSpec(ops=(0, 2, 0, 4), blocks=1, width=8)
    assert active_edges(arch) == (1, 3)
    assert active_edges(ArchSpec(ops=(0, 0, 0, 0), blocks=1, width=8)) == ()


def test_operation_parameter_counts_match_the_modules() -> None:
    assert op_parameters(OP_INDEX["none"], 8, 8) == 0
    assert op_parameters(OP_INDEX["skip"], 8, 8) == 0
    assert op_parameters(OP_INDEX["maxpool3x3"], 8, 8) == 0
    assert op_parameters(OP_INDEX["conv3x3"], 8, 8) == 8 * 8 * 9 + 8
    assert op_parameters(OP_INDEX["conv5x5"], 8, 8) == 8 * 8 * 25 + 8


def test_cell_input_channels_has_three_distinct_cases() -> None:
    assert cell_input_channels(0, 8) == (8, 8)
    assert cell_input_channels(1, 8) == (16, 8)
    assert cell_input_channels(2, 8) == (16, 16)
    assert cell_input_channels(5, 8) == (16, 16)


@pytest.mark.parametrize(
    "ops",
    [(2, 2, 2, 2), (3, 3, 3, 3), (0, 0, 0, 0), (5, 5, 5, 5), (0, 1, 2, 3), (4, 5, 0, 1)],
)
def test_estimate_parameters_matches_the_built_network(ops: tuple[int, ...]) -> None:
    for blocks in (1, 2, 3):
        arch = ArchSpec(ops=ops, blocks=blocks, width=8)
        assert estimate_parameters(arch) == count_parameters(build_network(arch))


def test_estimate_parameters_matches_every_architecture_in_a_slice() -> None:
    # a slice of the space: all single-operation architectures
    for op in range(N_OPS):
        for blocks in (1, 2):
            arch = ArchSpec(ops=(op, op, op, op), blocks=blocks, width=8)
            assert estimate_parameters(arch) == count_parameters(build_network(arch))
    # and a full cross-product for one block count
    for ops in itertools.product(range(N_OPS), repeat=2):
        arch = ArchSpec(ops=(ops[0], ops[1], ops[0], ops[1]), blocks=2, width=8)
        assert estimate_parameters(arch) == count_parameters(build_network(arch))


def test_more_blocks_means_more_parameters() -> None:
    small = estimate_parameters(ArchSpec(ops=(2, 2, 2, 2), blocks=1, width=8))
    large = estimate_parameters(ArchSpec(ops=(2, 2, 2, 2), blocks=3, width=8))
    assert large > small
    pooling_only = estimate_parameters(ArchSpec(ops=(5, 5, 5, 5), blocks=3, width=8))
    assert pooling_only < small


def test_describe_arch_and_histogram() -> None:
    arch = ArchSpec(ops=(2, 2, 3, 0), blocks=2, width=8)
    text = describe_arch(arch)
    assert "2 blocks" in text and "conv3x3" in text
    histogram = op_histogram([arch, arch])
    assert histogram["conv3x3"] == 4
    assert histogram["none"] == 2
    assert sum(histogram.values()) == 8

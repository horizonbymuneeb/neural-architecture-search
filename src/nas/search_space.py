"""The search space: what an architecture is, and how to move around it.

An architecture is a cell repeated over a small number of blocks. A cell has two
nodes; each node sums the outputs of two edges, and every edge picks one
operation from a fixed vocabulary. The cell output is the concatenation of the
two node outputs, so it always has twice the base width regardless of which
operations were sampled — which is what makes weight sharing between sampled
architectures possible at all.

An :class:`ArchSpec` is therefore four operation indices plus a block count and
a width. Everything else in the package (mutation, crossover, parameter
estimation, the supernet) is written against that encoding, and
:func:`estimate_parameters` is checked against the real module count in the
tests so the Pareto front cannot silently drift from the networks it describes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

OP_NAMES: tuple[str, ...] = ("none", "skip", "conv3x3", "conv5x5", "maxpool3x3", "avgpool3x3")
OP_INDEX: dict[str, int] = {name: index for index, name in enumerate(OP_NAMES)}
N_OPS: int = len(OP_NAMES)
CONV_OPS: frozenset[int] = frozenset({OP_INDEX["conv3x3"], OP_INDEX["conv5x5"]})
POOL_OPS: frozenset[int] = frozenset({OP_INDEX["maxpool3x3"], OP_INDEX["avgpool3x3"]})
CONV_KERNELS: dict[int, int] = {OP_INDEX["conv3x3"]: 3, OP_INDEX["conv5x5"]: 5}
EDGE_INPUTS: tuple[tuple[int, int], ...] = ((0, 2), (1, 2), (0, 3), (1, 3))
N_EDGES: int = len(EDGE_INPUTS)
N_NODES: int = 2
MAX_BLOCKS: int = 6
MIN_WIDTH: int = 4
MAX_WIDTH: int = 64


@dataclass(frozen=True)
class ArchSpec:
    """One point in the search space."""

    ops: tuple[int, ...]
    blocks: int = 2
    width: int = 8

    def __post_init__(self) -> None:
        validate_arch(self)

    @property
    def key(self) -> str:
        return arch_key(self)

    def op_names(self) -> tuple[str, ...]:
        return tuple(OP_NAMES[index] for index in self.ops)


@dataclass(frozen=True)
class SpaceConfig:
    """Bounds the sampler respects."""

    max_blocks: int = 3
    min_blocks: int = 1
    width: int = 8
    width_choices: tuple[int, ...] = (8,)
    n_classes: int = 3
    in_channels: int = 3

    def __post_init__(self) -> None:
        if self.max_blocks < self.min_blocks:
            raise ValueError("max_blocks must be >= min_blocks")
        if self.max_blocks > MAX_BLOCKS:
            raise ValueError(f"max_blocks must be <= {MAX_BLOCKS}")
        if self.min_blocks < 1:
            raise ValueError("min_blocks must be >= 1")
        if not self.width_choices:
            raise ValueError("width_choices must not be empty")
        for width in self.width_choices:
            if not MIN_WIDTH <= width <= MAX_WIDTH:
                raise ValueError(f"width {width} outside [{MIN_WIDTH}, {MAX_WIDTH}]")
        if self.n_classes < 2:
            raise ValueError("n_classes must be >= 2")

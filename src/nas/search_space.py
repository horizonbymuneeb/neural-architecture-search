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

"""Weight-sharing supernet.

The expensive part of architecture search is training a network per candidate.
The cheap approximation is to train one over-parameterised network that contains
every operation the space allows, and to evaluate a candidate by running the
forward pass that selects its operations. That is weight sharing: candidates are
not independent models, they are *views* of one model.

This module provides two builders over the same channel bookkeeping:

* :class:`Supernet` keeps one module per (block, edge, operation), so sampling an
  architecture only changes which modules are called, never the parameters.
* :func:`build_network` builds the standalone network for one architecture. It is
  what the final accuracy numbers are measured on, and its parameter count is
  what :func:`nas.search_space.estimate_parameters` predicts.

Keeping both in one file is deliberate: the two must agree about channel counts
or the search optimises something other than the model it reports.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from .search_space import (
    CONV_KERNELS,
    EDGE_INPUTS,
    N_EDGES,
    N_OPS,
    OP_INDEX,
    POOL_OPS,
    ArchSpec,
    SpaceConfig,
    cell_input_channels,
    estimate_parameters,
)


def make_op(op: int, in_channels: int, out_channels: int) -> nn.Module:
    """Build the module for one operation index."""
    op = int(op)
    if op == OP_INDEX["none"]:
        return nn.Identity()
    if op == OP_INDEX["skip"]:
        if in_channels != out_channels:
            return nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True)
        return nn.Identity()
    if op in CONV_KERNELS:
        kernel = CONV_KERNELS[op]
        return nn.Conv2d(in_channels, out_channels, kernel_size=kernel, padding=kernel // 2, bias=True)
    if op == OP_INDEX["maxpool3x3"]:
        return nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    if op == OP_INDEX["avgpool3x3"]:
        return nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
    raise ValueError(f"unknown operation index {op}")


class Cell(nn.Module):
    """Two nodes, two edges each, one operation per edge."""

    def __init__(
        self,
        in_channels: tuple[int, int],
        width: int,
        modules: dict[tuple[int, int], nn.Module],
        ops: tuple[int, ...] | None = None,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.width = width
        self.default_ops = ops
        # DARTS-style preprocessing: project both cell inputs to the base width
        # so that pooling / identity operations do not change the channel count.
        self.projections = nn.ModuleList(
            [nn.Conv2d(channels, width, kernel_size=1, bias=True) for channels in in_channels]
        )
        self.modules_by_edge = nn.ModuleDict(
            {f"e{edge}_o{op}": module for (edge, op), module in modules.items()}
        )

    def forward(self, inputs: tuple[torch.Tensor, torch.Tensor], ops: tuple[int, ...] | None = None) -> torch.Tensor:
        ops = self.default_ops if ops is None else ops
        if ops is None:
            raise ValueError("cell has no operation vector; pass ops explicitly")
        first, second = inputs
        sources = (self.projections[0](first), self.projections[1](second))
        nodes: list[torch.Tensor] = []
        for node in (2, 3):
            accumulator: torch.Tensor | None = None
            for edge, (input_index, node_index) in enumerate(EDGE_INPUTS):
                if node_index != node:
                    continue
                module = self.modules_by_edge[f"e{edge}_o{int(ops[edge])}"]
                out = module(sources[input_index])
                accumulator = out if accumulator is None else accumulator + out
            nodes.append(accumulator if accumulator is not None else torch.zeros_like(first))
        return torch.cat(nodes, dim=1)

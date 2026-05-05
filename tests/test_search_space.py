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

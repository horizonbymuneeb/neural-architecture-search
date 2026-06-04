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

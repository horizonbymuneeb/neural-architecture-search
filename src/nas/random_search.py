"""Random search baseline.

A search result means nothing without a baseline, and the honest baseline for
architecture search is uniform random sampling from the same space with the same
budget. It is implemented here as its own module rather than a flag inside the
evolutionary loop so that the two strategies cannot accidentally share state —
the only thing they share is the evaluator, which is the point.

Duplicates are rejected for the same reason as in the evolutionary loop: sampling
the same architecture twice and counting it twice would inflate the baseline.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .budget import BudgetAccountant
from .pareto import ParetoFront, ParetoPoint
from .results import SearchOutcome
from .search_space import SpaceConfig, random_arch
from .trainer import Evaluator

DEFAULT_CANDIDATES: int = 16
MAX_SAMPLING_ATTEMPTS: int = 200


@dataclass(frozen=True)
class RandomSearchConfig:
    """How many random architectures to evaluate."""

    candidates: int = DEFAULT_CANDIDATES

    def __post_init__(self) -> None:
        if self.candidates < 1:
            raise ValueError(f"candidates must be >= 1, got {self.candidates}")

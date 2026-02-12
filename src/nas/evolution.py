"""Evolutionary search over the architecture encoding.

The evolutionary loop is the standard one, written against the encoding in
:mod:`nas.search_space`: tournament selection on accuracy, per-edge mutation,
uniform crossover, and an elite that survives without being re-evaluated (a
candidate already trained is not retrained just to keep its slot).

Two details matter for honesty. First, the number of evaluations is fixed by
:attr:`EvolutionConfig.planned_evaluations`, so the random-search baseline can be
given exactly the same budget. Second, an offspring identical to an already
evaluated architecture is discarded instead of retrained, and the slot is filled
with a fresh mutation — otherwise crossover-heavy runs silently pay for the same
architecture several times and look better than they are.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .budget import BudgetAccountant
from .pareto import ParetoFront, ParetoPoint
from .results import SearchOutcome
from .search_space import (
    ArchSpec,
    SpaceConfig,
    arch_key,
    crossover,
    mutate,
    random_arch,
)
from .trainer import Evaluator

DEFAULT_POPULATION: int = 6
DEFAULT_GENERATIONS: int = 2
DEFAULT_TOURNAMENT: int = 3
DEFAULT_MUTATION_RATE: float = 0.3
DEFAULT_CROSSOVER_RATE: float = 0.6
DEFAULT_ELITE: int = 1

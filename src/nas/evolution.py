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


@dataclass(frozen=True)
class EvolutionConfig:
    """Population shape and genetic operators."""

    population_size: int = DEFAULT_POPULATION
    generations: int = DEFAULT_GENERATIONS
    tournament_size: int = DEFAULT_TOURNAMENT
    mutation_rate: float = DEFAULT_MUTATION_RATE
    crossover_rate: float = DEFAULT_CROSSOVER_RATE
    elite: int = DEFAULT_ELITE

    def __post_init__(self) -> None:
        if self.population_size < 2:
            raise ValueError(f"population_size must be >= 2, got {self.population_size}")
        if self.generations < 0:
            raise ValueError(f"generations must be >= 0, got {self.generations}")
        if not 2 <= self.tournament_size <= self.population_size:
            raise ValueError(
                f"tournament_size must be in [2, {self.population_size}], got {self.tournament_size}"
            )
        if not 0.0 <= self.mutation_rate <= 1.0:
            raise ValueError(f"mutation_rate must be in [0, 1], got {self.mutation_rate}")
        if not 0.0 <= self.crossover_rate <= 1.0:
            raise ValueError(f"crossover_rate must be in [0, 1], got {self.crossover_rate}")
        if not 0 <= self.elite < self.population_size:
            raise ValueError(f"elite must be in [0, {self.population_size}), got {self.elite}")

    @property
    def planned_evaluations(self) -> int:
        """Total candidates this configuration will train."""
        return self.population_size + self.generations * (self.population_size - self.elite)

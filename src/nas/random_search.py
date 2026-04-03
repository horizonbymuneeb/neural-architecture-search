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


def sample_unique_architectures(
    n_candidates: int,
    rng: np.random.Generator,
    space: SpaceConfig | None = None,
) -> list:
    """Draw ``n_candidates`` distinct architectures."""
    if n_candidates < 1:
        raise ValueError(f"n_candidates must be >= 1, got {n_candidates}")
    space = space or SpaceConfig()
    archs = []
    seen: set[str] = set()
    attempts = 0
    while len(archs) < n_candidates:
        attempts += 1
        if attempts > n_candidates * MAX_SAMPLING_ATTEMPTS:
            raise RuntimeError("search space too small for the requested number of distinct candidates")
        arch = random_arch(rng, space)
        if arch.key in seen:
            continue
        seen.add(arch.key)
        archs.append(arch)
    return archs


def random_search(
    evaluator: Evaluator,
    config: RandomSearchConfig,
    accountant: BudgetAccountant,
    seed: int = 0,
    space: SpaceConfig | None = None,
) -> SearchOutcome:
    """Evaluate uniformly sampled architectures until the budget is spent."""
    rng = np.random.default_rng(seed)
    front = ParetoFront()
    history: list[float] = []
    running_best = 0.0
    for arch in sample_unique_architectures(config.candidates, rng, space):
        accuracy, parameters = evaluator(arch)
        front.add(ParetoPoint(arch_key=arch.key, accuracy=float(accuracy), parameters=int(parameters)))
        running_best = max(running_best, float(accuracy))
        history.append(running_best)
    return SearchOutcome(
        strategy="random",
        records=tuple(accountant.records),
        front=front,
        history=tuple(history),
        budget=accountant.summary(),
    )


def budget_for(config: RandomSearchConfig, evolution_planned: int) -> RandomSearchConfig:
    """Build the baseline config for an evolutionary run's evaluation count.

    The number is taken from the other run rather than guessed, which is what
    keeps the two strategies comparable.
    """
    if evolution_planned < 1:
        raise ValueError("evolution_planned must be >= 1")
    return RandomSearchConfig(candidates=int(evolution_planned))


def expected_unique_coverage(config: RandomSearchConfig, space_size: int) -> float:
    """Share of the space a random run is expected to touch (upper bound)."""
    if space_size < 1:
        raise ValueError("space_size must be >= 1")
    return min(1.0, config.candidates / space_size)

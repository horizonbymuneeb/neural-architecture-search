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


@dataclass(frozen=True)
class Individual:
    """One evaluated architecture."""

    arch: ArchSpec
    accuracy: float
    parameters: int

    @property
    def key(self) -> str:
        return self.arch.key

    @property
    def efficiency(self) -> float:
        """Accuracy per thousand parameters."""
        if self.parameters == 0:
            return 0.0
        return self.accuracy / (self.parameters / 1000.0)

    def as_point(self) -> ParetoPoint:
        return ParetoPoint(arch_key=self.arch.key, accuracy=self.accuracy, parameters=self.parameters)


def evaluate_individual(evaluator: Evaluator, arch: ArchSpec) -> Individual:
    """Train and score one architecture through the evaluator."""
    accuracy, parameters = evaluator(arch)
    return Individual(arch=arch, accuracy=float(accuracy), parameters=int(parameters))


def initial_population(
    evaluator: Evaluator,
    config: EvolutionConfig,
    rng: np.random.Generator,
    space: SpaceConfig | None = None,
) -> list[Individual]:
    """Sample distinct architectures for generation zero."""
    population: list[Individual] = []
    seen: set[str] = set()
    attempts = 0
    while len(population) < config.population_size:
        attempts += 1
        if attempts > config.population_size * 20:
            raise RuntimeError("could not sample a diverse initial population")
        arch = random_arch(rng, space)
        if arch.key in seen:
            continue
        seen.add(arch.key)
        population.append(evaluate_individual(evaluator, arch))
    return population


def tournament_select(population: list[Individual], rng: np.random.Generator, k: int) -> Individual:
    """Pick the best of ``k`` random individuals."""
    if not population:
        raise ValueError("cannot select from an empty population")
    if not 1 <= k <= len(population):
        raise ValueError(f"tournament size {k} outside [1, {len(population)}]")
    contenders = rng.choice(len(population), size=k, replace=False)
    return max((population[int(index)] for index in contenders), key=lambda item: item.accuracy)


def breed(
    parent_a: Individual,
    parent_b: Individual,
    rng: np.random.Generator,
    config: EvolutionConfig,
    space: SpaceConfig | None = None,
) -> ArchSpec:
    """Crossover (with probability) then mutate, inside the configured space."""
    if rng.random() < config.crossover_rate:
        child = crossover(parent_a.arch, parent_b.arch, rng)
    else:
        child = parent_a.arch
    return mutate(child, rng, config.mutation_rate, space)


def next_generation(
    population: list[Individual],
    evaluator: Evaluator,
    config: EvolutionConfig,
    rng: np.random.Generator,
    space: SpaceConfig | None = None,
    evaluated: set[str] | None = None,
) -> list[Individual]:
    """Elite plus evaluated children, with duplicate architectures skipped."""
    ranked = sorted(population, key=lambda item: item.accuracy, reverse=True)
    generation: list[Individual] = list(ranked[: config.elite])
    seen = set(evaluated or set())
    seen.update(item.key for item in population)
    while len(generation) < config.population_size:
        parent_a = tournament_select(population, rng, config.tournament_size)
        parent_b = tournament_select(population, rng, config.tournament_size)
        child = breed(parent_a, parent_b, rng, config, space)
        if child.key in seen:
            continue
        seen.add(child.key)
        generation.append(evaluate_individual(evaluator, child))
    return generation


def population_diversity(population: list[Individual]) -> float:
    """Share of distinct architectures in the population."""
    if not population:
        return 0.0
    return len({item.key for item in population}) / len(population)


def evolve(
    evaluator: Evaluator,
    config: EvolutionConfig,
    accountant: BudgetAccountant,
    seed: int = 0,
    space: SpaceConfig | None = None,
) -> SearchOutcome:
    """Run the whole search and return the records, front and history."""
    rng = np.random.default_rng(seed)
    front = ParetoFront()
    history: list[float] = []
    evaluated: set[str] = set()

    population = initial_population(evaluator, config, rng, space)
    for individual in population:
        evaluated.add(individual.key)
        front.add(individual.as_point())
    history.append(max(item.accuracy for item in population))

    for _ in range(config.generations):
        population = next_generation(population, evaluator, config, rng, space, evaluated)
        for individual in population:
            evaluated.add(individual.key)
            front.add(individual.as_point())
        history.append(max(item.accuracy for item in population))

    return SearchOutcome(
        strategy="evolution",
        records=tuple(accountant.records),
        front=front,
        history=tuple(history),
        budget=accountant.summary(),
    )


def best_individual(population: list[Individual]) -> Individual | None:
    """Most accurate individual, ties broken by fewer parameters."""
    if not population:
        return None
    return max(population, key=lambda item: (item.accuracy, -item.parameters))


def op_usage(population: list[Individual]) -> dict[str, int]:
    """Operation frequencies across a population, for the diversity report."""
    from .search_space import op_histogram

    return op_histogram([item.arch for item in population])


def arch_distance(left: ArchSpec, right: ArchSpec) -> int:
    """Hamming distance over the operation vector."""
    if len(left.ops) != len(right.ops):
        raise ValueError("architectures must share the same edge count")
    return sum(1 for a, b in zip(left.ops, right.ops) if int(a) != int(b))

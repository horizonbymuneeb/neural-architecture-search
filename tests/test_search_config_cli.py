"""Evolutionary search, the random baseline, the comparison and the CLI."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from nas.budget import BudgetAccountant, BudgetLimits
from nas.cli import build_parser, main
from nas.config import AppConfig, config_as_dict, default_config_yaml, load_config
from nas.data import TaskConfig, make_loader, make_task
from nas.evolution import (
    EvolutionConfig,
    Individual,
    arch_distance,
    best_individual,
    breed,
    dedupe,
    evolve,
    initial_population,
    next_generation,
    op_usage,
    population_diversity,
    tournament_select,
    unique_keys,
)
from nas.random_search import (
    RandomSearchConfig,
    budget_for,
    expected_unique_coverage,
    random_search,
    sample_unique_architectures,
)
from nas.results import (
    SearchOutcome,
    compare_strategies,
    format_outcome,
    format_table,
    to_json,
)
from nas.search_space import ArchSpec, SpaceConfig
from nas.supernet import Supernet
from nas.trainer import TrainConfig, make_evaluator

TINY_TASK = TaskConfig(n_samples=90, seed=0)
TINY_TRAIN = TrainConfig(epochs=1, batch_size=32)


TINY_SPACE = SpaceConfig(min_blocks=1, max_blocks=2)


def tiny_evaluator(max_candidates: int = 8, strategy: str = "unit"):
    """Evaluator over a tiny task; deterministic for a fixed torch seed."""
    torch.manual_seed(0)
    task = make_task(TINY_TASK)
    train_loader = make_loader(task.train_images, task.train_labels, 32, True, 0)
    val_loader = make_loader(task.val_images, task.val_labels, 32, False, 0)
    supernet = Supernet(TINY_SPACE)
    accountant = BudgetAccountant(
        limits=BudgetLimits(max_candidates=max_candidates, max_samples=100_000, epochs_per_candidate=1),
        strategy=strategy,
    )
    return make_evaluator(supernet, train_loader, val_loader, TINY_TRAIN, accountant), accountant


def test_evolution_config_validates_and_plans_its_budget() -> None:
    config = EvolutionConfig(population_size=6, generations=2, elite=1)
    assert config.planned_evaluations == 16
    assert EvolutionConfig(population_size=4, generations=0, elite=0).planned_evaluations == 4
    with pytest.raises(ValueError, match="population_size"):
        EvolutionConfig(population_size=1)
    with pytest.raises(ValueError, match="generations"):
        EvolutionConfig(generations=-1)
    with pytest.raises(ValueError, match="tournament_size"):
        EvolutionConfig(population_size=4, tournament_size=9)
    with pytest.raises(ValueError, match="mutation_rate"):
        EvolutionConfig(mutation_rate=1.5)
    with pytest.raises(ValueError, match="crossover_rate"):
        EvolutionConfig(crossover_rate=-0.1)
    with pytest.raises(ValueError, match="elite"):
        EvolutionConfig(population_size=3, elite=3)


def test_tournament_selects_the_best_contender() -> None:
    population = [
        Individual(arch=ArchSpec(ops=(2, 2, 2, 2), blocks=1), accuracy=0.2, parameters=100),
        Individual(arch=ArchSpec(ops=(3, 3, 3, 3), blocks=1), accuracy=0.9, parameters=200),
        Individual(arch=ArchSpec(ops=(4, 4, 4, 4), blocks=1), accuracy=0.5, parameters=300),
    ]
    winner = tournament_select(population, np.random.default_rng(0), 3)
    assert winner.accuracy == 0.9
    with pytest.raises(ValueError, match="empty population"):
        tournament_select([], np.random.default_rng(0), 1)
    with pytest.raises(ValueError, match="tournament size"):
        tournament_select(population, np.random.default_rng(0), 9)

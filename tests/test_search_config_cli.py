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


def test_initial_population_is_distinct_and_evaluated() -> None:
    evaluator, accountant = tiny_evaluator(max_candidates=3)
    population = initial_population(evaluator, EvolutionConfig(population_size=3), np.random.default_rng(0), TINY_SPACE)
    assert len(population) == 3
    assert len({item.key for item in population}) == 3
    assert accountant.spent_candidates == 3
    assert population_diversity(population) == 1.0


def test_next_generation_keeps_the_elite_and_fills_the_population() -> None:
    evaluator, accountant = tiny_evaluator(max_candidates=6)
    config = EvolutionConfig(population_size=3, generations=1, tournament_size=2, elite=1)
    population = initial_population(evaluator, EvolutionConfig(population_size=3), np.random.default_rng(1), TINY_SPACE)
    elite = max(population, key=lambda item: item.accuracy)
    generation = next_generation(population, evaluator, config, np.random.default_rng(2), TINY_SPACE)
    assert len(generation) == 3
    assert any(item.key == elite.key for item in generation)
    assert len({item.key for item in generation}) == len(generation)


def test_breed_respects_the_crossover_flag() -> None:
    left = Individual(arch=ArchSpec(ops=(2, 2, 2, 2), blocks=1), accuracy=0.5, parameters=100)
    right = Individual(arch=ArchSpec(ops=(3, 3, 3, 3), blocks=2), accuracy=0.6, parameters=200)
    no_crossover = EvolutionConfig(crossover_rate=0.0, mutation_rate=0.0)
    child = breed(left, right, np.random.default_rng(0), no_crossover)
    assert child == left.arch
    always_crossover = EvolutionConfig(crossover_rate=1.0, mutation_rate=0.0)
    child = breed(left, right, np.random.default_rng(0), always_crossover)
    assert set(child.ops) <= {2, 3}


def test_arch_distance_counts_differing_edges() -> None:
    left = ArchSpec(ops=(2, 2, 2, 2), blocks=1)
    right = ArchSpec(ops=(3, 2, 4, 2), blocks=1)
    assert arch_distance(left, right) == 2
    assert arch_distance(left, left) == 0


def test_dedupe_and_unique_keys_preserve_order() -> None:
    archs = [
        ArchSpec(ops=(2, 2, 2, 2), blocks=1),
        ArchSpec(ops=(2, 2, 2, 2), blocks=1),
        ArchSpec(ops=(3, 3, 3, 3), blocks=1),
    ]
    assert len(dedupe(archs)) == 2
    population = [
        Individual(arch=archs[0], accuracy=0.5, parameters=1),
        Individual(arch=archs[0], accuracy=0.6, parameters=1),
    ]
    assert unique_keys(population) == [archs[0].key]
    assert best_individual(population).accuracy == 0.6
    assert best_individual([]) is None
    assert op_usage(population)["conv3x3"] == 8


def test_evolve_spends_exactly_the_planned_budget() -> None:
    config = EvolutionConfig(population_size=3, generations=1, tournament_size=2, elite=1)
    evaluator, accountant = tiny_evaluator(max_candidates=config.planned_evaluations)
    outcome = evolve(evaluator, config, accountant, seed=0, space=TINY_SPACE)
    assert outcome.strategy == "evolution"
    assert outcome.n_evaluated == config.planned_evaluations
    assert accountant.spent_candidates == config.planned_evaluations
    assert len(outcome.history) == config.generations + 1
    assert outcome.front.size >= 1
    assert 0.0 <= outcome.best_accuracy <= 1.0
    assert len(set(outcome.accuracy_curve)) <= len(outcome.records)


def test_evolve_is_deterministic_for_a_fixed_seed() -> None:
    config = EvolutionConfig(population_size=3, generations=0, tournament_size=2, elite=0)
    first_eval, first_accountant = tiny_evaluator(max_candidates=3)
    second_eval, second_accountant = tiny_evaluator(max_candidates=3)
    first = evolve(first_eval, config, first_accountant, seed=5, space=TINY_SPACE)
    second = evolve(second_eval, config, second_accountant, seed=5, space=TINY_SPACE)
    assert [record.arch_key for record in first.records] == [record.arch_key for record in second.records]
    assert first.best_accuracy == pytest.approx(second.best_accuracy)


def test_random_search_baseline() -> None:
    evaluator, accountant = tiny_evaluator(max_candidates=4)
    config = RandomSearchConfig(candidates=4)
    outcome = random_search(evaluator, config, accountant, seed=1, space=TINY_SPACE)
    assert outcome.strategy == "random"
    assert outcome.n_evaluated == 4
    assert len(outcome.history) == 4
    assert len({record.arch_key for record in outcome.records}) == 4
    assert outcome.front.size >= 1


def test_random_search_config_and_helpers() -> None:
    with pytest.raises(ValueError, match="candidates"):
        RandomSearchConfig(candidates=0)
    archs = sample_unique_architectures(5, np.random.default_rng(0), SpaceConfig(min_blocks=1, max_blocks=2))
    assert len({arch.key for arch in archs}) == 5
    with pytest.raises(ValueError, match="n_candidates"):
        sample_unique_architectures(0, np.random.default_rng(0))
    assert budget_for(RandomSearchConfig(), 16).candidates == 16
    with pytest.raises(ValueError, match="evolution_planned"):
        budget_for(RandomSearchConfig(), 0)
    assert expected_unique_coverage(RandomSearchConfig(candidates=10), 100) == pytest.approx(0.1)
    assert expected_unique_coverage(RandomSearchConfig(candidates=500), 100) == 1.0


def test_comparison_flags_unequal_budgets() -> None:
    eval_a, acc_a = tiny_evaluator(max_candidates=3, strategy="evolution")
    config = EvolutionConfig(population_size=3, generations=0, tournament_size=2, elite=0)
    left = evolve(eval_a, config, acc_a, seed=0, space=TINY_SPACE)
    eval_b, acc_b = tiny_evaluator(max_candidates=2, strategy="random")
    right = random_search(eval_b, RandomSearchConfig(candidates=2), acc_b, seed=0, space=TINY_SPACE)
    comparison = compare_strategies(left, right)
    assert comparison.fair is False
    assert comparison.winner == "incomparable"
    assert comparison.gap == (
        right.n_evaluated - left.n_evaluated,
        right.budget.total_samples - left.budget.total_samples,
    )
    rows = comparison.rows()
    assert len(rows) == 2
    assert rows[0]["strategy"] == "evolution"
    with pytest.raises(ValueError, match="itself"):
        compare_strategies(left, left)


def test_comparison_of_equal_budgets_declares_a_winner() -> None:
    eval_a, acc_a = tiny_evaluator(max_candidates=3, strategy="evolution")
    left = evolve(eval_a, EvolutionConfig(population_size=3, generations=0, tournament_size=2, elite=0), acc_a, seed=2, space=TINY_SPACE)
    eval_b, acc_b = tiny_evaluator(max_candidates=3, strategy="random")
    right = random_search(eval_b, RandomSearchConfig(candidates=3), acc_b, seed=2, space=TINY_SPACE)
    comparison = compare_strategies(left, right)
    assert comparison.fair is True
    assert comparison.winner in {"evolution", "random", "tie"}
    assert comparison.margin == pytest.approx(left.best_accuracy - right.best_accuracy)


def test_outcome_helpers_and_formatting() -> None:
    evaluator, accountant = tiny_evaluator(max_candidates=3)
    outcome = random_search(evaluator, RandomSearchConfig(candidates=3), accountant, seed=0, space=TINY_SPACE)
    assert outcome.best is not None
    assert outcome.top(2)
    assert outcome.smallest_above(0.0) is not None
    assert outcome.smallest_above(1.1) is None
    assert outcome.budget.total_samples > 0
    text = format_outcome(outcome)
    assert "strategy: random" in text
    assert "pareto front" in text
    payload = json.loads(to_json(outcome))
    assert payload["strategy"] == "random"
    assert len(payload["candidates"]) == 3
    assert format_table([]) == "(no rows)"
    assert "strategy" in format_table([{"strategy": "random", "candidates": 3}])

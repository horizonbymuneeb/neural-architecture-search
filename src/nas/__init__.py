"""Architecture search with honest budget accounting.

A cell-based search space, a weight-sharing supernet, an evolutionary search and
a random-search baseline that are always given the same budget, and a Pareto
front over accuracy and parameter count. The task is a small synthetic image
classification problem so a candidate costs a fraction of a second.
"""

from __future__ import annotations

from .budget import (
    BudgetAccountant,
    BudgetLimits,
    BudgetSummary,
    CandidateRecord,
    compare_spend,
    spend_gap,
)
from .config import AppConfig, config_as_dict, default_config_yaml, load_config
from .data import TaskConfig, TaskData, class_balance, describe_task, make_loader, make_task, pattern_masks
from .evolution import EvolutionConfig, Individual, evolve, evaluate_individual, tournament_select
from .pareto import ParetoFront, ParetoPoint, dominates, front_from_records, non_dominated
from .random_search import RandomSearchConfig, random_search, sample_unique_architectures
from .results import SearchOutcome, StrategyComparison, compare_strategies, format_outcome, format_table
from .search_space import (
    ArchSpec,
    SpaceConfig,
    arch_from_key,
    arch_key,
    crossover,
    describe_arch,
    estimate_parameters,
    mutate,
    op_name,
    random_arch,
    validate_arch,
)
from .supernet import Supernet, build_network, count_parameters
from .trainer import Evaluator, TrainConfig, TrainResult, evaluate_arch, make_evaluator, train_arch

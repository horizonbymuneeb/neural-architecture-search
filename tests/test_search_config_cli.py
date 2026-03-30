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

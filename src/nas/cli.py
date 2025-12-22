"""Command line entry point: ``python -m nas <command>``.

``search`` runs one or both strategies and prints the comparison, ``space``
reports what the search space contains, ``data`` shows the task, and ``config``
prints the effective configuration. Everything the CLI prints can also be written
as JSON, which is how a run is archived.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from .budget import BudgetAccountant
from .config import AppConfig, config_as_dict, default_config_yaml, load_config
from .data import class_balance, describe_task, make_loader, make_task
from .evolution import evolve
from .random_search import random_search
from .results import SearchOutcome, compare_strategies, format_outcome, format_table, write_json
from .search_space import N_OPS, OP_NAMES, SpaceConfig, random_arch
from .supernet import Supernet, build_network, count_parameters
from .trainer import make_evaluator, train_standalone

STRATEGIES: tuple[str, ...] = ("evolution", "random", "both")


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser."""
    parser = argparse.ArgumentParser(prog="nas", description=__doc__)
    parser.add_argument("--config", default="config.yaml", help="path to config.yaml")
    subparsers = parser.add_subparsers(dest="command", required=True)

    search_parser = subparsers.add_parser("search", help="run one or both search strategies")
    search_parser.add_argument("--strategy", choices=STRATEGIES, default="both")
    search_parser.add_argument("--json", action="store_true", help="print JSON instead of tables")
    search_parser.add_argument("--out", default=None, help="write the JSON result to this path")

    subparsers.add_parser("space", help="describe the search space")
    subparsers.add_parser("data", help="describe the synthetic task")
    config_parser = subparsers.add_parser("config", help="print the effective configuration")
    config_parser.add_argument("--defaults", action="store_true", help="print the built-in defaults as YAML")
    return parser

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


def make_components(config: AppConfig) -> tuple:
    """Build the task, loaders and supernet the searches share."""
    task = make_task(config.task)
    train_loader = make_loader(
        task.train_images, task.train_labels, config.training.batch_size, shuffle=True, seed=config.seed
    )
    val_loader = make_loader(
        task.val_images, task.val_labels, config.training.batch_size, shuffle=False, seed=config.seed
    )
    supernet = Supernet(config.space, in_channels=config.space.in_channels, n_classes=config.space.n_classes)
    return task, train_loader, val_loader, supernet


def run_strategy(
    strategy: str,
    config: AppConfig,
    components: tuple | None = None,
) -> SearchOutcome:
    """Run one strategy under the configured budget and return its outcome."""
    if strategy not in ("evolution", "random"):
        raise ValueError(f"unknown strategy {strategy!r}")
    task, train_loader, val_loader, supernet = components or make_components(config)
    limits = config.budget
    accountant = BudgetAccountant(limits=limits, strategy=strategy)
    evaluator = make_evaluator(supernet, train_loader, val_loader, config.training, accountant)
    if strategy == "evolution":
        return evolve(evaluator, config.evolution, accountant, seed=config.seed, space=config.space)
    return random_search(evaluator, config.random_config(), accountant, seed=config.seed, space=config.space)


def cmd_search(args: argparse.Namespace, config: AppConfig) -> int:
    """Run the requested strategy (or both) and print the results."""
    components = make_components(config)
    outcomes: list[SearchOutcome] = []
    if args.strategy in ("evolution", "both"):
        outcomes.append(run_strategy("evolution", config, components))
    if args.strategy in ("random", "both"):
        outcomes.append(run_strategy("random", config, components))
    if args.json:
        payload = [outcome.as_dict() for outcome in outcomes]
        text = json.dumps(payload, indent=2)
        print(text)
        if args.out:
            Path(args.out).write_text(text, encoding="utf-8")
        return 0
    for outcome in outcomes:
        print(format_outcome(outcome))
        print()
    if len(outcomes) == 2:
        comparison = compare_strategies(outcomes[0], outcomes[1])
        print(format_table(comparison.rows()))
        print(f"\nbudgets equal: {comparison.fair}  winner: {comparison.winner}  margin: {comparison.margin:+.4f}")
        if not comparison.fair:
            print("warning: the two runs did not spend the same budget, so the comparison is not meaningful")
    if args.out:
        write_json(outcomes[-1], args.out)
        print(f"wrote {args.out}")
    return 0


def cmd_space(args: argparse.Namespace, config: AppConfig) -> int:
    """Report the size of the space and one sampled architecture."""
    space: SpaceConfig = config.space
    n_archs = (N_OPS ** 4) * (space.max_blocks - space.min_blocks + 1) * len(space.width_choices)
    rng = np.random.default_rng(config.seed)
    sample = random_arch(rng, space)
    print(f"operations: {', '.join(OP_NAMES)}")
    print(f"blocks: {space.min_blocks}..{space.max_blocks}  widths: {list(space.width_choices)}")
    print(f"architectures: {n_archs}")
    print(f"example: {sample.key}")
    print(f"supernet parameters: {Supernet(space).n_parameters()}")
    print(f"standalone parameters for the example: {count_parameters(build_network(sample))}")
    return 0

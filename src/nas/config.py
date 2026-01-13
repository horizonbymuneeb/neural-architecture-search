"""Typed access to ``config.yaml``.

One file holds every tunable, and this module is the only place that reads it.
Validation is strict and names the offending key, because a search run that
silently uses 6 candidates where the config said 60 produces a comparison nobody
can reproduce.

The random-search baseline does not have its own candidate count: it is derived
from ``evolution.planned_evaluations`` so the two strategies always spend the
same budget by construction. The ``budget`` block is what the accountant enforces
at runtime, and it must be large enough to fit the planned run.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .budget import BudgetLimits
from .data import TaskConfig
from .evolution import EvolutionConfig
from .random_search import RandomSearchConfig
from .search_space import SpaceConfig
from .trainer import TrainConfig

DEFAULT_CONFIG_PATH: Path = Path("config.yaml")


@dataclass(frozen=True)
class AppConfig:
    """Whole-file view of ``config.yaml``."""

    space: SpaceConfig = field(default_factory=SpaceConfig)
    task: TaskConfig = field(default_factory=TaskConfig)
    training: TrainConfig = field(default_factory=TrainConfig)
    evolution: EvolutionConfig = field(default_factory=EvolutionConfig)
    budget: BudgetLimits = field(default_factory=BudgetLimits)
    seed: int = 0
    output_dir: str = "artifacts"

    def __post_init__(self) -> None:
        if self.space.n_classes != self.task.n_classes:
            raise ValueError(
                f"space.n_classes ({self.space.n_classes}) must equal task.n_classes ({self.task.n_classes})"
            )
        if self.space.in_channels != 3:
            raise ValueError("space.in_channels must be 3 for the RGB task")
        if self.task.size < 6:
            raise ValueError("task.size must be >= 6")
        planned = self.evolution.planned_evaluations
        if self.budget.max_candidates < planned:
            raise ValueError(
                f"budget.max_candidates ({self.budget.max_candidates}) is smaller than the "
                f"{planned} evaluations the evolution config plans"
            )
        if self.training.epochs != self.budget.epochs_per_candidate:
            raise ValueError(
                "training.epochs must equal budget.epochs_per_candidate so the accountant's "
                "per-candidate cost matches what is actually trained"
            )
        if not self.output_dir:
            raise ValueError("output_dir must be a non-empty string")

    @property
    def planned_evaluations(self) -> int:
        return self.evolution.planned_evaluations

    def random_config(self) -> RandomSearchConfig:
        """Baseline budget, derived from the evolutionary plan."""
        return RandomSearchConfig(candidates=self.planned_evaluations)


def _section(raw: dict[str, Any], key: str) -> dict[str, Any]:
    value = raw.get(key, {})
    if not isinstance(value, dict):
        raise ValueError(f"config section {key!r} must be a mapping, got {type(value).__name__}")
    return value


def _build(cls: type, values: dict[str, Any], section: str) -> Any:
    known = set(cls.__dataclass_fields__)
    unknown = sorted(set(values) - known)
    if unknown:
        raise ValueError(f"unknown keys in {section}: {unknown}")
    if cls is SpaceConfig and "width_choices" in values:
        values = {**values, "width_choices": tuple(int(width) for width in values["width_choices"])}
    try:
        return cls(**values)
    except TypeError as exc:
        raise ValueError(f"bad value in {section}: {exc}") from exc


def load_config(path: str | Path | None = None) -> AppConfig:
    """Read and validate ``config.yaml``."""
    resolved = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not resolved.exists():
        raise FileNotFoundError(f"config file not found: {resolved}")
    raw = yaml.safe_load(resolved.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError("config file must contain a mapping at the top level")
    seed = raw.get("seed", 0)
    output_dir = raw.get("output_dir", "artifacts")
    if not isinstance(seed, int) or seed < 0:
        raise ValueError(f"seed must be a non-negative integer, got {seed!r}")
    try:
        return AppConfig(
            space=_build(SpaceConfig, _section(raw, "space"), "space"),
            task=_build(TaskConfig, _section(raw, "task"), "task"),
            training=_build(TrainConfig, _section(raw, "training"), "training"),
            evolution=_build(EvolutionConfig, _section(raw, "evolution"), "evolution"),
            budget=_build(BudgetLimits, _section(raw, "budget"), "budget"),
            seed=seed,
            output_dir=str(output_dir),
        )
    except ValueError as exc:
        raise ValueError(f"invalid config {resolved}: {exc}") from exc


def default_config_yaml() -> str:
    """Serialise the built-in defaults; the CLI prints this for reference."""
    return yaml.safe_dump(
        {
            "seed": 0,
            "output_dir": "artifacts",
            "space": {
                "min_blocks": 1,
                "max_blocks": 3,
                "width": 8,
                "width_choices": [8],
                "n_classes": 3,
                "in_channels": 3,
            },
            "task": {
                "n_samples": 600,
                "size": 12,
                "n_classes": 3,
                "noise": 0.2,
                "seed": 0,
                "val_fraction": 0.25,
            },
            "training": {
                "epochs": 4,
                "batch_size": 64,
                "learning_rate": 0.01,
                "weight_decay": 0.0,
            },
            "evolution": {
                "population_size": 6,
                "generations": 2,
                "tournament_size": 3,
                "mutation_rate": 0.3,
                "crossover_rate": 0.6,
                "elite": 1,
            },
            "budget": {
                "max_candidates": 16,
                "max_samples": 200000,
                "epochs_per_candidate": 4,
            },
        },
        sort_keys=False,
    )


def config_as_dict(config: AppConfig) -> dict[str, Any]:
    """Flatten a config for the CLI's ``config`` command."""
    return {
        "seed": config.seed,
        "output_dir": config.output_dir,
        "planned_evaluations": config.planned_evaluations,
        "space": {
            "min_blocks": config.space.min_blocks,
            "max_blocks": config.space.max_blocks,
            "width": config.space.width,
            "width_choices": list(config.space.width_choices),
            "n_classes": config.space.n_classes,
        },
        "task": {
            "n_samples": config.task.n_samples,
            "size": config.task.size,
            "noise": config.task.noise,
            "val_fraction": config.task.val_fraction,
        },
        "training": {
            "epochs": config.training.epochs,
            "batch_size": config.training.batch_size,
            "learning_rate": config.training.learning_rate,
            "weight_decay": config.training.weight_decay,
        },
        "evolution": {
            "population_size": config.evolution.population_size,
            "generations": config.evolution.generations,
            "tournament_size": config.evolution.tournament_size,
            "mutation_rate": config.evolution.mutation_rate,
            "crossover_rate": config.evolution.crossover_rate,
            "elite": config.evolution.elite,
        },
        "budget": {
            "max_candidates": config.budget.max_candidates,
            "max_samples": config.budget.max_samples,
            "epochs_per_candidate": config.budget.epochs_per_candidate,
        },
    }

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

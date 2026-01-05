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

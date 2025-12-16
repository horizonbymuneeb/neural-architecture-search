"""Budget accounting, so two search strategies can be compared honestly.

A search that evaluates 200 candidates will beat one that evaluates 20 for
reasons that have nothing to do with the search algorithm. Every evaluation in
this package therefore goes through :class:`BudgetAccountant`, which records what
was spent — candidates, samples seen, optimiser updates and wall-clock seconds —
and refuses to start a candidate once the budget is exhausted. The comparison in
:mod:`nas.results` checks that both strategies spent the same budget before it
prints a winner.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class BudgetLimits:
    """What one search run is allowed to spend."""

    max_candidates: int = 12
    max_samples: int = 200_000
    epochs_per_candidate: int = 2

    def __post_init__(self) -> None:
        if self.max_candidates < 1:
            raise ValueError(f"max_candidates must be >= 1, got {self.max_candidates}")
        if self.max_samples < 1:
            raise ValueError(f"max_samples must be >= 1, got {self.max_samples}")
        if self.epochs_per_candidate < 1:
            raise ValueError(f"epochs_per_candidate must be >= 1, got {self.epochs_per_candidate}")

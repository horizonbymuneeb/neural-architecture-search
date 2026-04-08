"""Results of a search run, and the comparison between two of them.

A :class:`SearchOutcome` bundles the candidate records, the Pareto front, the
per-generation best accuracy and the spend summary. Keeping them together is
what lets :func:`compare_strategies` refuse to declare a winner when the two
runs did not spend the same budget: a comparison of unequal budgets is a
measurement of the budget, not of the search.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from .budget import CandidateRecord, BudgetSummary, compare_spend, spend_gap
from .pareto import ParetoFront


@dataclass(frozen=True)
class SearchOutcome:
    """Everything one search run produced."""

    strategy: str
    records: tuple[CandidateRecord, ...]
    front: ParetoFront
    history: tuple[float, ...]
    budget: BudgetSummary

    @property
    def n_evaluated(self) -> int:
        return len(self.records)

    @property
    def best(self) -> CandidateRecord | None:
        if not self.records:
            return None
        return max(self.records, key=lambda record: (record.accuracy, -record.parameters))

    @property
    def best_accuracy(self) -> float:
        best = self.best
        return best.accuracy if best is not None else 0.0

    @property
    def accuracy_curve(self) -> tuple[float, ...]:
        """Best-so-far accuracy after each evaluation."""
        curve: list[float] = []
        running = 0.0
        for record in self.records:
            running = max(running, record.accuracy)
            curve.append(running)
        return tuple(curve)

    def top(self, n: int = 3) -> list[CandidateRecord]:
        """The ``n`` most accurate candidates."""
        return sorted(self.records, key=lambda record: record.accuracy, reverse=True)[:n]

    def smallest_above(self, target: float) -> CandidateRecord | None:
        """Cheapest candidate that reaches ``target`` accuracy."""
        candidates = [record for record in self.records if record.accuracy >= target]
        if not candidates:
            return None
        return min(candidates, key=lambda record: record.parameters)

    def as_dict(self) -> dict[str, object]:
        return {
            "strategy": self.strategy,
            "budget": self.budget.as_dict(),
            "best": self.best.as_row() if self.best else None,
            "pareto": self.front.to_rows(),
            "history": [round(value, 4) for value in self.history],
            "candidates": [record.as_row() for record in self.records],
        }


@dataclass(frozen=True)
class StrategyComparison:
    """Side-by-side result of two runs, with a fairness verdict."""

    left: SearchOutcome
    right: SearchOutcome
    fair: bool
    gap: tuple[int, int]

    @property
    def winner(self) -> str:
        """Strategy with the higher best accuracy; ``tie`` when equal."""
        if not self.fair:
            return "incomparable"
        if self.left.best_accuracy > self.right.best_accuracy:
            return self.left.strategy
        if self.right.best_accuracy > self.left.best_accuracy:
            return self.right.strategy
        return "tie"

    @property
    def margin(self) -> float:
        """Accuracy difference between the two best candidates."""
        return self.left.best_accuracy - self.right.best_accuracy

    def rows(self) -> list[dict[str, float | int | str]]:
        rows: list[dict[str, float | int | str]] = []
        for outcome in (self.left, self.right):
            rows.append(
                {
                    "strategy": outcome.strategy,
                    "candidates": outcome.n_evaluated,
                    "samples": outcome.budget.total_samples,
                    "seconds": round(outcome.budget.total_seconds, 2),
                    "best_accuracy": round(outcome.best_accuracy, 4),
                    "mean_accuracy": round(outcome.budget.mean_accuracy, 4),
                    "pareto_size": outcome.front.size,
                    "best_params": outcome.best.parameters if outcome.best else 0,
                }
            )
        return rows

"""Pareto front over accuracy and parameter count.

Search returns a population, not a single winner, because "best" depends on the
deployment: a 2k-parameter network that is two points less accurate is usually
the right answer on a microcontroller. A point is kept only when no other point
beats it on both axes, which is the definition of Pareto dominance implemented
here. The front is also where the evolutionary search gets its diversity signal
from, so it is computed during the run rather than at the end.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ParetoPoint:
    """One candidate on the accuracy/parameter plane."""

    arch_key: str
    accuracy: float
    parameters: int

    def __post_init__(self) -> None:
        if not 0.0 <= self.accuracy <= 1.0:
            raise ValueError(f"accuracy must be in [0, 1], got {self.accuracy}")
        if self.parameters < 0:
            raise ValueError(f"parameters must be >= 0, got {self.parameters}")

    def as_row(self) -> dict[str, float | str | int]:
        return {
            "arch": self.arch_key,
            "accuracy": round(self.accuracy, 4),
            "parameters": self.parameters,
        }


def dominates(left: ParetoPoint, right: ParetoPoint) -> bool:
    """True when ``left`` is at least as good on both axes and better on one."""
    at_least_as_good = left.accuracy >= right.accuracy and left.parameters <= right.parameters
    strictly_better = left.accuracy > right.accuracy or left.parameters < right.parameters
    return at_least_as_good and strictly_better

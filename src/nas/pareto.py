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


def non_dominated(points: list[ParetoPoint]) -> list[ParetoPoint]:
    """The subset of ``points`` that no other point dominates, ordered by size."""
    kept: list[ParetoPoint] = []
    for candidate in points:
        if any(dominates(other, candidate) for other in points if other is not candidate):
            continue
        if any(
            existing.arch_key == candidate.arch_key
            and existing.accuracy == candidate.accuracy
            and existing.parameters == candidate.parameters
            for existing in kept
        ):
            continue
        kept.append(candidate)
    return sorted(kept, key=lambda point: (point.parameters, -point.accuracy))


@dataclass
class ParetoFront:
    """Incremental tracker: add points as they are evaluated."""

    points: list[ParetoPoint] = field(default_factory=list)

    @property
    def size(self) -> int:
        return len(self.points)

    def add(self, point: ParetoPoint) -> bool:
        """Insert a point; returns True when the front changed."""
        if any(dominates(existing, point) for existing in self.points):
            return False
        # a point on the same objective coordinates does not change the front,
        # whatever its architecture key is
        if any(
            existing.accuracy == point.accuracy and existing.parameters == point.parameters
            for existing in self.points
        ):
            return False
        survivors = [existing for existing in self.points if not dominates(point, existing)]
        changed = len(survivors) != len(self.points) or point not in self.points
        self.points = sorted(survivors + [point], key=lambda item: (item.parameters, -item.accuracy))
        return changed

    def front(self) -> list[ParetoPoint]:
        """Current front, smallest network first."""
        return list(self.points)

    def best_accuracy(self) -> ParetoPoint | None:
        """Most accurate point on the front (ties broken by fewer parameters)."""
        if not self.points:
            return None
        return max(self.points, key=lambda point: (point.accuracy, -point.parameters))

    def smallest(self) -> ParetoPoint | None:
        """Smallest network on the front."""
        if not self.points:
            return None
        return min(self.points, key=lambda point: (point.parameters, -point.accuracy))

    def covers_accuracy(self, target: float) -> bool:
        """Whether some point on the front reaches ``target`` accuracy."""
        return any(point.accuracy >= target for point in self.points)

    def hypervolume(self, reference_accuracy: float = 0.0, reference_parameters: int | None = None) -> float:
        """Area dominated by the front above a reference point.

        With the reference at ``(reference_accuracy, reference_parameters)`` the
        area is measured in accuracy x parameters, which is the standard
        two-objective indicator: bigger means a better accuracy/size trade-off.
        """
        if not self.points:
            return 0.0
        max_parameters = reference_parameters if reference_parameters is not None else max(
            point.parameters for point in self.points
        )
        area = 0.0
        best_so_far = reference_accuracy
        # sweep from the smallest network to the largest: each point contributes
        # the accuracy it adds over everything smaller, across the width it saves
        for point in sorted(self.points, key=lambda item: item.parameters):
            height = point.accuracy - best_so_far
            width = max(0, max_parameters - point.parameters)
            if height > 0 and width > 0:
                area += height * width
            best_so_far = max(best_so_far, point.accuracy)
        return float(area)

    def to_rows(self) -> list[dict[str, float | str | int]]:
        return [point.as_row() for point in self.points]

    def merge(self, other: "ParetoFront") -> "ParetoFront":
        """A new front holding both sets of points."""
        merged = ParetoFront()
        for point in self.points + other.points:
            merged.add(point)
        return merged


def front_from_records(records: list[tuple[str, float, int]]) -> ParetoFront:
    """Build a front from ``(arch_key, accuracy, parameters)`` triples."""
    front = ParetoFront()
    for arch_key, accuracy, parameters in records:
        front.add(ParetoPoint(arch_key=arch_key, accuracy=accuracy, parameters=parameters))
    return front

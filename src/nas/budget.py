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


@dataclass(frozen=True)
class CandidateRecord:
    """One evaluated candidate and what it cost."""

    index: int
    strategy: str
    arch_key: str
    accuracy: float
    parameters: int
    samples: int
    updates: int
    seconds: float

    def __post_init__(self) -> None:
        if self.index < 0:
            raise ValueError("index must be >= 0")
        if not 0.0 <= self.accuracy <= 1.0:
            raise ValueError(f"accuracy must be in [0, 1], got {self.accuracy}")
        if self.parameters < 0:
            raise ValueError("parameters must be >= 0")

    @property
    def accuracy_per_parameter(self) -> float:
        """Accuracy per thousand parameters; 0 when the network is parameter-free."""
        if self.parameters == 0:
            return 0.0
        return self.accuracy / (self.parameters / 1000.0)

    def as_row(self) -> dict[str, float | str | int]:
        return {
            "index": self.index,
            "strategy": self.strategy,
            "arch": self.arch_key,
            "accuracy": round(self.accuracy, 4),
            "parameters": self.parameters,
            "samples": self.samples,
            "updates": self.updates,
            "seconds": round(self.seconds, 3),
        }


@dataclass(frozen=True)
class BudgetSummary:
    """Aggregate spend of one search run."""

    strategy: str
    n_candidates: int
    total_samples: int
    total_updates: int
    total_seconds: float
    mean_accuracy: float
    best_accuracy: float
    limits: BudgetLimits

    def as_dict(self) -> dict[str, float | int | str]:
        return {
            "strategy": self.strategy,
            "candidates": self.n_candidates,
            "samples": self.total_samples,
            "updates": self.total_updates,
            "seconds": round(self.total_seconds, 3),
            "mean_accuracy": round(self.mean_accuracy, 4),
            "best_accuracy": round(self.best_accuracy, 4),
        }

    def same_spend_as(self, other: "BudgetSummary") -> bool:
        """True when two runs consumed the same number of candidates and samples."""
        return self.n_candidates == other.n_candidates and self.total_samples == other.total_samples


@dataclass
class BudgetAccountant:
    """Records every candidate evaluated under one budget."""

    limits: BudgetLimits = field(default_factory=BudgetLimits)
    strategy: str = "unnamed"
    records: list[CandidateRecord] = field(default_factory=list)

    @property
    def spent_candidates(self) -> int:
        return len(self.records)

    @property
    def spent_samples(self) -> int:
        return sum(record.samples for record in self.records)

    @property
    def spent_updates(self) -> int:
        return sum(record.updates for record in self.records)

    @property
    def spent_seconds(self) -> float:
        return sum(record.seconds for record in self.records)

    @property
    def remaining(self) -> int:
        """Candidate slots left by count.

        The per-candidate sample cost is only known when a candidate is
        evaluated, so this is a count of slots; :meth:`can_afford` is the check
        that also respects the sample ceiling.
        """
        return max(0, self.limits.max_candidates - self.spent_candidates)

    @property
    def remaining_samples(self) -> int:
        """Samples left before the ceiling is hit."""
        return max(0, self.limits.max_samples - self.spent_samples)

    @property
    def is_exhausted(self) -> bool:
        return self.remaining == 0

    def can_afford(self, samples: int) -> bool:
        """Whether one more candidate of ``samples`` rows fits in the budget."""
        if self.is_exhausted:
            return False
        return self.spent_samples + samples <= self.limits.max_samples

    def record(
        self,
        arch_key: str,
        accuracy: float,
        parameters: int,
        samples: int,
        updates: int,
        seconds: float,
    ) -> CandidateRecord:
        """Append one evaluation, refusing to exceed the budget."""
        if not self.can_afford(samples):
            raise RuntimeError(
                f"budget exhausted: {self.spent_candidates} candidates, {self.spent_samples} samples spent"
            )
        entry = CandidateRecord(
            index=self.spent_candidates,
            strategy=self.strategy,
            arch_key=arch_key,
            accuracy=float(accuracy),
            parameters=int(parameters),
            samples=int(samples),
            updates=int(updates),
            seconds=float(seconds),
        )
        self.records.append(entry)
        return entry

    def best(self) -> CandidateRecord | None:
        """Highest-accuracy record, or ``None`` when nothing was evaluated."""
        if not self.records:
            return None
        return max(self.records, key=lambda record: (record.accuracy, -record.parameters))

    def summary(self) -> BudgetSummary:
        """Aggregate the records; safe to call before any candidate ran."""
        accuracies = [record.accuracy for record in self.records]
        return BudgetSummary(
            strategy=self.strategy,
            n_candidates=self.spent_candidates,
            total_samples=self.spent_samples,
            total_updates=self.spent_updates,
            total_seconds=self.spent_seconds,
            mean_accuracy=float(sum(accuracies) / len(accuracies)) if accuracies else 0.0,
            best_accuracy=max(accuracies) if accuracies else 0.0,
            limits=self.limits,
        )

    def rows(self) -> list[dict[str, float | str | int]]:
        return [record.as_row() for record in self.records]

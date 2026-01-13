"""Budget accounting and the Pareto front."""

from __future__ import annotations

import pytest

from nas.budget import (
    BudgetAccountant,
    BudgetLimits,
    BudgetSummary,
    CandidateRecord,
    compare_spend,
    spend_gap,
)
from nas.pareto import ParetoFront, ParetoPoint, dominates, front_from_records, non_dominated


def record(index: int = 0, accuracy: float = 0.5, parameters: int = 1000) -> CandidateRecord:
    return CandidateRecord(
        index=index,
        strategy="unit",
        arch_key=f"arch-{index}",
        accuracy=accuracy,
        parameters=parameters,
        samples=100,
        updates=5,
        seconds=0.25,
    )


def test_candidate_record_validates_its_fields() -> None:
    with pytest.raises(ValueError, match="accuracy"):
        record(accuracy=1.5)
    with pytest.raises(ValueError, match="parameters"):
        record(parameters=-1)
    with pytest.raises(ValueError, match="index"):
        record(index=-1)


def test_candidate_record_reports_efficiency_and_rows() -> None:
    entry = record(accuracy=0.8, parameters=2000)
    assert entry.accuracy_per_parameter == pytest.approx(0.4)
    assert record(accuracy=0.8, parameters=0).accuracy_per_parameter == 0.0
    row = entry.as_row()
    assert row["arch"] == "arch-0"
    assert row["accuracy"] == 0.8


def test_budget_limits_validation() -> None:
    with pytest.raises(ValueError, match="max_candidates"):
        BudgetLimits(max_candidates=0)
    with pytest.raises(ValueError, match="max_samples"):
        BudgetLimits(max_samples=0)
    with pytest.raises(ValueError, match="epochs_per_candidate"):
        BudgetLimits(epochs_per_candidate=0)


def test_accountant_records_and_aggregates() -> None:
    accountant = BudgetAccountant(limits=BudgetLimits(max_candidates=3, max_samples=10_000), strategy="unit")
    assert accountant.remaining == 3
    assert accountant.summary().n_candidates == 0
    accountant.record("a", 0.4, 1000, samples=100, updates=5, seconds=0.1)
    accountant.record("b", 0.7, 2000, samples=200, updates=10, seconds=0.3)
    assert accountant.spent_candidates == 2
    assert accountant.spent_samples == 300
    assert accountant.spent_updates == 15
    assert accountant.spent_seconds == pytest.approx(0.4)
    assert accountant.remaining == 1
    summary = accountant.summary()
    assert summary.mean_accuracy == pytest.approx(0.55)
    assert summary.best_accuracy == pytest.approx(0.7)
    assert summary.n_candidates == 2
    assert accountant.best().arch_key == "b"
    assert len(accountant.rows()) == 2


def test_accountant_refuses_to_overspend() -> None:
    accountant = BudgetAccountant(limits=BudgetLimits(max_candidates=1, max_samples=1000), strategy="unit")
    accountant.record("a", 0.5, 100, samples=100, updates=1, seconds=0.1)
    assert accountant.is_exhausted
    assert not accountant.can_afford(100)
    with pytest.raises(RuntimeError, match="budget exhausted"):
        accountant.record("b", 0.5, 100, samples=100, updates=1, seconds=0.1)

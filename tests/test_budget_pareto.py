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


def test_accountant_respects_the_sample_ceiling() -> None:
    accountant = BudgetAccountant(limits=BudgetLimits(max_candidates=10, max_samples=150), strategy="unit")
    accountant.record("a", 0.5, 100, samples=100, updates=1, seconds=0.1)
    # slots remain by count, but the sample ceiling is what blocks the next one
    assert accountant.remaining == 9
    assert accountant.remaining_samples == 50
    assert not accountant.can_afford(100)
    assert accountant.can_afford(50)


def test_spend_comparison_helpers() -> None:
    left = BudgetSummary(
        strategy="a", n_candidates=3, total_samples=300, total_updates=9,
        total_seconds=1.0, mean_accuracy=0.5, best_accuracy=0.6, limits=BudgetLimits(),
    )
    right = BudgetSummary(
        strategy="b", n_candidates=3, total_samples=300, total_updates=9,
        total_seconds=2.0, mean_accuracy=0.55, best_accuracy=0.65, limits=BudgetLimits(),
    )
    assert compare_spend(left, right)
    assert spend_gap(left, right) == (0, 0)
    assert left.as_dict()["candidates"] == 3
    unequal = BudgetSummary(
        strategy="c", n_candidates=4, total_samples=400, total_updates=12,
        total_seconds=3.0, mean_accuracy=0.5, best_accuracy=0.6, limits=BudgetLimits(),
    )
    assert not compare_spend(left, unequal)
    assert spend_gap(left, unequal) == (1, 100)


def test_pareto_point_validation() -> None:
    with pytest.raises(ValueError, match="accuracy"):
        ParetoPoint(arch_key="a", accuracy=-0.1, parameters=10)
    with pytest.raises(ValueError, match="parameters"):
        ParetoPoint(arch_key="a", accuracy=0.5, parameters=-5)

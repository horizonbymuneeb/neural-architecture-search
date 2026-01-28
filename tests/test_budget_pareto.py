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


def test_dominance_is_strict_on_at_least_one_axis() -> None:
    strong = ParetoPoint(arch_key="a", accuracy=0.8, parameters=1000)
    weak = ParetoPoint(arch_key="b", accuracy=0.7, parameters=2000)
    tradeoff = ParetoPoint(arch_key="c", accuracy=0.9, parameters=3000)
    same = ParetoPoint(arch_key="d", accuracy=0.8, parameters=1000)
    assert dominates(strong, weak)
    assert not dominates(weak, strong)
    assert not dominates(strong, tradeoff)
    assert not dominates(strong, same)
    assert dominates(strong, ParetoPoint(arch_key="e", accuracy=0.8, parameters=1500))


def test_non_dominated_keeps_the_tradeoff_points_only() -> None:
    points = [
        ParetoPoint(arch_key="a", accuracy=0.6, parameters=500),
        ParetoPoint(arch_key="b", accuracy=0.7, parameters=1500),
        ParetoPoint(arch_key="c", accuracy=0.5, parameters=800),
        ParetoPoint(arch_key="d", accuracy=0.75, parameters=2000),
    ]
    front = non_dominated(points)
    keys = [point.arch_key for point in front]
    assert keys == ["a", "b", "d"]
    assert "c" not in keys


def test_front_adds_and_prunes() -> None:
    front = ParetoFront()
    assert front.size == 0
    assert front.best_accuracy() is None
    assert front.smallest() is None
    assert front.hypervolume() == 0.0
    assert front.add(ParetoPoint(arch_key="a", accuracy=0.6, parameters=1000)) is True
    assert front.add(ParetoPoint(arch_key="b", accuracy=0.6, parameters=1000)) is False
    assert front.add(ParetoPoint(arch_key="c", accuracy=0.5, parameters=2000)) is False
    assert front.add(ParetoPoint(arch_key="d", accuracy=0.7, parameters=3000)) is True
    assert front.size == 2
    assert [point.arch_key for point in front.front()] == ["a", "d"]
    # a point that dominates an existing one removes it
    assert front.add(ParetoPoint(arch_key="e", accuracy=0.65, parameters=900)) is True
    assert [point.arch_key for point in front.front()] == ["e", "d"]
    assert front.smallest().arch_key == "e"
    assert front.best_accuracy().arch_key == "d"
    assert front.covers_accuracy(0.66) is True
    assert front.covers_accuracy(0.99) is False


def test_hypervolume_matches_a_hand_computed_area() -> None:
    front = ParetoFront()
    front.add(ParetoPoint(arch_key="small", accuracy=0.5, parameters=100))
    front.add(ParetoPoint(arch_key="large", accuracy=0.9, parameters=500))
    # reference point (accuracy 0, parameters 500):
    # small contributes (0.5 - 0) * (500 - 100) = 200
    # large contributes (0.9 - 0.5) * (500 - 500) = 0
    assert front.hypervolume(reference_accuracy=0.0, reference_parameters=500) == pytest.approx(200.0)


def test_front_merge_and_rows() -> None:
    left = ParetoFront()
    left.add(ParetoPoint(arch_key="a", accuracy=0.6, parameters=1000))
    right = ParetoFront()
    right.add(ParetoPoint(arch_key="b", accuracy=0.8, parameters=2000))
    merged = left.merge(right)
    assert merged.size == 2
    rows = merged.to_rows()
    assert rows[0]["arch"] == "a"
    assert front_from_records([("x", 0.5, 100), ("y", 0.6, 200)]).size == 2

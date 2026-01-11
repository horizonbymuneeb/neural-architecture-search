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

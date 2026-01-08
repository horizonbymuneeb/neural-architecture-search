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

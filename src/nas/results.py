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

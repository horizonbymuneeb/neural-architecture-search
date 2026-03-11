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

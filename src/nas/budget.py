"""Budget accounting, so two search strategies can be compared honestly.

A search that evaluates 200 candidates will beat one that evaluates 20 for
reasons that have nothing to do with the search algorithm. Every evaluation in
this package therefore goes through :class:`BudgetAccountant`, which records what
was spent — candidates, samples seen, optimiser updates and wall-clock seconds —
and refuses to start a candidate once the budget is exhausted. The comparison in
:mod:`nas.results` checks that both strategies spent the same budget before it
prints a winner.
"""

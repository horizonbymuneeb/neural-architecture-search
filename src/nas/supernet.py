"""Weight-sharing supernet.

The expensive part of architecture search is training a network per candidate.
The cheap approximation is to train one over-parameterised network that contains
every operation the space allows, and to evaluate a candidate by running the
forward pass that selects its operations. That is weight sharing: candidates are
not independent models, they are *views* of one model.

This module provides two builders over the same channel bookkeeping:

* :class:`Supernet` keeps one module per (block, edge, operation), so sampling an
  architecture only changes which modules are called, never the parameters.
* :func:`build_network` builds the standalone network for one architecture. It is
  what the final accuracy numbers are measured on, and its parameter count is
  what :func:`nas.search_space.estimate_parameters` predicts.

Keeping both in one file is deliberate: the two must agree about channel counts
or the search optimises something other than the model it reports.
"""

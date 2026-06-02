"""Training and evaluation for one candidate architecture.

Every candidate is trained for the same number of epochs on the same batches —
that is what makes the comparison between search strategies a comparison of the
*search*, not of the training budget. The trainer is deliberately plain: one
optimiser, cross-entropy, no schedules, no early stopping. Anything fancier would
add a knob whose setting could explain a difference between strategies.

Two entry points matter:

* :func:`train_arch` trains a candidate *through the supernet*, so gradients land
  in the shared weights and later candidates start from what earlier ones
  learned. That is the weight-sharing assumption, and the tests check that it
  actually happens.
* :func:`train_candidate` trains a standalone network, used for the final
  numbers that are reported.
"""

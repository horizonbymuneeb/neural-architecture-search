# neural-architecture-search

A small, complete implementation of cell-based architecture search that reports
what it actually spent. The interesting part of NAS is not the evolutionary loop
— it is the accounting: a search that evaluates 200 candidates will beat one that
evaluates 20 for reasons that have nothing to do with the algorithm. Here every
candidate goes through a budget accountant, the random-search baseline is given
exactly the same number of candidates and the same training recipe, and the
comparison refuses to declare a winner when the two runs spent different
budgets.

The task is a synthetic three-class image problem generated in memory, so a
candidate costs about a second on CPU and nothing is downloaded.

## What is in the box

| Module | Job |
| --- | --- |
| `src/nas/search_space.py` | The encoding: six operations per edge, two nodes per cell, four edges, a bounded block count. Sampling, mutation, crossover, key roundtrip, and the analytic parameter formula. |
| `src/nas/supernet.py` | Weight-sharing supernet holding every operation, plus `build_network` for the standalone network of one architecture. |
| `src/nas/data.py` | The synthetic task: three classes of 12x12 RGB images, deterministic from a seed, balanced by construction. |
| `src/nas/trainer.py` | One fixed training recipe for every candidate, training through the shared weights, and the evaluator that records each run into the accountant. |
| `src/nas/budget.py` | `CandidateRecord`, `BudgetAccountant`, `BudgetSummary`: what was spent, and refusal to overspend. |
| `src/nas/pareto.py` | Pareto dominance, an incremental front over accuracy vs parameters, and hypervolume. |
| `src/nas/evolution.py` | Tournament selection, mutation, crossover, elitism, duplicate suppression, and the search loop. |
| `src/nas/random_search.py` | The uniform-sampling baseline, with the same budget as the evolutionary run. |
| `src/nas/results.py` | `SearchOutcome`, the fair comparison between two outcomes, and the text/JSON formatting. |
| `src/nas/config.py` | Typed, validated view of `config.yaml`. |
| `src/nas/cli.py` | `search`, `space`, `data`, `config` commands. |
| `tests/` | 88 tests across the five modules above. |

## Quickstart

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt

# what is the search space, and what does a supernet cost?
make space

# check the task is learnable before trusting any search result
make data

# run both strategies under the same budget and compare them
make search          # PYTHONPATH=src python -m nas search --strategy both

# or one strategy, as JSON
PYTHONPATH=src python -m nas search --strategy evolution --json --out artifacts/evolution.json

make test            # 88 tests, about 14 s on CPU
```

`pip install -e .` also works and makes the `nas` console script available, so
`nas search --strategy both` runs without the `PYTHONPATH` prefix.

A full `make search` (16 candidates per strategy, four epochs each) takes about
90 seconds on a two-core box.

## Design notes

**The supernet and the reported model must agree about channels.** A candidate is
scored by running its sub-network inside the supernet, but the number that ends
up on the Pareto front is the parameter count of the *standalone* network built
for that architecture. If those two disagree the search optimises a model nobody
reports. `estimate_parameters` is checked against `count_parameters(build_network(arch))`
for every single-operation architecture and a cross-product of edge
combinations, and the tests fail if a single case drifts. The channel
bookkeeping is where this goes wrong in practice: block 0 reads the stem twice,
block 1 reads a cell output next to the stem, and later blocks read two cell
outputs, so the input channels of a cell are three different cases.

**Each cell input is projected to the base width first.** Without that
projection, a pooling or identity operation passes its input channels through
unchanged while a convolution returns `width` channels, and summing the two
outputs of a node fails — which is exactly the bug the full-space forward test
caught. Projecting both inputs with a 1x1 convolution makes every operation
`width -> width`, at the cost of two small matrices per cell, and that cost is in
the parameter estimate.

**Elites are not re-evaluated, and duplicates are not paid for twice.** An
individual that survives a generation keeps its accuracy; an offspring whose
architecture was already trained is discarded and replaced with a fresh mutation.
Both rules exist so that the reported number of evaluations equals the number of
distinct architectures actually trained, which is what the random baseline is
compared against.

**The budget is a first-class object, not a counter in a loop.** The accountant
records candidates, samples, optimiser updates and wall-clock seconds, and raises
when a run tries to exceed either the candidate ceiling or the sample ceiling.
`compare_strategies` reports `fair: false` and `winner: incomparable` rather than
printing a winner from unequal spend — a comparison of unequal budgets measures
the budget.

**Accuracy saturates on this task, and that is reported rather than hidden.** The
three-class synthetic problem is easy: many architectures reach 1.0 accuracy, so
the Pareto front is often decided by parameter count alone and the two strategies
tie. A harder task would separate them; the point of this repo is that the
accounting makes that visible instead of producing an impressive-looking number.

## Tests

88 tests, run with `python -m pytest -q` (about 14 s, single-threaded CPU).

- `tests/test_search_space.py` (21) — the vocabulary and topology are pinned,
  every invalid encoding raises, keys round-trip, sampling respects the space and
  is seed-reproducible, mutation with rate 1 changes every edge and rate 0 changes
  nothing, crossover only takes edges from its parents, and
  `estimate_parameters` equals the built network for 6 operation patterns × 3
  block counts plus a cross-product of edge pairs.
- `tests/test_supernet.py` (15) — each operation builds the expected module type,
  skip projects when channels change, forward pass shapes for 1–3 blocks and all
  six operations, the supernet rejects architectures it cannot represent, it holds
  more parameters than any single architecture, `arch_parameters` is a strict
  subset, two candidates really do share weight tensors, and training one
  candidate changes another candidate's output (that is weight sharing working).
- `tests/test_data_trainer.py` (16) — masks are binary and distinct, the task is
  deterministic and seed-sensitive, the split is complete and disjoint, loaders
  batch correctly, an untrained network scores at chance, four epochs reach over
  85% validation accuracy while the loss falls, and the evaluator records the
  right sample count and then raises when the budget is exhausted.
- `tests/test_budget_pareto.py` (13) — records validate their fields, the
  accountant aggregates and refuses to overspend on either ceiling, dominance is
  strict on at least one axis, the front prunes dominated points and de-duplicates
  identical coordinates, and hypervolume matches a hand-computed area.
- `tests/test_search_config_cli.py` (23) — tournament selection picks the best
  contender, populations are distinct, elites survive a generation, the search
  spends exactly the planned number of evaluations and is reproducible for a
  fixed seed, the baseline samples distinct architectures, the comparison flags
  unequal budgets and names a winner only when they match, config validation
  catches unknown keys and inconsistent budgets, and the CLI runs both strategies
  on a temporary config and writes JSON.

## Limitations

- The search space is small on purpose (6^4 operation combinations × block counts).
  Exhaustive enumeration would be feasible here; the repo is about the
  infrastructure around a search, not about beating a large benchmark.
- Weight sharing is not trained with a supernet-level objective (no DARTS-style
  bilevel optimisation, no path dropout). Candidates are trained sequentially
  through the shared weights, which favours architectures evaluated later — a
  known bias of naive weight sharing, and the reason the random baseline is given
  the same ordering budget rather than being run afterwards.
- Accuracy is measured on a synthetic task with 150 validation images, so a
  difference of a few points is inside the noise. No confidence intervals are
  reported.
- Training is a fixed recipe: no schedules, no early stopping, no augmentation.
  The final reported numbers are therefore a lower bound on what each
  architecture could do with a proper training budget.
- Only two objectives are tracked. Hypervolume is implemented for that 2-D case
  and would need a different algorithm for three or more.
- No GPU path and no distributed evaluation; candidates are trained one at a time
  on CPU.

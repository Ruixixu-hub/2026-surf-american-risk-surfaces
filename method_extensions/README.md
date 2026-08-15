# SURF Method Extensions

This directory is a separate, Codex-assisted research portfolio built on top
of the original SURF American-option workflow. It keeps successful methods,
negative results, incomplete neural experiments, code, tests, protocols, and
curated evidence in one auditable location.

The original Experiments 01--18 remain outside this directory and are not
relabelled as extension work.

## Navigation

- [`BENCHMARK_HIERARCHY.md`](BENCHMARK_HIERARCHY.md): formal benchmark roles.
- [`METHOD_STATUS.md`](METHOD_STATUS.md): current GO/STOP/INCOMPLETE decisions.
- [`benchmarks/`](benchmarks/): historical, strengthened, and accuracy
  benchmarks.
- [`reduced_order_methods/`](reduced_order_methods/): POD, RB-VI, localized
  bases, and the positive-premium basis operator.
- [`neural_methods/`](neural_methods/): pointwise MLP, PINN, and DeepONet work.
- [`experiments/`](experiments/): canonical Experiments 21--61.
- [`src/`](src/): source snapshot used by the extension experiments.
- [`tests/`](tests/): extension-specific tests.
- [`results/`](results/): curated formal evidence and decision files.
- [`reports/`](reports/): method reports with claim limits.
- [`portfolio_summary/`](portfolio_summary/): cross-method comparisons.

Canonical files are stored once under `experiments/`, `src/`, `results/`, and
`reports/`. Method landing pages link to those files instead of duplicating
them.

## Evidence policy

- Formal Mac results are included for Policy Iteration, numerical-reference
  audits, POD diagnostics, RB-VI, boundary/localized bases, the basis operator,
  and Projected LU.
- PINN and DeepONet contain implementation code and protocols but are marked
  **NO FORMAL RESULTS YET**. Tiny smoke runs are not published as performance
  evidence.
- Large regenerable snapshots and neural checkpoints are excluded. Their
  manifests, hashes, configuration, metrics, and regeneration commands are
  retained.

See [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) before running experiments.

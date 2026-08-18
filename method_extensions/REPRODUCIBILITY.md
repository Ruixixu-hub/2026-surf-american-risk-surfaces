# Reproducibility

Run commands from `method_extensions/` unless stated otherwise.

## Environment

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-extensions.txt
```

Install PyTorch separately using the platform-appropriate official wheel. The
formal PINN and DeepONet plans require Windows CUDA; those runs have not yet
produced formal results.

## Source path

```bash
export PYTHONPATH="$PWD/src"
```

## Frozen original inputs

The small Stage-4 regime and split CSVs used by the unified solver comparison
are included under `results/04_surrogate_dataset/v1_small_grid/`. The 48 MB
dataset bundle, Stage-5 model artifacts, RB snapshots, and neural checkpoints
remain excluded. Before regenerating artifact-heavy stages, copy those files
from the repository root:

```bash
mkdir -p results/04_surrogate_dataset results/05_surrogate_models
cp ../results/04_surrogate_dataset/v1_small_grid/dataset_v1_small_grid.npz \
  results/04_surrogate_dataset/v1_small_grid/
cp -R ../results/05_surrogate_models/price_premium results/05_surrogate_models/
```

The 202 train-only RB snapshots and neural checkpoints are intentionally not
versioned. Regenerate RB snapshots with Experiment 36 before running the
artifact-dependent RB, basis-operator, or DeepONet stages.

## Tests

Core strict-solver tests do not require regenerated learning artifacts:

```bash
python -m pytest -q \
  tests/test_policy_iteration.py \
  tests/test_projected_lu.py \
  tests/test_penalty_newton.py \
  tests/test_poster_unified_study.py \
  tests/test_reduced_basis_vi.py
```

Run the full extension suite after regenerating the train-only snapshots:

```bash
python -m pytest -q tests
```

## Registered experiment order

- 21--28: common protocol, Policy/warm-start, Greek reference, POD diagnostics.
- 29--35: PINN Arms C/D/E; no formal results yet.
- 36--41: primal/dual RB-VI.
- 42--45: boundary-aligned/localized basis falsification.
- 46--51: positive-premium basis operator.
- 52--57: positive-premium DeepONet; no formal results yet.
- 58--61: Projected LU validation and held-out benchmark.
- 62--64: unified poster strict-solver comparison, numerical-reference audit,
  and evidence synthesis.
- 68--72: published in 't Hout DIRK-P protocol, paper-case reproduction,
  12-regime high-accuracy audit, decision synthesis, and the documented
  correction restoring the project-wide frozen `1e-12` VI acceptance gate.

Do not rerun or tune held-out experiments after reading their formal scores.

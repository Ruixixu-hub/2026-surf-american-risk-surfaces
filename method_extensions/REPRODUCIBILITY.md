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

The extension protocols depend on the original repository's frozen Stage-4
dataset and Stage-5 surrogate manifests. They are not duplicated here. Before
regenerating artifact-heavy stages, copy them from the repository root:

```bash
mkdir -p results/04_surrogate_dataset results/05_surrogate_models
cp -R ../results/04_surrogate_dataset/v1_small_grid results/04_surrogate_dataset/
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

Do not rerun or tune held-out experiments after reading their formal scores.

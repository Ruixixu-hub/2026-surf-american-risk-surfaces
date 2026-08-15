# 2026-surf-american-risk-surfaces

SURF 2026 computational-finance project on free-boundary-aware American-option
risk surfaces. The original workflow uses Crank--Nicolson with PSOR, validates
the obstacle/LCP formulation, and studies prices, exercise boundaries, and
Greeks.

## Repository sections

- `experiments/01`--`18`, `src/`, `results/01`--`06`, and the corresponding
  reports contain the original project workflow.
- [`method_extensions/`](method_extensions/) is the separate Codex-assisted
  research portfolio for strengthened benchmarks, numerical references,
  reduced-order methods, PINNs, and operator learning.

## Benchmark hierarchy

1. Historical benchmark: **CN + PSOR**.
2. Strengthened benchmark 1: **CN + Policy Iteration**.
3. Strengthened benchmark 2: **CN + Projected LU / Brennan--Schwartz**.
4. High-accuracy numerical reference: **DIRK/related time integrators + Policy
   Iteration + a strike-concentrated nonuniform grid**. This is an accuracy
   reference, not the main online-speed competitor.

European-option formulas remain validation controls. The main research target
and all three speed benchmarks are American-option problems.

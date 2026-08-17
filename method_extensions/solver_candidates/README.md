# Strict-Solver Candidates

This directory records candidate LCP solvers that were evaluated under the
common strict-solver protocol but were not promoted to benchmark status.

- [`01_penalty_newton/`](01_penalty_newton/): finite-penalty semismooth Newton;
  failed the common correctness gate in the unified 67-regime experiment.

A method is not promoted because it is fast in isolation. It must first pass
the frozen obstacle, equation, complementarity, trajectory-match, and boundary
checks.

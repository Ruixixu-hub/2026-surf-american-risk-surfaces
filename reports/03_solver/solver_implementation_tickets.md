# Solver Implementation Tickets Roadmap

> **For agentic workers:** implement this roadmap ticket by ticket only. Do not implement all tickets in one pass. After each ticket, run the relevant tests, show `git status --short`, and stop for human review before moving on.

**Goal:** Convert the solver validation requirements into a concrete, reviewable coding roadmap for a future CN/PSOR solver implementation.

**Architecture:** The future solver should be built in small layers: analytic utilities, grid/operator setup, European Crank-Nicolson validation, American PSOR/LCP logic, diagnostics, boundary extraction, Greeks, refinement/domain checks, and final report assembly. This file is not code and does not validate or implement the solver.

**Tech Stack:** Future implementation is expected to use Python, tests under `tests/`, source modules under `src/`, validation scripts under `experiments/`, and generated validation artifacts under `results/`, but all Python paths are tentative until Ticket 0 confirms the repository structure.

---

## Purpose

This file is an implementation-ticket roadmap, not solver code. It translates the validation requirements in `reports/03_solver/solver_validation_plan.md` and the project PDFs into ordered coding stages that a future Codex session can implement safely one ticket at a time.

The roadmap exists to prevent a future implementation from jumping directly into a full CN/PSOR solver, stress maps, datasets, or neural networks. Each ticket is intentionally small, testable, and reviewable. The solver should be trusted only after the validation ladder has been completed and reviewed by humans.

## Source Abbreviations

- PR: `reports/00_planning/planning_report.md`
- LM: `reports/01_literature/literature_map.md`
- FN: `reports/02_math/formulation_note.md`
- SVP: `reports/03_solver/solver_validation_plan.md`
- SH: `docs/student_handout_full_report.pdf`
- MG: `docs/Student_Methodology_FreeBoundary_Risk_Surfaces.pdf`

## Rules for Future Coding Sessions

- Codex must not implement all tickets at once.
- Each ticket must be implemented in a separate step.
- After each ticket, run the relevant tests and show `git status --short`.
- Humans should review each ticket before moving on.
- Do not move to stress maps, datasets, or neural networks until solver validation is accepted.
- Keep strike normalized to `K = 1` for first validation work unless a later reviewed document changes that default.
- Treat `Smax = 4K`, target region `S/K in [0.4, 1.8]`, and coarse/medium/fine grids as tentative validation defaults to be confirmed by evidence.

## Ordered Implementation Tickets

### Ticket 0: Repo/Test Setup Check

**Objective:** Confirm package layout, test command, dependency availability, and artifact locations before solver code.

**Source documents to read first:** PR stages and gates; SVP sections 14 and 19; SH reproducibility notes.

**Files likely to be created or modified later:** `pyproject.toml`, `src/`, `tests/`, `experiments/`, all tentative.

**Functions/classes likely to be needed:** None. This ticket defines test/import conventions only.

**Tests required:** Run existing test discovery if present. Compile existing Python if present. If there is no Python source yet, record that no code exists yet.

**Expected output:** A short setup note with the chosen test command, import convention, dependency status, and tentative file layout.

**Pass/fail criteria:** Pass if future imports and tests have a clear path. Fail if package layout, dependencies, or artifact locations remain ambiguous.

**Possible failure modes:** Hidden prior code, missing scientific Python stack, unclear package namespace, or a test command that writes unexpected tracked files.

**Human review needed before moving on:** Yes.

**Allowed to write code later:** No solver code. Setup-only changes are allowed after review.

### Ticket 1: Payoff Functions and European Black-Scholes Closed-Form Utilities

**Objective:** Build trusted scalar/vector payoff and European closed-form utilities before finite differences.

**Source documents to read first:** FN sections 2-3; SVP sections 3-4; MG formulation.

**Files likely to be created or modified later:** `src/solvers/black_scholes.py`, `tests/test_black_scholes.py`.

**Functions/classes likely to be needed:** Payoff function for puts and calls; European call/put closed-form utility with continuous dividends; a small parameter container if useful.

**Tests required:** Payoff below, at, and above strike; maturity equals payoff; put-call parity with dividends; known European price fixtures; scalar and array input behavior.

**Expected output:** Reusable payoff and European pricing utilities that later validation tickets can trust.

**Pass/fail criteria:** Pass if utilities match analytic identities and tolerate the intended scalar/vector inputs. Fail if the dividend term, call/put signs, maturity behavior, or array shapes are wrong.

**Possible failure modes:** Wrong `q` discounting, wrong `tau` handling, instability at `sigma = 0` or `tau = 0`, option-type mixups, or shape mismatch.

**Human review needed before moving on:** Yes.

**Allowed to write code later:** Yes.

### Ticket 2: Finite-Difference Grid and Black-Scholes Operator Setup

**Objective:** Build `S` and `tau` grids, interior indexing, finite-difference operator coefficients, and boundary-value helpers.

**Source documents to read first:** FN sections 5-7; SVP sections 3 and 9; MG sections 5.2-5.5.

**Files likely to be created or modified later:** `src/solvers/grid.py`, `src/solvers/operator.py`, tests under `tests/`.

**Functions/classes likely to be needed:** Spot/time grid builder; interior-node index helper; Black-Scholes operator coefficient builder; put/call boundary condition evaluator.

**Tests required:** Endpoint and index checks; interior matrix or coefficient shape checks; manufactured derivative checks for `U_S` and `U_SS`; put and dividend-call boundary value checks.

**Expected output:** Deterministic grid/operator objects usable by the European Crank-Nicolson validation mode.

**Pass/fail criteria:** Pass if coefficients and boundaries match the formulation and interior indexing is explicit. Fail if indexing, drift sign, or boundary adjustments are unclear.

**Possible failure modes:** Off-by-one interior nodes, wrong `(r - q)` drift sign, missing boundary right-hand-side terms, inconsistent `Smax`, or inconsistent `Delta S`/`Delta tau`.

**Human review needed before moving on:** Yes.

**Allowed to write code later:** Yes.

### Ticket 3: European Crank-Nicolson Validation Mode

**Objective:** Implement Crank-Nicolson continuation without American projection and compare against closed-form European prices.

**Source documents to read first:** SVP section 4; FN section 11; MG validation ladder.

**Files likely to be created or modified later:** `src/solvers/cn.py`, `experiments/01_solver_validation.py`, tests under `tests/`.

**Functions/classes likely to be needed:** European CN solver; tridiagonal linear solve helper if the implementation does not use a library solve; target-region error metrics over `S/K in [0.4, 1.8]`.

**Tests required:** European put and call checks for `q = 0` and `q > 0`; medium/fine refinement showing error improvement; separate reporting if boundary or payoff-kink regions dominate errors.

**Expected output:** European error table with option type, parameters, grid size, maximum absolute error, RMSE, and maximum-error location.

**Pass/fail criteria:** Pass if closed-form errors are small in the target region and improve under refinement. Fail if ordinary interior errors are large, unstable, or do not improve.

**Possible failure modes:** Time marched in the wrong direction, bad boundary adjustment, wrong closed-form comparison, wrong dividend treatment, or an overly coarse domain/grid.

**Human review needed before moving on:** Yes, before PSOR work begins.

**Allowed to write code later:** Yes.

### Ticket 4: PSOR/LCP Core for American Option Obstacle

**Objective:** Add projected SOR LCP solving for one CN time step and then full American time marching.

**Source documents to read first:** FN section 6; SVP sections 3 and 10; MG sections 5.4 and 5.8.

**Files likely to be created or modified later:** `src/solvers/cn_psor.py`, `tests/test_psor.py`.

**Functions/classes likely to be needed:** PSOR LCP solver; American CN/PSOR solver; convergence metadata container with iteration counts, tolerance, and status flags.

**Tests required:** Toy LCP projection test; American obstacle never violated beyond tolerance; convergence/failure metadata is exposed; nonconvergence is reported rather than hidden.

**Expected output:** American solver result containing the value surface, PSOR iteration counts, tolerances, and convergence status.

**Pass/fail criteria:** Pass if projection is enforced and convergence behavior is explicit. Fail if values fall below payoff, if convergence failures are silent, or if PSOR is applied inconsistently.

**Possible failure modes:** Stopping too early, relaxation instability, stale iterate use, projection not applied at every interior node, wrong row update order, or wrong payoff vector.

**Human review needed before moving on:** Yes.

**Allowed to write code later:** Yes.

### Ticket 5: American Put Validation

**Objective:** Validate the first genuinely American early-exercise case.

**Source documents to read first:** SVP section 6; FN sections 7-9; MG qualitative checks.

**Files likely to be created or modified later:** Validation experiment/test files and result tables, likely under `experiments/`, `tests/`, and `results/01_solver_validation/`.

**Functions/classes likely to be needed:** American put validation runner; representative spot sampler; continuation premium calculator.

**Tests required:** Obstacle violation near tolerance; selected prices at `S/K = 0.8, 1.0, 1.2`; low-spot exercise sanity; preliminary boundary/premium checks.

**Expected output:** American put validation table, obstacle summary, selected price values, and provisional boundary/premium diagnostics.

**Pass/fail criteria:** Pass if put value dominates payoff, continuation premium is nonnegative up to tolerance, and the boundary is financially plausible. Fail if the exercise region is reversed, if the boundary is above strike without explanation, or if obstacle violations grow.

**Possible failure modes:** Wrong payoff, wrong lower boundary, missing projection, complementarity not solved even though `U >= payoff`, coarse-grid artifact, or boundary extraction reversed.

**Human review needed before moving on:** Yes.

**Allowed to write code later:** Yes.

### Ticket 6: No-Dividend American Call Validation

**Objective:** Confirm `q = 0` American call matches the European call and has no genuine early exercise.

**Source documents to read first:** SVP section 5; FN section 7; LM no-dividend call theorem notes.

**Files likely to be created or modified later:** Call validation tests and experiment outputs, likely under `tests/`, `experiments/`, and `results/01_solver_validation/`.

**Functions/classes likely to be needed:** American-versus-European call comparison; no-boundary diagnostic for the `q = 0` control.

**Tests required:** Maximum American-European call difference in the target region; obstacle check; boundary diagnostic showing no stable early-exercise boundary before maturity.

**Expected output:** No-dividend American call comparison table with maximum difference, grid size, obstacle violation, and boundary-detection status.

**Pass/fail criteria:** Pass if the American and European calls agree within numerical tolerance and the difference shrinks or remains negligible under refinement. Fail if the solver invents stable early exercise.

**Possible failure modes:** Wrong call payoff, wrong call upper boundary, dividend accidentally nonzero, upper boundary distorting the interior, or boundary threshold too aggressive.

**Human review needed before moving on:** Yes.

**Allowed to write code later:** Yes.

### Ticket 7: Dividend-Paying American Call Validation

**Objective:** Validate high-spot early-exercise behavior for positive dividends.

**Source documents to read first:** SVP section 7; FN section 7; MG boundary conditions.

**Files likely to be created or modified later:** Dividend-call validation tests/results, likely under `tests/`, `experiments/`, and `results/01_solver_validation/`.

**Functions/classes likely to be needed:** Dividend-call validation runner; `q` sweep helper; high-spot boundary sanity checker.

**Tests required:** `q = 0` control; positive-dividend cases such as `q = 0.03, 0.06, 0.10`; high-spot boundary sanity; preliminary `Smax` sensitivity check.

**Expected output:** American-versus-European call differences by `q`, dividend-call boundary curves, obstacle summaries, and complementarity summaries.

**Pass/fail criteria:** Pass if `q = 0` remains a no-exercise control and positive `q` behavior is financially plausible and stable. Fail if `Smax` or the upper boundary creates artificial high-spot exercise.

**Possible failure modes:** Wrong `(r - q)` sign, no-dividend and dividend logic mixed together, put-style boundary extraction applied to calls, or upper-boundary distortion.

**Human review needed before moving on:** Yes.

**Allowed to write code later:** Yes.

### Ticket 8: Obstacle and Complementarity Diagnostics

**Objective:** Formalize diagnostics required in every American validation case.

**Source documents to read first:** SVP section 10; FN section 11; MG required diagnostics.

**Files likely to be created or modified later:** `src/diagnostics/obstacle.py`, `src/diagnostics/complementarity.py`, tests under `tests/`.

**Functions/classes likely to be needed:** Obstacle violation; equation gap; complementarity residual; residual summary with maximum, location, and optional percentiles.

**Tests required:** Synthetic exact exercise case; synthetic continuation case; boundary nodes excluded from LCP residual; regression use in Tickets 5-7.

**Expected output:** Reusable diagnostics that distinguish obstacle violations from complementarity failures.

**Pass/fail criteria:** Pass if diagnostics correctly identify value-gap and equation-gap problems and keep boundary nodes separate from solved interior nodes. Fail if boundary conditions pollute interior residuals or if maximum residuals are hidden.

**Possible failure modes:** Using the wrong right-hand side time level, multiplying incompatible shapes, including imposed boundary nodes in the LCP residual, or summarizing only averages.

**Human review needed before moving on:** Yes.

**Allowed to write code later:** Yes.

### Ticket 9: Continuation Premium and Boundary Extraction

**Objective:** Extract put and dividend-call boundaries from `U - payoff`, not from visual plots.

**Source documents to read first:** SVP section 11; FN sections 8-9; MG section 5.9.

**Files likely to be created or modified later:** `src/boundaries/extraction.py`, tests under `tests/`.

**Functions/classes likely to be needed:** Continuation premium calculator; exercise mask; put boundary extractor; call boundary extractor; missing-boundary metadata.

**Tests required:** Synthetic low-spot put boundary; synthetic high-spot call boundary; no-boundary case for `q = 0`; interpolation and threshold behavior.

**Expected output:** Boundary arrays with missing-boundary flags, option-type-aware search direction, interpolation method, and threshold metadata.

**Pass/fail criteria:** Pass if put and call boundary directions are correct and threshold artifacts are labeled. Fail if the extractor invents boundaries, reverses option type, or silently drops ambiguous time levels.

**Possible failure modes:** Reversed search direction, divide-by-zero interpolation, threshold too large or too small, boundary dominated by `Smax`, or noisy near-maturity boundary.

**Human review needed before moving on:** Yes.

**Allowed to write code later:** Yes.

### Ticket 10: Delta and Gamma Diagnostics

**Objective:** Compute finite-difference Greeks and report fragile regions honestly.

**Source documents to read first:** SVP section 12; FN section 10; MG section 5.10.

**Files likely to be created or modified later:** `src/greeks/finite_difference.py`, tests under `tests/`.

**Functions/classes likely to be needed:** Central Delta; central Gamma; kink mask; boundary mask; Delta-bound diagnostics; convexity or negative-Gamma diagnostics.

**Tests required:** Known quadratic surface derivative tests; put Delta generally in `[-1, 0]`; call Delta generally in `[0, 1]`; negative Gamma diagnostics; masked versus unmasked reporting.

**Expected output:** Greek arrays, mask arrays, and diagnostic summaries for representative put and dividend-call cases.

**Pass/fail criteria:** Pass if stable interior Greeks are separated from payoff-kink and free-boundary bands. Fail if noisy Gamma is hidden, overinterpreted, or allowed to dominate headline reporting.

**Possible failure modes:** Endpoint derivative misuse, mask too broad or too narrow, treating every Gamma spike as a coding bug, or ignoring Delta bound violations.

**Human review needed before moving on:** Yes.

**Allowed to write code later:** Yes.

### Ticket 10A: Rannacher Smoothing Comparison Gate

**Objective:** Add Rannacher smoothing only as a named, reported option after plain CN/PSOR is understood.

**Source documents to read first:** SVP section 13; MG section 5.7.

**Files likely to be created or modified later:** CN solver option/tests and report comparison, likely under `src/solvers/`, `tests/`, and `results/01_solver_validation/`.

**Functions/classes likely to be needed:** Optional startup backward-Euler half-step mode; smoothing metadata; smoothed-versus-unsmoothed comparison helper.

**Tests required:** Smoothed versus unsmoothed price, boundary, Delta, and Gamma comparison on representative put and dividend-call cases.

**Expected output:** Decision note stating whether smoothing is default off/on for later experiments, with evidence and clear result labeling.

**Pass/fail criteria:** Pass if smoothing is explicit, documented, and not masking LCP or indexing bugs. Fail if smoothing silently changes all results or is used to hide defects.

**Possible failure modes:** Using smoothing to cover obstacle/complementarity errors, undocumented result tables, or inconsistent smoothing settings across validation cases.

**Human review needed before moving on:** Yes.

**Allowed to write code later:** Yes, only after Tickets 3-10 pass.

### Ticket 11: Grid Refinement and Domain Sensitivity Experiments

**Objective:** Validate robustness across coarse/medium/fine grids and `Smax = 4K, 5K, 6K`.

**Source documents to read first:** SVP sections 8-9; PR Gate 2; SH Experiment 02.

**Files likely to be created or modified later:** `experiments/02_convergence_domain.py` or `experiments/01_solver_validation.py`, `results/01_solver_validation/`.

**Functions/classes likely to be needed:** Grid-refinement runner; domain-sensitivity runner; result table serializer; optional plot serializer.

**Tests required:** Price differences shrink from coarse-to-medium to medium-to-fine; target region remains stable under larger `Smax`; boundary, Greeks, obstacle, complementarity, PSOR iteration counts, and runtime are summarized.

**Expected output:** Refinement/domain tables for prices, boundaries, Greeks, diagnostics, PSOR iterations, and runtime.

**Pass/fail criteria:** Pass if main price and boundary conclusions stabilize and target-region behavior is insensitive to domain extension. Fail if refinement changes the main financial interpretation or if `Smax` materially changes the target region.

**Possible failure modes:** Domain too small, fine grid too slow, jagged boundary curves, new obstacle violations under refinement, or Gamma instability outside known delicate bands.

**Human review needed before moving on:** Yes.

**Allowed to write code later:** Yes.

### Ticket 12: Solver Validation Report Assembly

**Objective:** Assemble evidence into `reports/03_solver/solver_validation_report.md`.

**Source documents to read first:** SVP sections 15-18; PR reports and gates; SH experiment template.

**Files likely to be created or modified later:** `reports/03_solver/solver_validation_report.md`, result tables, and result figures.

**Functions/classes likely to be needed:** None unless lightweight report-generation helpers already exist.

**Tests required:** Verify all required tables exist; rerun the final selected test command; check `git status --short`; cross-check every SVP acceptance criterion.

**Expected output:** Solver validation report covering closed-form validation, no-dividend call, American put, dividend call, obstacle/complementarity diagnostics, boundary extraction, Greeks, grid refinement, domain sensitivity, limitations, and human-review gate.

**Pass/fail criteria:** Pass if every SVP acceptance criterion is addressed. Fail if any missing validation item is not explicitly marked as a blocker.

**Possible failure modes:** Reporting only price error, hiding failed regimes, omitting complementarity, omitting Greeks, implying readiness for neural labels before human approval, or skipping domain sensitivity.

**Human review needed before moving on:** Yes, this is the final solver-validation gate.

**Allowed to write code later:** No solver code. Documentation and result assembly only.

## Recommended First Coding Prompt

Use this prompt only after leaving this planning/documentation stage:

> Implement Ticket 1 only. Read the ticket file and the listed source documents first. Add payoff functions and European Black-Scholes closed-form utilities with focused unit tests. Do not implement grids, CN, PSOR, boundary extraction, Greeks, experiments, reports, stress maps, datasets, or neural models. After implementation, run the Ticket 1 tests and show `git status --short`.

## Minimum Safe Path

If time is limited, complete Tickets 0-6, 8-9, 11, and 12. This gives a basic validated solver checkpoint, but it is not enough to move to stress maps or neural labels unless Ticket 7 and Ticket 10 are either completed or explicitly listed as blockers in the report.

Minimum safe path summary:

1. Confirm repo/test setup.
2. Build payoff and European closed-form utilities.
3. Build grid/operator setup.
4. Validate European CN against closed form.
5. Implement PSOR/LCP core.
6. Validate American puts.
7. Validate no-dividend American calls.
8. Add obstacle/complementarity diagnostics.
9. Add continuation-premium boundary extraction.
10. Run grid/domain sensitivity.
11. Assemble a report that clearly states any incomplete dividend-call or Greek validation as blocking.

## Full Thorough Path

Complete Tickets 0-12, including Ticket 10A if Greeks are noisy or the report needs a smoothing comparison. Only after human approval of Ticket 12 should the project begin boundary/Greek reporting beyond validation, stress-regime maps, dataset construction, or neural surrogate work.

Full thorough path summary:

1. Complete setup, analytic utilities, grid/operator, European CN, and PSOR core.
2. Validate American puts, no-dividend calls, and dividend-paying calls.
3. Add obstacle and complementarity diagnostics required in every American case.
4. Extract boundaries from continuation premium using option-type-aware logic.
5. Compute Delta and Gamma with masks and diagnostic bands.
6. Compare Rannacher smoothing only after the plain method is understood.
7. Run grid refinement and domain sensitivity.
8. Assemble the solver validation report and stop for human approval.

## Self-Check

1. Does this ticket plan cover all validation requirements from `solver_validation_plan.md`?
   - Yes. It covers European closed-form checks, no-dividend American call validation, American put validation, dividend-paying American call validation, obstacle/complementarity diagnostics, boundary extraction, Greeks, grid refinement, domain sensitivity, Rannacher smoothing as a reported option, and final report assembly.
2. Are the tickets small enough to review one by one?
   - Yes. Each ticket has a single validation or implementation responsibility and a required human-review gate.
3. Does it prevent Codex from implementing the whole project automatically?
   - Yes. The rules require one-ticket-at-a-time implementation, tests, git status, and human review before moving on.
4. Is the first coding step clearly identified?
   - Yes. Ticket 1 is the first coding ticket, and the recommended first coding prompt explicitly limits Codex to payoff and European Black-Scholes utilities only.

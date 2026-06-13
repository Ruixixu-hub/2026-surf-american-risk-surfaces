# SURF2026 American Option Risk Surfaces Planning Report

## Source Basis

This planning report is based on the two project documents in `docs/`:

- `docs/student_handout_full_report.pdf`
- `docs/Student_Methodology_FreeBoundary_Risk_Surfaces.pdf`

The report is intentionally a planning and research-readiness document. It does not implement
algorithms, run experiments, or claim that the project is ready for neural surrogate modelling.

## 1. Whole-Picture Explanation

This project is about American options, which are harder than European options because the
holder can exercise early. That early-exercise feature creates a moving boundary: on one side it
is better to exercise, and on the other side it is better to continue holding the option.

The project's central object is not just the option price. It is the full risk surface:

- price over spot price, time, volatility, rate, dividend yield, and maturity;
- Delta and Gamma for risk and hedging;
- the early-exercise boundary;
- stress maps showing how the boundary and Greeks change across regimes.

The safest project logic is:

1. Build a trusted classical numerical solver using Crank-Nicolson finite differences plus PSOR.
2. Validate it against known financial facts and convergence checks.
3. Use it to generate reliable price, boundary, and Greek labels.
4. Study stress regimes and risk surfaces.
5. Only then train neural surrogates.
6. Treat neural networks as fast accelerators and diagnostics, not replacements for the solver.
7. Preserve financial structure: American price must stay above payoff, Delta must obey bounds,
   and boundary claims must be checked directly.

The key lesson from the handout is that one neural network should not be expected to solve
price, boundary, and Greeks all at once. The stronger final direction is a modular workflow:

- positive continuation-premium price surrogate;
- direct boundary diagnostic head;
- bounded supervised Delta head;
- PSOR spot checks for audit and new regimes.

## 2. Main Research Question and Subquestions

Main research question:

How can we generate fast, stable, and financially consistent American option price and risk
surfaces while preserving the payoff obstacle and accurately recovering the early-exercise
boundary?

Subquestions:

1. How do volatility, interest rate, dividend yield, and maturity affect the early-exercise
   boundary?
2. Where do Delta and Gamma become unstable near the exercise boundary?
3. Can constrained neural surrogates reproduce price and risk behavior without violating payoff,
   monotonicity, convexity, or Delta bounds?
4. Which regimes can safely use neural acceleration, and which regimes still require direct PSOR
   solving?
5. How should failed model claims be repaired instead of hidden?

## 3. Knowledge Priority

### Must Understand Before Implementation

1. Basic option concepts: call, put, strike, maturity, payoff, intrinsic value, time value.
2. Difference between European and American options.
3. Risk-neutral pricing at a conceptual level.
4. Black-Scholes PDE meaning, not full derivation.
5. Time-to-maturity transformation.
6. American option obstacle condition: value cannot fall below payoff.
7. Continuation region, exercise region, and free boundary.
8. Why Delta and Gamma become unstable near payoff kinks and boundaries.
9. What CN/PSOR is supposed to enforce.
10. Solver validation logic: closed-form checks, no-dividend American call theorem, grid
    refinement, domain sensitivity.

### Can Learn While Implementing

1. Finite-difference stencil details.
2. Crank-Nicolson matrix construction.
3. PSOR update mechanics and relaxation tuning.
4. Boundary extraction from continuation premium.
5. Numerical Delta and Gamma formulas.
6. Rannacher smoothing.
7. Dataset generation from PDE surfaces.
8. Regime-split evaluation.
9. Neural architecture details such as softplus premium output and bounded Delta heads.

### Useful But Not Urgent

1. Full stochastic calculus derivation.
2. Optimal stopping theory proofs.
3. Variational inequality existence and uniqueness theory.
4. Detailed numerical analysis convergence proofs.
5. Heston, jump diffusion, rough volatility, and local volatility extensions.
6. Market calibration and empirical option data.

### Can Be Treated as Black-Box Temporarily

1. Formal measure-theoretic probability.
2. Full proof of Black-Scholes PDE.
3. Full proof of PSOR convergence.
4. Deep PINN theory.
5. Advanced publication strategy and journal positioning.
6. Alternative high-dimensional American option methods such as duality and LSM, until the core
   one-dimensional Black-Scholes workflow is stable.

## 4. Pickup Knowledge Plan

For students with accounting plus data and ML background, learn in this order:

1. Finance intuition first
   - Learn payoff diagrams, early exercise intuition, dividend effect, interest rate effect, and why
     American puts and dividend-paying calls have exercise boundaries.
2. Mathematical objects second
   - Learn what the PDE, variational inequality, obstacle condition, and free boundary mean in
     plain language. Do not start with proofs.
3. Numerical solver logic third
   - Understand grids, finite differences, Crank-Nicolson, and PSOR as a disciplined way to
     approximate the American option value while enforcing the obstacle.
4. Diagnostics fourth
   - Learn what can go wrong: obstacle violation, unstable boundary, bad Gamma near kinks,
     domain too small, grid too coarse, random splits that exaggerate neural accuracy.
5. Surrogate modeling fifth
   - Connect existing ML knowledge to this project: features, labels, train-test split, regime
     split, losses, constraints, and why architecture matters more than only adding penalties.
6. Research writing last
   - Learn how each experiment supports or rejects a claim. A failed autograd Delta result is not
     wasted: it just changes the paper claim.

## 5. Four-Layer Literature Review Plan

### Layer 1: Classical American Option Pricing

Purpose: understand the financial problem and benchmark methods.

Read for:

- American option as early-exercise problem;
- finite-difference valuation;
- Longstaff-Schwartz as contrast, not the main method;
- duality as background for bounds.

Anchor topics:

- Brennan and Schwartz finite-difference American put valuation;
- Longstaff and Schwartz least-squares Monte Carlo;
- Haugh and Kogan duality methods;
- textbook treatments of American options under Black-Scholes.

Output report:

- 2-3 pages explaining why American options need a free-boundary formulation and why this
  project chooses CN/PSOR.

### Layer 2: Free-Boundary, Variational Inequality, and LCP Methods

Purpose: understand the solver's mathematical structure.

Read for:

- obstacle problem;
- continuation and exercise regions;
- variational inequality;
- LCP discretization;
- PSOR projection;
- boundary extraction and complementarity diagnostics.

Anchor topics:

- Black-Scholes PDE with dividends;
- forward time variable;
- Crank-Nicolson LCP;
- boundary conditions for puts and dividend-paying calls;
- Rannacher smoothing.

Output report:

- 3-5 pages explaining the solver, validation ladder, and expected failure modes.

### Layer 3: Scientific Machine Learning, PINNs, and DGM

Purpose: understand why neural PDE solvers are attractive but risky.

Read for:

- PINNs and PDE residual losses;
- Deep Galerkin Method;
- constraint enforcement problems;
- why PDE residuals should not be forced inside the exercise region;
- why nice plots can hide boundary errors.

Anchor topics:

- Raissi, Perdikaris, and Karniadakis on PINNs;
- Sirignano and Spiliopoulos on DGM;
- PINN limitations for nonsmooth and constrained PDEs.

Output report:

- 2-3 pages comparing mesh-free neural PDE ideas with the safer supervised-surrogate approach.

### Layer 4: Deep Learning for American Options and Risk Surfaces

Purpose: place this project's contribution.

Read for:

- deep optimal stopping;
- neural continuation value methods;
- neural free-boundary methods;
- supervised surrogates for risk surfaces;
- boundary and Greek diagnostics.

Anchor topics:

- Becker, Cheridito, and Jentzen on deep optimal stopping;
- deep American option pricing and hedging;
- free-boundary-aware neural methods;
- recent American option PINN and process-informed papers;
- risk-surface applications.

Output report:

- 4-6 pages identifying the project gap: not "a plain PINN for American puts," but
  structure-preserving risk surfaces with price, boundary, and Delta diagnostics.

## 6. Recommended Reading Order

1. Project PDFs first: read both documents completely.
2. Introductory option pricing notes: payoff, arbitrage, Black-Scholes intuition.
3. American option chapter from a computational finance textbook.
4. Finite differences for Black-Scholes PDE.
5. Variational inequality and LCP explanation.
6. PSOR algorithm and obstacle projection.
7. Boundary extraction and Greeks.
8. Classical papers: Brennan-Schwartz, Longstaff-Schwartz, Haugh-Kogan.
9. PINNs and DGM papers.
10. Deep learning American option papers.
11. Only after this: neural surrogate architecture papers and implementation examples.

## 7. Correct Project Workflow

### Stage 0: Planning and Knowledge Pickup

- Read project PDFs.
- Build glossary.
- Write literature map.
- Define claims before coding.
- Decide success and failure metrics before each experiment.

### Stage 1: Mathematical Formulation Report

- Explain payoff, PDE, variational inequality, LCP, free boundary, and continuation premium.
- Normalize strike to `K = 1` for experiments.
- Separate put and dividend-paying call behavior.

### Stage 2: Classical Solver Validation

- Implement only after the formulation is understood.
- Validate European closed-form case.
- Validate no-dividend American call equals European call.
- Check American put obstacle behavior.
- Run grid refinement and domain sensitivity.

### Stage 3: Boundary and Greek Diagnostics

- Extract exercise boundary from continuation premium.
- Compute Delta and Gamma.
- Apply masks near boundary and payoff kink when needed.
- Report instability instead of hiding it.

### Stage 4: Stress-Regime Risk Analytics

- Sweep volatility, rate, dividend yield, and maturity.
- Produce required price, Delta, Gamma, boundary, heatmap, convergence, and runtime figures.
- Identify financially interesting regimes.

### Stage 5: Dataset Construction

- Generate labels only from validated solver.
- Store features, price, premium, payoff, boundary, exercise indicator, Greek masks, and sample
  weights.
- Use regime splits, not only random point splits.

### Stage 6: Baseline Surrogates

- Train plain price model.
- Train obstacle-weighted or constrained variants.
- Test whether they violate payoff or financial shape.
- Reject weak models honestly.

### Stage 7: Structure-Preserving Surrogate Workflow

- Prefer positive continuation-premium price model.
- Train direct boundary heads.
- Train bounded Delta head only if autograd Delta fails.
- Keep modular architecture.

### Stage 8: Integrated Application Workflow

- Combine price, boundary, and Delta components.
- Test on fresh PSOR cases.
- Report speed, price RMSE, boundary MAE, Delta RMSE, and constraint violations.
- Use PSOR for audit outside the training envelope.

### Stage 9: Claim Audit and Paper Writing

- Keep only claims supported by evidence.
- Reframe failed claims.
- Write limitations clearly.
- Preserve failed-attempt summaries.

## 8. Reports to Produce

1. `00_planning/planning_report.md`
   - This report: whole picture, knowledge plan, workflow, risks, gates.
2. `01_literature/literature_map.md`
   - Four-layer literature review with annotated readings and project relevance.
3. `02_math/formulation_note.md`
   - Beginner-friendly explanation of payoff, PDE, VI, LCP, boundary, continuation premium,
     Greeks.
4. `03_solver/solver_validation_report.md`
   - Closed-form validation, no-dividend call check, obstacle and complementarity diagnostics,
     grid and domain sensitivity.
5. `04_boundary_greeks/boundary_greek_report.md`
   - Boundary extraction, Delta/Gamma behavior, instability near kink and boundary, masking
     rules.
6. `05_stress_maps/stress_regime_report.md`
   - Parameter sweeps, risk maps, qualitative finance interpretation.
7. `06_dataset/dataset_construction_report.md`
   - Feature definitions, labels, masks, regime splits, quality checks.
8. `07_surrogates/baseline_surrogate_report.md`
   - Plain and constrained model results, obstacle violations, rejected claims.
9. `08_surrogates/structure_preserving_report.md`
   - Positive-premium price model, boundary head, bounded Delta head.
10. `09_integrated/integrated_workflow_report.md`
    - Fresh PSOR comparison, speedup, accuracy, violations, use-case interpretation.
11. `10_failures/failure_log.md`
    - Failed attempts, why they failed, how claims changed.
12. `11_claim_audit/final_claim_matrix.md`
    - Ready claims, restricted claims, rejected claims, wording for paper.

## 9. Codex vs Human Responsibilities

Codex should do:

- summarize papers into structured notes;
- explain formulas in beginner language;
- draft implementation plans after decision gates;
- generate code step by step later;
- write tests and experiment scripts later;
- produce tables, plots, and report drafts later;
- maintain failure logs;
- compare results against predefined metrics;
- suggest claim wording.

Humans should check:

- whether the financial interpretation makes sense;
- whether assumptions match supervisor expectations;
- whether literature coverage is academically sufficient;
- whether reported claims are too strong;
- whether figures tell a useful research story;
- whether failed results are honestly described;
- whether final paper framing fits the target journal;
- whether any implementation output is plausible before trusting it.

Human supervisor should approve:

- final research question;
- decision to move past solver validation;
- decision to start neural surrogate stage;
- decision to include or exclude neural results from main paper;
- final claim matrix.

## 10. Beginner Risk Warnings

Likely mistakes:

1. Starting with neural networks too early.
2. Treating American options like European options plus a small adjustment.
3. Forgetting that price must be at least payoff.
4. Using random grid-point splits and overestimating model performance.
5. Reporting price RMSE only while ignoring boundary and Greeks.
6. Trusting autograd Delta from a price network without validation.
7. Forcing PDE loss inside the exercise region.
8. Hiding bad dividend-call boundary results.
9. Using too small an `Smax` domain.
10. Confusing numerical noise near the payoff kink with a simple coding bug.
11. Forgetting grid and domain sensitivity checks.
12. Making broad claims like "neural networks replace finite differences."
13. Overfitting the research story to successful experiments only.
14. Losing failed attempts instead of documenting them.

## 11. Decision Gates

### Gate 1: Literature to Implementation

Move to solver implementation only when students can explain:

- American vs European option difference;
- payoff obstacle;
- continuation and exercise regions;
- free boundary;
- why CN/PSOR is used;
- validation ladder;
- what metrics will decide solver reliability.

### Gate 2: Solver to Application Experiments

Move to stress maps only when:

- European closed-form test passes;
- no-dividend American call matches European call;
- obstacle violation is near tolerance;
- grid refinement improves price and boundary stability;
- domain extension does not materially change region of interest;
- complementarity diagnostics are understood.

### Gate 3: Application Experiments to Neural Surrogate

Move to neural modeling only when:

- stress maps show clear financial messages;
- boundary and Greek diagnostics are produced;
- dataset labels are trustworthy;
- regime split is defined;
- success criteria and failure criteria are written before training.

### Gate 4: Price Surrogate to Boundary and Delta Heads

Move beyond price only when:

- price model preserves the obstacle;
- held-out regime performance is acceptable;
- near-boundary price behavior is diagnosed;
- boundary cannot be safely inferred from threshold proxy alone;
- autograd Delta has been directly tested.

### Gate 5: Solver Plus Surrogate to Paper Claim

Claim the full workflow only when:

- fresh PSOR comparison is run;
- price RMSE, boundary MAE, Delta RMSE, speedup, and violations are reported;
- failed claims are restricted or rejected;
- the final wording says "accelerator and diagnostic workflow," not "replacement solver."

## 12. Next-Action Checklist

1. Read both PDFs fully and make a one-page glossary.
2. Write a short explanation of American early exercise in your own words.
3. Make a concept map: payoff, obstacle, continuation premium, boundary, Delta, Gamma.
4. Start Layer 1 literature notes.
5. Start Layer 2 solver notes before any coding.
6. Prepare the solver validation checklist.
7. Define report templates before experiments.
8. Create a failure log template.
9. Ask the supervisor to confirm the staged workflow and decision gates.
10. After approval, begin the literature map and formulation note before any solver code.

## 13. Additional Suggestions from Codex

Build a research notebook discipline from day one:

- Every experiment should start with a claim.
- Every claim should have a success metric and a failure metric.
- Every failed attempt should produce a short failure note.
- Every plot should answer a financial question.
- Every neural model should be compared against both PSOR accuracy and financial constraints.
- Every final claim should have a matching table row showing evidence.

Recommended standing rule:

Before training any model, write the sentence you hope to claim in the paper. After the
experiment, either keep it, weaken it, or reject it.

## 14. Self-Review

First check: beginner suitability

The plan avoids assuming prior knowledge of stochastic calculus, variational inequalities, LCPs,
or free-boundary theory. It starts from option intuition and separates must-know concepts from
black-box material.

Second check: project document alignment

The plan follows the PDFs' order: formulation, CN/PSOR benchmark, diagnostics, stress maps,
dataset, constrained surrogates, direct boundary head, bounded Delta head, integrated workflow,
and claim audit.

Third check: avoiding premature implementation

The plan explicitly blocks implementation until literature, formulation, solver validation
criteria, and decision gates are understood. Neural modeling is delayed until the classical
benchmark and application evidence are stable.

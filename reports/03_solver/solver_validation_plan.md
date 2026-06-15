# Solver Validation Plan for the CN/PSOR American Option Solver

## Source basis

This planning document is based on the current project materials:

- `reports/00_planning/planning_report.md`
- `reports/01_literature/literature_map.md`
- `reports/02_math/formulation_note.md`
- `docs/student_handout_full_report.pdf`
- `docs/Student_Methodology_FreeBoundary_Risk_Surfaces.pdf`

This is a planning document only. It does not validate a solver, does not implement CN/PSOR, does
not run experiments, and does not create neural-network labels.

## 1. Purpose of the solver validation plan

The purpose of this plan is to define how the future Crank-Nicolson / projected successive
over-relaxation solver, abbreviated CN/PSOR, must be tested before it is trusted.

Validation must come before stress maps, datasets, and neural surrogates because every later stage
depends on the solver labels. If the solver is wrong, then:

- stress maps may show patterns created by numerical error rather than option economics;
- early-exercise boundaries may be misplaced;
- Delta and Gamma surfaces may be misleading;
- neural surrogate models may learn incorrect prices, boundaries, and Greeks;
- low neural-network error would only mean the network copied unreliable labels well.

For this project, the solver is the reference standard. The neural stage should be an accelerator and
diagnostic layer built on top of trusted solver outputs, not a replacement for solver validation.

The validation goal is not to prove the numerical method from first principles. The goal is to build
enough evidence that the first implementation respects the financial structure of American options
and behaves stably under reasonable grid and domain changes.

## 2. Solver scope for the first implementation

The first implementation should stay deliberately narrow.

The solver should cover:

- American put options under Black-Scholes with continuous dividends;
- dividend-paying American call options under Black-Scholes with continuous dividends;
- European option mode, or an equivalent comparison mode, for validation against closed-form
  Black-Scholes prices;
- normalized strike `K = 1` unless a later document explicitly states otherwise;
- spot domain `S in [0, Smax]`, with starting default `Smax = 4K` marked as tentative;
- main region of interest `S/K in [0.4, 1.8]`, because this is where later risk surfaces are expected
  to be interpreted.

The first implementation should not cover:

- stochastic volatility;
- jumps;
- local volatility;
- market calibration;
- high-dimensional baskets;
- neural-network pricing;
- full production dataset generation.

Those extensions are only reasonable after the basic one-dimensional solver passes this validation
plan.

## 3. Mathematical objects to validate

The future implementation must validate each mathematical object separately enough that a failure
can be diagnosed.

Payoff:

- put payoff: `Phi_put(S) = max(K - S, 0)`;
- call payoff: `Phi_call(S) = max(S - K, 0)`;
- at `tau = 0`, the numerical value should equal the payoff.

Finite spot/time grid:

- spot grid: `S_i = i * Delta S`, for `i = 0, ..., M`;
- time-to-maturity grid: `tau_n = n * Delta tau`, for `n = 0, ..., N`;
- solver marches from known payoff at `tau = 0` toward larger time-to-maturity.

Black-Scholes operator:

- continuous operator:
  `L U = 0.5 * sigma^2 * S^2 * U_SS + (r - q) * S * U_S - r * U`;
- the finite-difference version `L_h` should approximate `U_S` and `U_SS` on interior nodes;
- boundary conditions must be included consistently in the right-hand side.

Crank-Nicolson continuation step:

- continuation equation:
  `A U^{n+1} = b^n`;
- standard matrix form:
  `A = I - (Delta tau / 2) L_h`;
  `b^n = [I + (Delta tau / 2) L_h] U^n`, with boundary adjustments.

American obstacle condition:

- American value must satisfy `U >= Phi`;
- any point with `U < Phi` is financially invalid.

LCP/complementarity logic:

- LCP form at each time step:
  `A U^{n+1} >= b^n`;
  `U^{n+1} >= Phi`;
  `(U^{n+1} - Phi)^T (A U^{n+1} - b^n) = 0`;
- in plain language, each grid point should either continue like a PDE point or sit on the payoff
  obstacle.

PSOR projection:

- PSOR should solve the LCP by updating each node and projecting it back above payoff;
- projection is the mechanism that prevents the American value from falling below immediate
  exercise value.

Boundary extraction from continuation premium:

- continuation premium:
  `P_cont(S, tau) = U(S, tau) - Phi(S)`;
- exercise region has premium near zero;
- continuation region has positive premium;
- the boundary is the transition between those regions.

Delta and Gamma finite-difference diagnostics:

- Delta: `Delta = partial U / partial S`;
- Gamma: `Gamma = partial^2 U / partial S^2`;
- central finite differences should be used on interior nodes;
- one-sided or masked reporting may be needed near boundaries, payoff kinks, and the free boundary.

## 4. Validation Case 1: European closed-form check

Purpose:

- Check the finite-difference grid, Black-Scholes operator, boundary handling, and Crank-Nicolson
  continuation step before adding American projection complexity.

What should be compared:

- finite-difference European prices against closed-form Black-Scholes prices with continuous
  dividends;
- compare prices mainly in the region `S/K in [0.4, 1.8]`;
- report both maximum absolute error and root mean squared error in the region of interest.

Call or put cases:

- at minimum, test a European put, because the American put is the core early-exercise case;
- also test a European call if the same solver mode will support dividend-paying American calls.

Suggested parameter settings:

- `K = 1`;
- `Smax = 4K` as the tentative baseline domain;
- `r = 0.05`;
- `q = 0.00` and `q = 0.03`;
- `sigma = 0.20` and `sigma = 0.40`;
- `T = 0.5` and `T = 1.0`;
- use at least a medium and fine grid before claiming the test passes.

Expected output:

- table of European closed-form errors by option type, parameter set, and grid size;
- optional price-error plot over `S/K`;
- short note identifying where the largest error occurs.

Pass/fail criteria:

- tentative: maximum absolute error in the region of interest should be small and should decrease
  under grid refinement;
- tentative: medium-to-fine grid error should improve rather than jump unpredictably;
- errors near `S = 0`, `S = Smax`, and the payoff kink should be reported separately if they dominate;
- the test fails if the error is large across the ordinary interior region or does not improve when
  the grid is refined.

What failure would mean:

- wrong finite-difference coefficients;
- incorrect time direction;
- missing or wrong boundary adjustments;
- incorrect dividend treatment;
- incorrect closed-form comparison;
- grid/domain too coarse for the selected parameters.

The American projection should not be debugged until this European check is credible.

## 5. Validation Case 2: No-dividend American call check

Purpose:

- Check that the American exercise logic does not create artificial early exercise where theory says
  early exercise should not occur.

Why American call with `q = 0` should match European call:

- under the standard Black-Scholes assumptions with no dividends, exercising a call early gives up
  remaining optionality without receiving any dividend benefit;
- therefore, a no-dividend American call should have the same value as the corresponding European
  call;
- this is a powerful sanity check because it tests the obstacle logic and boundary extraction, not
  only the PDE solver.

Suggested parameter settings:

- `K = 1`;
- `q = 0.00`;
- `Smax = 4K` as tentative baseline;
- `r = 0.01` and `r = 0.05`;
- `sigma = 0.20` and `sigma = 0.40`;
- `T = 0.5` and `T = 1.0`;
- compare at multiple grids, such as coarse, medium, and fine.

Expected output:

- table comparing American call price to European closed-form call price;
- maximum absolute difference in the region `S/K in [0.4, 1.8]`;
- reported obstacle violation;
- boundary diagnostic showing no genuine early-exercise boundary before maturity.

Pass/fail criteria:

- tentative: American and European call values should agree within numerical tolerance in the region
  of interest;
- tentative: the maximum difference should shrink or remain negligible under refinement;
- no stable early-exercise region should appear for `q = 0`;
- tiny numerical threshold artifacts should be flagged, not interpreted as finance.

What failure would mean:

- projection is being applied incorrectly;
- call payoff or call boundary conditions are wrong;
- dividend yield is not actually zero in the computation;
- upper boundary is distorting the interior;
- boundary extraction threshold is too aggressive;
- the solver is inventing early exercise and cannot yet be trusted.

## 6. Validation Case 3: American put obstacle and boundary sanity check

Purpose:

- Check the first genuinely American case where early exercise should matter.

Obstacle condition `U >= payoff`:

- for every grid point, the American put value should be at least
  `Phi_put(S) = max(K - S, 0)`;
- the maximum obstacle violation should be computed as
  `max_i,n max(Phi_i - U_i^n, 0)`.

Expected put boundary behavior:

- the exercise region is usually at low spot values;
- the boundary is usually below the strike for positive time-to-maturity;
- near maturity, the boundary should move toward the payoff kink around `S = K`;
- higher volatility often lowers the put exercise boundary because waiting becomes more valuable.

Continuation premium check:

- compute `P_cont = U - Phi_put`;
- in the exercise region, the premium should be near zero;
- in the continuation region, the premium should be positive;
- the boundary should be extracted from the transition from near-zero premium to positive premium,
  not from visual guessing.

Suggested parameter settings:

- `K = 1`;
- `r = 0.05`;
- `q = 0.00` or `q = 0.03`;
- `sigma = 0.20`, `0.40`, and optionally `0.60`;
- `T = 1.0`;
- `Smax = 4K` as tentative baseline.

Expected output:

- table of obstacle violations;
- table of complementarity residuals;
- selected price values at representative spots such as `S/K = 0.8, 1.0, 1.2`;
- boundary curve over time-to-maturity;
- continuation premium plot for selected time levels.

Pass/fail criteria:

- tentative: obstacle violation should be near the PSOR tolerance and should not grow with time;
- boundary curve should be smooth enough to interpret after grid refinement;
- boundary should stay financially plausible, usually below `K` for the put case;
- continuation premium should be nonnegative up to numerical tolerance;
- the extracted exercise region should be low-spot, not high-spot.

What failure would mean:

- PSOR projection is wrong or not applied at every update;
- payoff is incorrectly defined;
- boundary extraction logic is reversed;
- complementarity is not being solved even if `U >= Phi`;
- grid is too coarse near the free boundary;
- domain or boundary conditions are distorting the solution.

## 7. Validation Case 4: Dividend-paying American call sanity check

Purpose:

- Check the call case where early exercise can become financially meaningful because dividends can
  make holding the underlying asset more attractive than holding the option.

Why dividend-paying calls may have early exercise:

- a call holder who exercises receives the asset;
- with dividends, owning the asset can provide dividend benefit;
- when dividend yield is high enough, early exercise of a deep-in-the-money call may become optimal;
- this behavior is different from the no-dividend call case and must be validated separately.

Expected boundary behavior:

- the exercise region for dividend-paying calls is usually at high spot values;
- the call boundary may appear above the strike;
- increasing `q` should generally expand the early-exercise region, meaning the boundary may move
  closer to the strike;
- with `q = 0`, the boundary should disappear as a genuine early-exercise boundary.

Suggested `q` values:

- `q = 0.00` as the no-dividend control;
- `q = 0.03` as a moderate dividend case;
- `q = 0.06` as a higher dividend case;
- `q = 0.10` as a stress case;
- all thresholds and final parameter choices are tentative and should be reviewed after initial
  results.

Suggested additional parameters:

- `K = 1`;
- `r = 0.05`;
- `sigma = 0.20` and `sigma = 0.40`;
- `T = 0.5`, `1.0`, and optionally `2.0`;
- `Smax = 4K`, then repeat with larger domains.

Expected output:

- table of American call versus European call differences by `q`;
- boundary curves for dividend-paying calls;
- obstacle violation and complementarity residual tables;
- note explaining whether the boundary appears only when dividends are large enough.

Pass/fail criteria:

- tentative: `q = 0` should match the European call check;
- positive `q` cases may show early exercise at high spot values;
- the boundary should move in a financially sensible direction as `q` increases;
- boundary behavior should remain stable under grid and domain refinement;
- obstacle and complementarity diagnostics must remain acceptable.

What failure would mean:

- call upper boundary condition is wrong;
- dividend term `(r - q) S U_S` is wrong;
- no-dividend and dividend call logic are mixed up;
- boundary extraction is not option-type aware;
- `Smax` is too small and is creating artificial high-spot exercise behavior.

## 8. Grid refinement plan

Why grid refinement is needed:

- a single grid can hide numerical artifacts;
- prices, boundaries, and Greeks should become more stable when the grid is refined;
- formal convergence proof is not required at this stage, but unstable refinement behavior is a
  warning sign.

Suggested coarse/medium/fine grids:

- coarse: approximately `M = 100`, `N = 100`;
- medium: approximately `M = 200`, `N = 200`;
- fine: approximately `M = 400`, `N = 400`;
- these are tentative starting choices and can be adjusted for runtime and stability;
- if a later implementation uses different grid counts, the report should explain why.

Which outputs to compare:

- prices at representative spot/time points;
- maximum and RMSE price differences against the finest available grid;
- extracted boundary curves;
- Delta and Gamma in stable interior regions;
- obstacle violation;
- complementarity residual;
- PSOR iteration counts and runtime.

Price stability:

- price surfaces should change less from medium to fine than from coarse to medium;
- largest differences should be identified and explained.

Boundary stability:

- boundary curves should become less jagged with refinement;
- boundary locations should not jump materially in the region used for interpretation;
- if the boundary is unstable near maturity, report it separately rather than hiding it.

Greek stability:

- Delta should be more stable than Gamma;
- Gamma may remain noisy near `S = K`, near maturity, and near the free boundary;
- Greek comparisons should include masked interior regions and special reporting near delicate
  bands.

Pass/fail interpretation:

- pass: prices stabilize, boundary movement shrinks, obstacle/complementarity diagnostics remain
  acceptable, and Greek instability is localized and explainable;
- fail: refinement changes the main financial interpretation, creates new obstacle violations, moves
  boundaries unpredictably, or makes Delta/Gamma unusable outside known delicate regions.

## 9. Domain sensitivity plan

Why `Smax` matters:

- the Black-Scholes problem is defined on an unbounded spot domain, but the solver uses a finite
  domain;
- if `Smax` is too small, the artificial upper boundary can distort prices and boundaries in the
  region we care about;
- this is especially important for dividend-paying calls because high-spot behavior matters.

Suggested `Smax` values:

- baseline: `Smax = 4K`;
- sensitivity checks: `Smax = 5K` and `Smax = 6K`;
- for high-volatility or long-maturity cases, consider a larger `Smax` if the first checks are not
  stable.

Target region of interest:

- primary interpretation region: `S/K in [0.4, 1.8]`;
- domain sensitivity should focus on whether this region changes when `Smax` changes.

What should remain stable:

- prices in the target region;
- extracted boundary curves inside the target region;
- Delta and Gamma away from the upper boundary;
- obstacle and complementarity diagnostics;
- qualitative conclusions about early exercise.

Pass/fail interpretation:

- pass: increasing `Smax` does not materially change prices, boundaries, or Greeks in the target
  region;
- fail: changing `Smax` changes the main boundary or price conclusions, which means the domain is
  too small or the boundary condition is wrong;
- if only far out-of-region values change, report that separately and keep interpretation focused on
  the stable region.

## 10. Obstacle and complementarity diagnostics

Obstacle violation definition in plain language:

- the obstacle violation measures how far the computed American value falls below immediate
  exercise value;
- American options should never be worth less than payoff;
- a positive violation means the solver has broken the most basic American option constraint.

Formula:

`ObstacleViolation = max_i,n max(Phi_i - U_i^n, 0)`.

Complementarity residual definition in plain language:

- complementarity checks whether each point is behaving like either a continuation point or an
  exercise point;
- being above payoff is not enough by itself;
- the value must also satisfy the LCP logic connecting the equation gap and the payoff gap.

Use:

- value gap: `U - Phi`;
- equation gap: `A U - b`;
- pointwise product: `(U - Phi) * (A U - b)`;
- maximum absolute product over interior nodes and time steps is the main residual.

Where they should be computed:

- obstacle violation: all grid nodes and all time levels;
- complementarity residual: interior nodes and time levels where the LCP right-hand side is defined;
- boundary nodes should be reported separately because they are imposed conditions, not solved
  interior LCP nodes.

Acceptable behavior:

- tentative: obstacle violation should be near the PSOR stopping tolerance;
- tentative: complementarity residual should be small and should improve or remain controlled under
  grid refinement;
- residuals should be summarized by maximum and, if useful, by percentile to avoid one point hiding
  the overall behavior.

What large violations imply:

- large obstacle violation means projection or payoff handling is wrong;
- large complementarity residual means the solver may not satisfy the LCP even if values are above
  payoff;
- violations concentrated at boundaries suggest boundary-condition or indexing problems;
- violations concentrated near the free boundary may suggest grid, tolerance, or boundary extraction
  issues.

## 11. Boundary extraction validation

Continuation premium definition:

`P_cont(S, tau) = U(S, tau) - Phi(S)`.

For a valid American option, `P_cont >= 0` up to numerical tolerance.

How boundary should be extracted:

- at each time level, inspect the continuation premium across spot;
- identify the transition between near-zero premium and positive premium;
- for puts, search from low spot upward because exercise is usually low-spot;
- for dividend-paying calls, search from high spot downward because exercise is usually high-spot;
- if `S_i` is the last exercise node and `S_{i+1}` is the first continuation node, linear interpolation
  can be used as a tentative first method:
  `S_f approx S_i - P_i / (P_{i+1} - P_i) * (S_{i+1} - S_i)`.

Interpolation or threshold questions to decide later:

- what premium threshold counts as effectively zero;
- whether threshold should scale with `K`, grid size, or solver tolerance;
- whether to interpolate using raw premium or a smoothed premium curve;
- whether to suppress boundary reporting at time levels where no genuine exercise region is found;
- how to report near-maturity boundary behavior where payoff kinks are most visible.

Put boundary sanity:

- exercise region should be at low `S`;
- boundary should usually be below `K`;
- higher volatility often lowers the exercise boundary;
- boundary should become more stable with grid refinement.

Dividend-call boundary sanity:

- exercise region should be at high `S` when dividends are large enough;
- boundary may be above `K`;
- increasing `q` should generally expand the early-exercise region;
- `q = 0` should not produce a genuine early-exercise boundary.

Failure modes:

- boundary reversed by option type;
- threshold too large, causing false exercise regions;
- threshold too small, causing noisy missing boundaries;
- boundary dominated by `Smax`;
- jagged boundary caused by coarse grid;
- interpreting a premium-threshold artifact as a real financial boundary.

## 12. Greeks validation

Delta finite-difference check:

- compute Delta on interior nodes using a central finite difference:
  `Delta_i approx (U_{i+1} - U_{i-1}) / (2 Delta S)`;
- use one-sided differences only if needed near domain boundaries, and label them clearly;
- check that put Delta is generally between `-1` and `0`;
- check that call Delta is generally between `0` and `1`;
- report any bound violations.

Gamma finite-difference check:

- compute Gamma on interior nodes using:
  `Gamma_i approx (U_{i+1} - 2 U_i + U_{i-1}) / (Delta S)^2`;
- option values should usually be convex in `S`, so large negative Gamma values need investigation;
- numerical Gamma is more fragile than price or Delta.

Why Greeks may be unstable near payoff kink and free boundary:

- the payoff has a kink at `S = K`;
- the American free boundary separates exercise and continuation behavior;
- near maturity, both features become sharper;
- Gamma measures curvature, so it can spike or become noisy around non-smooth regions.

Whether masks or special reporting bands may be needed:

- use a kink band around `S = K`;
- use a boundary band around the extracted free boundary;
- use a strict interior mask for headline Delta/Gamma error reporting;
- still report stress behavior inside the masked bands separately, because those regions are
  financially important.

What should be reported before trusting Greek surfaces:

- Delta and Gamma plots for representative put and dividend-call cases;
- Delta bound violations;
- negative Gamma or convexity violations;
- Greek stability under grid refinement;
- Greek sensitivity to Rannacher smoothing;
- clear statement of any masked regions and why they were masked.

## 13. Rannacher smoothing decision

What Rannacher smoothing is:

- Rannacher smoothing replaces the first one or two Crank-Nicolson time steps with several smaller
  backward Euler half-steps;
- after those startup steps, the solver continues with Crank-Nicolson.

Why it may matter near payoff kinks:

- the terminal payoff is non-smooth at `S = K`;
- Crank-Nicolson can produce oscillations when started from non-smooth initial data;
- these oscillations may be especially visible in Delta and Gamma.

Whether it should be included in the first implementation or tested as an extension:

- recommended beginner approach: implement the first validation solver without hiding smoothing
  inside the method;
- once the unsmoothed solver is understood, add Rannacher smoothing as a clearly named option;
- compare unsmoothed and smoothed results for price, boundary, Delta, and Gamma;
- report whether smoothing was used in every result table.

Recommended approach for beginners:

- first make the plain CN/PSOR logic pass basic validation;
- then add Rannacher smoothing as an early extension if Greeks are noisy near maturity;
- do not use smoothing to cover up obstacle, complementarity, indexing, or boundary-condition bugs.

## 14. Proposed output files for the future implementation stage

Do not create these files during this planning stage. They are likely future files once implementation
begins:

- `src/solvers/black_scholes.py`
- `src/solvers/cn_psor.py`
- `src/diagnostics/obstacle.py`
- `src/diagnostics/complementarity.py`
- `src/boundaries/extraction.py`
- `src/greeks/finite_difference.py`
- `experiments/01_solver_validation.py`
- `reports/03_solver/solver_validation_report.md`
- `results/01_solver_validation/`

These names are proposed for clarity only. Final paths should follow the repository structure chosen
at implementation time.

## 15. Proposed result tables and figures

European closed-form error table:

- option type;
- parameters;
- grid size;
- maximum absolute error;
- RMSE;
- location of maximum error.

No-dividend American call error table:

- parameters;
- grid size;
- maximum absolute difference from European call;
- obstacle violation;
- whether any exercise boundary was detected.

Obstacle violation table:

- option type;
- parameters;
- grid size;
- maximum obstacle violation;
- location of maximum violation.

Complementarity residual table:

- option type;
- parameters;
- grid size;
- maximum complementarity residual;
- percentile residuals if useful.

Grid refinement table:

- coarse, medium, and fine grid results;
- price differences;
- boundary differences;
- Delta/Gamma differences in masked and unmasked regions;
- runtime and PSOR iteration counts.

Domain sensitivity table:

- `Smax`;
- target region price differences;
- boundary differences;
- Greek differences;
- qualitative conclusion.

Boundary curve plot:

- put boundary over time-to-maturity;
- dividend-call boundary over time-to-maturity;
- separate curves for selected `sigma` or `q` values.

Optional Delta/Gamma diagnostic plots:

- Delta surface;
- Gamma surface;
- Gamma near boundary;
- Gamma near payoff kink;
- masked versus unmasked Greek diagnostic comparison.

## 16. Acceptance criteria before moving to stress maps

Do not move to stress maps, dataset construction, or neural surrogates until all of the following are
true:

- European closed-form check passes in the target region;
- no-dividend American call check passes and does not show genuine early exercise;
- American put obstacle violation is near tolerance;
- complementarity residual is acceptable and understood;
- grid refinement stabilizes prices;
- grid refinement stabilizes boundary curves enough for interpretation;
- domain sensitivity shows that `Smax` is not distorting the target region;
- boundary behavior is financially plausible for both puts and dividend-paying calls;
- Delta and Gamma have been computed, plotted, and interpreted carefully;
- any Greek masks or special reporting bands are documented;
- humans have reviewed the validation tables and agreed that the solver is credible enough for
  risk-surface experiments.

All numerical thresholds are tentative until implementation evidence exists. The solver should not be
called validated simply because one example looks reasonable.

## 17. Beginner risk warnings

Common mistakes beginners may make:

- starting neural surrogate modelling before solver validation;
- treating American options like European options plus a small adjustment;
- forgetting that `U >= payoff` must hold everywhere;
- marching time in the wrong direction;
- mixing calendar time `t` and time-to-maturity `tau`;
- using the wrong sign for the dividend term;
- applying put boundary logic to calls or call boundary logic to puts;
- choosing `Smax` too small;
- ignoring boundary adjustments in the Crank-Nicolson right-hand side;
- stopping PSOR too early;
- using a relaxation parameter without checking convergence;
- extracting the boundary from visual plots instead of continuation premium;
- interpreting numerical threshold artifacts as real exercise boundaries;
- reporting only price error and ignoring boundary and Greeks;
- treating noisy Gamma near kinks as automatically a coding bug;
- hiding unstable regimes instead of documenting them;
- using solver labels for neural training before the validation gate is passed.

## 18. Human review checklist

Before allowing Codex or any developer to implement the solver, humans should check:

- Can the team explain the difference between European and American exercise?
- Can the team explain the payoff obstacle in plain language?
- Can the team explain why the no-dividend American call is a validation case?
- Are the option types in scope limited to American puts and dividend-paying American calls?
- Is `K = 1` accepted for the first normalized experiments?
- Are `Smax = 4K` and `S/K in [0.4, 1.8]` accepted as tentative starting defaults?
- Are the proposed grid sizes reasonable for the available compute time?
- Are obstacle violation and complementarity residual required in every validation report?
- Is boundary extraction based on continuation premium?
- Are Delta and Gamma treated as diagnostics that need careful masking and interpretation?
- Is Rannacher smoothing planned as a reported option rather than a hidden fix?
- Are all numerical thresholds marked tentative until implementation evidence exists?
- Has the team agreed not to build stress maps or neural labels until validation passes?
- Has a supervisor or knowledgeable reviewer approved moving from this plan to implementation?

## 19. Next recommended action

The next recommended action is:

1. Review this solver validation plan with the project team.
2. Confirm tentative defaults for `K`, `Smax`, target spot region, grid sizes, and validation
   thresholds.
3. Create the first solver implementation prompt.
4. Implement only the validation-oriented solver stage.
5. Do not implement the whole project, stress maps, dataset generation, or neural surrogates yet.

The next implementation prompt should ask for a small, testable CN/PSOR validation stage whose only
goal is to produce the tables and diagnostics described here.

## Self-check

- The plan defines how the solver will be tested before it is trusted.
- The plan prevents premature neural surrogate modelling.
- The plan explains pass/fail criteria in beginner-friendly language.
- The plan identifies major failure modes, including obstacle violations, complementarity failures,
  unstable boundaries, domain sensitivity, and fragile Greeks.
- The plan does not claim that the solver has already been validated.

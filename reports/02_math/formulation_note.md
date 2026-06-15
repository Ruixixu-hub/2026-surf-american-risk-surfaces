# SURF2026 Mathematical Formulation Note

## Source Basis

This note is based only on the current project materials:

- `reports/00_planning/planning_report.md`
- `reports/01_literature/literature_map.md`
- `docs/student_handout_full_report.pdf`
- `docs/Student_Methodology_FreeBoundary_Risk_Surfaces.pdf`

It is a bridge from the literature map to future solver implementation. It is not a proof, not a
solver manual, and not an implementation file.

## 1. Purpose of This Formulation Note

Before writing solver code, we need to know exactly what mathematical problem the code is
supposed to approximate. American option pricing is not ordinary curve fitting, because the
option holder can exercise before maturity. That early-exercise right creates a payoff obstacle,
an exercise region, a continuation region, and a free boundary.

The literature map explains why these ideas matter. This note translates those ideas into the
objects future code will need:

- the payoff functions;
- the Black-Scholes risk-neutral model with dividends;
- the American option variational inequality;
- the forward time-to-maturity form used by finite differences;
- the linear complementarity problem, or LCP, solved by CN/PSOR;
- boundary, continuation premium, and Greek diagnostics.

The goal is preparation. After reading this note, we should be able to explain what the future
solver must solve and how we will later check whether it is trustworthy.

## 2. Option Basics

An option is a contract whose value depends on an underlying asset, such as a stock.

Important notation:

- `S`: spot price, meaning the current underlying asset price.
- `K`: strike price, meaning the contract exercise price.
- `T`: maturity, meaning the final expiration time.
- `t`: calendar time before maturity.
- `tau = T - t`: time-to-maturity, meaning how much time remains before expiration.
- `r`: continuously compounded risk-free interest rate.
- `q`: continuous dividend yield on the underlying asset.
- `sigma`: volatility of the underlying asset.

For a put option, the holder has the right to sell the asset for `K`. Its payoff at exercise is

`Phi_put(S) = max(K - S, 0)`.

For a call option, the holder has the right to buy the asset for `K`. Its payoff at exercise is

`Phi_call(S) = max(S - K, 0)`.

The payoff is also called intrinsic value. It is what the option is worth if exercised immediately.

A European option can be exercised only at maturity. An American option can be exercised at
any time up to maturity. Therefore, an American option must be worth at least as much as its
immediate exercise payoff:

`option value >= payoff`.

This inequality is the payoff obstacle. It is the central constraint in this project.

## 3. Risk-Neutral Black-Scholes Setup

The project uses the dividend-adjusted Black-Scholes model. Under the risk-neutral measure,
the underlying asset follows

`dS_t = (r - q) S_t dt + sigma S_t dW_t`.

Beginner interpretation:

- `S_t` is the asset price at time `t`.
- `dW_t` represents random market movement.
- `sigma S_t dW_t` is the random volatility part.
- `(r - q) S_t dt` is the risk-neutral drift.
- `r` appears because future cash flows are discounted at the risk-free rate.
- `q` reduces the risk-neutral drift because continuous dividends are paid out of the asset.

"Risk-neutral" does not mean investors are actually risk-neutral. It means we price the option
using a mathematically adjusted probability model where expected growth is tied to `r` and `q`.
This lets the option price be written as a discounted expected payoff, with early exercise handled
through an optimal stopping choice.

In original calendar time, let `V(S,t)` be the American option value. A compact way to describe
the value is

`V(S,t) = sup over exercise times E[exp(-r(exercise_time - t)) Phi(S_exercise_time) | S_t = S]`.

This says: among all allowed future exercise times, choose the one with the largest discounted
expected payoff. We do not need the formal proof before coding; the important idea is that the
price chooses between exercising now and continuing.

## 4. American Option Variational Inequality

Define the Black-Scholes spatial operator

`L V = 0.5 sigma^2 S^2 V_SS + (r - q) S V_S - r V`.

The terms have the following roles:

- `0.5 sigma^2 S^2 V_SS`: curvature or diffusion term from volatility.
- `(r - q) S V_S`: drift term from the dividend-adjusted risk-neutral movement of `S`.
- `-r V`: discounting term.

For American options, we do not have one plain PDE everywhere. Instead, the value must satisfy
a variational inequality. In forward time-to-maturity notation, the source documents write it as

`U(S,tau) >= Phi(S)`,

`U_tau - L U >= 0`,

`(U(S,tau) - Phi(S)) (U_tau - L U) = 0`.

Here `U(S,tau)` is the forward-time value, explained in the next section.

Plain-language meaning of the three conditions:

1. `U >= Phi`: the American option value cannot be below immediate exercise value.
2. `U_tau - L U >= 0`: the value must be compatible with the continuation equation and the
   obstacle constraint.
3. `(U - Phi)(U_tau - L U) = 0`: at each point, at least one of the two gaps is zero.

The last condition is called complementarity. It means:

- If `U > Phi`, the option is worth more alive than exercised. This is the continuation region,
  and the Black-Scholes PDE holds there: `U_tau - L U = 0`.
- If `U = Phi`, exercising is optimal. This is the exercise region.

The free boundary is the moving curve that separates these two regions. It is "free" because its
location is not known in advance; it must be discovered as part of solving the option problem.

## 5. Time-to-Maturity Transformation

The original option value is usually written in calendar time as `V(S,t)`, where maturity happens
at `t = T`. Numerical solvers often prefer to start at maturity and march forward in remaining
time. For that, define

`tau = T - t`

and

`U(S,tau) = V(S,T - tau)`.

When `tau = 0`, the option is at maturity, so the value is just the payoff:

`U(S,0) = Phi(S)`.

As `tau` increases from `0` to `T`, the solver moves from the known payoff backward in calendar
time but forward in time-to-maturity. In the continuation region, the PDE becomes

`U_tau = L U`.

This forward-time form is the one used for the finite-difference and CN/PSOR discussion.

## 6. From Variational Inequality to Linear Complementarity Problem

To solve the problem numerically later, we will truncate the spot domain to

`S in [0, Smax]`

and use a time grid

`tau in [0, T]`.

At each grid point, finite differences approximate the derivatives `U_S`, `U_SS`, and `U_tau`.
After discretization, the continuous variational inequality becomes a linear complementarity
problem, or LCP.

For a Crank-Nicolson time step from `tau_n` to `tau_{n+1}`, the continuation equation has the
matrix form

`A U^{n+1} = b^n`

if there were no American exercise constraint. For the American option, we instead solve

`A U^{n+1} >= b^n`,

`U^{n+1} >= Phi`,

`(U^{n+1} - Phi)^T (A U^{n+1} - b^n) = 0`.

The usual Crank-Nicolson definitions are

`A = I - (Delta tau / 2) L_h`,

`b^n = [I + (Delta tau / 2) L_h] U^n`,

with boundary adjustments included in `b^n`.

Here `L_h` is the finite-difference version of the Black-Scholes operator `L`.

Why this matters for CN/PSOR:

- Crank-Nicolson gives the time-stepping equation in the continuation region.
- The LCP adds the American payoff obstacle.
- PSOR, projected successive over-relaxation, is used because it can solve the linear system
  while projecting values back above the payoff.

In plain language, PSOR keeps asking: "What would the linear equation suggest here, and is that
below payoff?" If the linear suggestion falls below payoff, projection pushes the value back to
the obstacle.

This note intentionally does not describe implementation details such as update loops, matrix
storage, relaxation tuning, or stopping criteria. Those belong in the later solver validation plan.

## 7. Boundary Conditions

Because the numerical domain is finite, the future solver must impose values at `S = 0` and
`S = Smax`.

### American Put

For an American put:

`U(0,tau) = K`.

If the asset price is zero, a put with strike `K` can be exercised for value `K`.

At the upper boundary:

`U(Smax,tau) approx 0`.

If the asset price is very high, a put is far out of the money and should be close to worthless.

### Dividend-Paying American Call

For an American call:

`U(0,tau) = 0`.

If the asset price is zero, the right to buy at positive strike has no value.

At the upper boundary, the methodology guide uses

`U(Smax,tau) approx max(Smax - K, Smax exp(-q tau) - K exp(-r tau))`.

This compares immediate exercise value with a large-spot European-style asymptotic value.
The exact effect of this boundary should later be checked through domain sensitivity, because
too small an `Smax` can distort the region we care about.

### No-Dividend American Call Validation

When `q = 0`, a standard American call should match the corresponding European call under the
usual Black-Scholes assumptions. Early exercise should not be genuinely optimal. This is a key
validation check because it tests whether the solver is creating artificial early exercise.

## 8. Continuation Premium

Define the continuation premium as

`P_cont(S,tau) = U(S,tau) - Phi(S)`.

For an American option,

`P_cont(S,tau) >= 0`.

The continuation premium measures how much extra value the option has from waiting instead of
exercising immediately.

Why it is important:

- `P_cont = 0` means value equals payoff, so the point is in the exercise region.
- `P_cont > 0` means continuation has extra value, so the point is in the continuation region.
- The free boundary is located where this premium transitions from zero to positive.

Why this matters later for neural surrogate modelling:

If a future model predicts price `U` directly, it might accidentally predict `U < Phi`, which is
financially invalid. The project documents therefore prefer predicting a nonnegative continuation
premium and reconstructing price as

`U_hat = Phi + P_hat_cont`,

where `P_hat_cont >= 0`.

That representation enforces the payoff obstacle by construction. The details of model design
belong later; the mathematical idea to remember now is that positive premium protects the core
American option constraint.

## 9. Free-Boundary Extraction

The free boundary is identified from the continuation premium.

At a fixed `tau`, inspect `P_cont(S,tau)` across spot values:

- exercise region: `P_cont = 0`;
- continuation region: `P_cont > 0`;
- boundary: transition between the two.

For an American put, exercise is typically optimal when `S` is low enough. The put boundary is
usually below the strike. Intuitively, if the stock is very low, the put is deep in the money, and
exercising can be better than waiting.

For a dividend-paying American call, early exercise can occur when `S` is high and dividends
are large enough. The call boundary can appear above the strike. Intuitively, exercising a call
early sacrifices remaining optionality, but it may let the holder capture the asset before
dividends reduce its value.

Uncertainty to carry forward: the exact premium threshold, interpolation rule, and boundary
stability criteria should be specified in the solver validation plan. This note only fixes the
mathematical idea: extract the boundary from continuation premium, not from visual guessing.

## 10. Greeks

Greeks measure option sensitivity. The two most important Greeks for this project are Delta
and Gamma.

Delta is

`Delta = partial U / partial S`.

It measures how much the option value changes when the spot price changes a little.

Gamma is

`Gamma = partial^2 U / partial S^2`.

It measures how quickly Delta changes as spot changes. Gamma is a curvature measure.

Why they matter:

- Delta is central for hedging and directional risk.
- Gamma shows where Delta changes quickly and where hedging is more unstable.
- Risk surfaces need price, boundary, Delta, and Gamma, not only price.

Why they can be unstable:

- The payoff has a kink at `S = K`.
- The American exercise boundary creates a change between continuation and exercise behavior.
- Near maturity, these non-smooth features become more pronounced.

So a large or noisy Gamma near the payoff kink or the free boundary is not automatically a coding
mistake. It may reflect the mathematical difficulty of the problem. Later diagnostics may need
masks or separate reporting bands near `S = K` and near the extracted free boundary.

Uncertainty to carry forward: the exact Greek mask width and whether to use smoothing should
be decided in the solver validation plan.

## 11. Solver Validation Requirements

Before using a solver to generate risk surfaces or neural labels, the project documents require a
validation ladder.

### European Closed-Form Check

First, disable the American projection logic and compare finite-difference prices with the
closed-form Black-Scholes European price. This checks the basic PDE discretization.

### No-Dividend American Call Check

For `q = 0`, the American call should match the European call. This checks that the solver does
not invent early exercise where the theory says it should not appear.

### American Put Obstacle Check

For an American put, confirm that

`U(S,tau) >= Phi_put(S)`

across the grid. The put is the core case where early exercise matters.

### Grid Refinement

Solve the same case on finer grids. Prices, boundaries, and Greeks should become more stable
as the grid is refined. The project does not require formal convergence proof at this stage, but
the numerical behavior should improve rather than jump unpredictably.

### Domain Sensitivity

Increase `Smax` and check whether the region of interest changes materially. If values or
boundaries in the target spot range change a lot, the domain was probably too small.

### Obstacle Violation

A useful diagnostic is

`max over grid points of max(Phi - U, 0)`.

This should be near the chosen numerical tolerance. Large obstacle violation means the solver is
breaking the American option constraint.

### Complementarity Residual

The complementarity residual checks whether the two LCP gaps are consistent:

- value gap: `U - Phi`;
- equation gap: `A U - b`.

Conceptually, at each interior grid node, at least one of these should be zero or close to zero.
The methodology guide measures this using a maximum over interior nodes and time steps of the
product of the value gap and equation gap.

This residual matters because a price can be above payoff and still fail the LCP logic.

## 12. Beginner Checklist Before Coding

Before writing solver code, we should be able to explain the following in our own words:

- What `S`, `K`, `T`, `tau`, `r`, `q`, and `sigma` mean.
- The payoff of a call and a put.
- The difference between European and American exercise.
- Why an American option value cannot be below payoff.
- What the continuation region and exercise region mean.
- What the free boundary separates.
- Why the time-to-maturity variable starts from the known payoff.
- Why discretization creates an LCP rather than an ordinary linear system.
- Why CN/PSOR is appropriate for the American obstacle problem.
- What boundary conditions are planned for puts and dividend-paying calls.
- Why no-dividend American calls are a validation case.
- Why continuation premium is useful for both boundary extraction and future positive-premium
  surrogate modelling.
- What Delta and Gamma measure.
- Why Greeks may be unstable near the payoff kink and free boundary.
- What obstacle violation and complementarity residual are checking.

The following can remain black-box temporarily:

- the full stochastic calculus derivation of the risk-neutral model;
- formal proofs of the Black-Scholes PDE;
- existence and uniqueness theory for variational inequalities;
- formal convergence proof for PSOR;
- advanced free-boundary regularity theory;
- neural-network architecture details.

## 13. Implementation Choices Deferred to the Solver Validation Plan

This note fixes the mathematical formulation, but it does not choose every numerical parameter.
The next plan should decide:

- baseline grid sizes in `S` and `tau`;
- baseline `Smax` and domain sensitivity cases;
- PSOR tolerance and relaxation parameter;
- whether Rannacher smoothing is part of the default solver;
- premium threshold and interpolation rule for boundary extraction;
- Greek finite-difference formulas and mask widths;
- validation thresholds for price error, obstacle violation, complementarity residual, boundary
  stability, and Greek stability.

These are implementation and validation choices, not formulation choices.

## 14. Next Recommended Action

The next recommended action is to prepare

`reports/03_solver/solver_validation_report.md`

or a solver validation plan before writing solver code.

That plan should define exact validation cases, tolerances, grid/domain checks, diagnostics, and
acceptance criteria. Only after that should the project move into CN/PSOR implementation.

## 15. Self-Check

This note is intended to satisfy four checks:

1. It should be understandable for accounting students with data, ML, and Python background.
2. It translates the project into mathematical objects: payoff, PDE operator, variational
   inequality, LCP, premium, boundary, and Greeks.
3. It prepares solver validation without jumping into coding.
4. It marks implementation-specific uncertainty instead of pretending all numerical choices are
   already settled.

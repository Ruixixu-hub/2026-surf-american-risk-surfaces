# SURF2026 American Option Risk Surfaces Literature Map

## Source Basis

This literature map is based on the current project planning sources:

- `reports/00_planning/planning_report.md`
- `docs/student_handout_full_report.pdf`
- `docs/Student_Methodology_FreeBoundary_Risk_Surfaces.pdf`

It is a learning and research-orientation document for the SURF2026 American option risk
surfaces project. It is not a solver design, not a model specification, and not a final paper-level
literature review.

## 1. Purpose of the Literature Map

The purpose of this map is to help us understand the research field before coding. American
option pricing is not just a normal regression problem. The option holder can exercise early, so
the price is constrained by the payoff, and the boundary between "exercise now" and "continue
holding" becomes a central object. If we start coding before understanding this structure, we may
build a solver or neural model that produces good-looking price plots while violating the main
financial logic.

This map helps us answer three preparation questions:

1. What problem are we actually studying?
2. Which literature layers support the project?
3. What must we understand before formulation, solver implementation, and neural surrogate
   modelling?

This differs from a final literature review. A final review in a paper must verify complete
bibliographic details, compare methods in a polished academic argument, and justify a precise
research gap. This map is earlier and more practical. It is a reading guide, concept map, and
claim-safety checklist. It tells us what to learn, what to treat as background, and how each
literature area connects to the future project workflow.

## 2. Project Positioning

This project studies American options. Unlike European options, which can be exercised only at
maturity, American options can be exercised before maturity. This early-exercise right creates a
free boundary: at each time-to-maturity, there is a region where exercise is optimal and a region
where continuation is optimal.

The central object is not only the option price. The project is about risk surfaces:

- price as spot, time, volatility, interest rate, dividend yield, and maturity change;
- Delta and Gamma, which describe sensitivity to the underlying asset price;
- the early-exercise boundary, which describes the exercise policy;
- stress maps showing how the boundary and Greeks change across regimes.

The intended computational workflow is classical first and neural second. The trusted benchmark
is Crank-Nicolson finite differences with projected successive over-relaxation, abbreviated
CN/PSOR. CN/PSOR matters because it is designed to solve the American option problem while
respecting the payoff obstacle. Neural networks enter later as constrained accelerators and
diagnostic tools, not as replacements for the benchmark.

This is therefore not a generic neural-network project. A generic project might ask: "Can a neural
network fit option prices?" This project asks a sharper computational finance question: "Can we
generate fast, stable, financially consistent American option price, boundary, and Greek surfaces
while preserving the free-boundary structure?" That difference matters. A model with low average
price error can still be unacceptable if it violates payoff, gives misleading boundary estimates, or
produces unreliable Greeks near the exercise frontier.

## 3. Four Literature Layers

### 3.1 Classical American Option Pricing

What this layer studies:

This layer studies American options as financial contracts with an early-exercise feature. It
introduces the basic pricing problem, the economic reason early exercise can be valuable, and the
main families of valuation methods: finite differences, simulation, optimal stopping, and bounds.

Why it matters for our project:

This is the finance foundation. We need it to understand why American options are different from
European options, why dividend-paying calls can have early exercise, and why the project cannot
be evaluated only by price RMSE. Without this layer, the free boundary may look like a technical
detail instead of the financial object that drives the whole project.

Key concepts:

- call and put payoff;
- strike, maturity, spot price, volatility, risk-free rate, and dividend yield;
- intrinsic value and time value;
- European option versus American option;
- early exercise and optimal stopping;
- continuation value;
- no-dividend American call theorem: without dividends, early exercise of a standard call is not
  optimal under the usual Black-Scholes assumptions;
- American put early exercise intuition;
- benchmark methods and reference prices.

Representative papers, topics, and keywords:

- Brennan and Schwartz, finite-difference valuation of American put options. The methodology
  guide lists: M. J. Brennan and E. S. Schwartz, "The valuation of American put options,"
  Journal of Finance, 1977.
- Longstaff and Schwartz, least-squares Monte Carlo for American options. The methodology
  guide lists their 2001 Review of Financial Studies paper.
- Haugh and Kogan, duality methods for American option pricing. The methodology guide lists
  their 2004 Operations Research paper.
- Textbook topics: American option pricing, optimal stopping, early exercise premium,
  American put-call differences, dividend effects.
- Useful search keywords: `American option early exercise`, `American put finite difference`,
  `least squares Monte Carlo American option`, `duality American option pricing`.

What we must understand before coding:

- Why American value must be at least payoff.
- Why early exercise creates a boundary, not just a price adjustment.
- Why American puts and dividend-paying American calls behave differently.
- Why a no-dividend American call is a key validation check.
- Why simulation methods such as Longstaff-Schwartz are useful background but not the main
  benchmark for this one-dimensional Black-Scholes free-boundary project.

What can remain background only:

- Full proofs of optimal stopping theory.
- Advanced martingale duality proofs.
- High-dimensional Monte Carlo implementation details.
- Market microstructure or real exchange exercise behavior.

How it supports the future paper contribution:

This layer justifies the project as a computational finance problem. It explains why a validated
American option benchmark is necessary before any neural model is trusted. It also gives the
paper a finance language: exercise policy, continuation value, risk surfaces, and economic stress
regimes.

### 3.2 Free-Boundary / Variational Inequality / LCP Methods

What this layer studies:

This layer studies the mathematical and numerical structure of American option pricing. In the
Black-Scholes setting with dividends, the option value satisfies a variational inequality rather
than a plain PDE. The value must stay above the payoff obstacle, the PDE holds in the
continuation region, and the exercise region is where value equals payoff. After discretization,
this becomes a linear complementarity problem, or LCP.

Why it matters for our project:

This is the solver foundation. CN/PSOR is not chosen because it is fashionable; it is chosen
because the American option problem has an obstacle constraint. The solver must enforce
complementarity, extract the boundary, and produce stable price and Greek surfaces. If we do not
understand this layer, we cannot design validation checks or know whether a numerical result is
financially plausible.

Key concepts:

- Black-Scholes PDE with continuous dividends;
- time-to-maturity transformation;
- payoff obstacle;
- variational inequality;
- complementarity conditions;
- continuation region and exercise region;
- free boundary;
- LCP after finite-difference discretization;
- Crank-Nicolson time stepping;
- PSOR projection;
- boundary conditions for puts and dividend-paying calls;
- Rannacher smoothing for payoff kinks;
- Delta and Gamma finite-difference diagnostics;
- obstacle violation and complementarity residual.

Representative papers, topics, and keywords:

- Classical finite-difference American option work, including Brennan and Schwartz as listed in
  the methodology guide.
- Computational finance textbook chapters on finite differences for Black-Scholes equations.
- Numerical analysis topics: obstacle problem, variational inequality, LCP, projected SOR,
  free-boundary extraction.
- Useful search keywords: `American option variational inequality`, `Black-Scholes obstacle
  problem`, `linear complementarity problem option pricing`, `Crank-Nicolson PSOR American
  option`, `Rannacher smoothing option pricing`, `free boundary American put`.

What we must understand before coding:

- The American option value cannot fall below payoff.
- In the continuation region, the Black-Scholes PDE applies.
- In the exercise region, the value equals payoff.
- The free boundary separates those regions.
- PSOR enforces the obstacle by projecting each update back above payoff.
- Boundary extraction should use continuation premium, not visual guessing.
- Greeks are delicate near payoff kinks and near the free boundary.
- Validation must include closed-form European checks, the no-dividend American call check,
  obstacle checks, grid refinement, and domain sensitivity.

What can remain background only:

- Full existence and uniqueness theory for variational inequalities.
- Formal convergence proofs for PSOR.
- Advanced free-boundary regularity theory.
- Alternative methods such as front-fixing, penalty methods, and high-order schemes, unless the
  supervisor asks for them later.

How it supports the future paper contribution:

This layer supports the benchmark claim. The future paper can only claim reliable risk surfaces
if the classical solver is validated and the boundary/Greek diagnostics are understood. It also
provides the language for failure checks: obstacle violation, complementarity error, boundary
instability, and Gamma concentration.

### 3.3 Scientific Machine Learning / PINNs / DGM

What this layer studies:

This layer studies neural methods for differential equations. Physics-informed neural networks,
or PINNs, train neural networks using equation residuals and boundary/initial conditions. The
Deep Galerkin Method, or DGM, is another mesh-free deep learning approach for PDEs. These
methods are attractive because they may reduce grid dependence and handle higher-dimensional
problems.

Why it matters for our project:

This layer explains why neural PDE methods are tempting but risky for American options. A
plain PDE residual is natural in the continuation region, but American options also have an
exercise region where the continuation PDE should not be forced in the same way. The obstacle
and free boundary make the problem non-smooth and constrained. This is why the project should
be cautious about broad "PINN solves American options" claims.

Key concepts:

- PDE residual loss;
- boundary and initial condition losses;
- automatic differentiation;
- mesh-free collocation points;
- PINNs;
- DGM;
- constraint enforcement;
- non-smooth payoff;
- residuals in the continuation region versus exercise region;
- why visually smooth plots can hide boundary or obstacle errors.

Representative papers, topics, and keywords:

- Raissi, Perdikaris, and Karniadakis on physics-informed neural networks. The methodology
  guide lists their 2019 Journal of Computational Physics paper.
- Sirignano and Spiliopoulos on DGM. The methodology guide lists their 2018 Journal of
  Computational Physics paper.
- Topics on PINN limitations for non-smooth solutions, constrained PDEs, and financial PDEs.
- Useful search keywords: `physics informed neural networks`, `PINN financial PDE`, `Deep
  Galerkin Method financial PDE`, `PINN obstacle problem`, `PINN variational inequality`,
  `PINN American option`.

What we must understand before coding:

- A neural network can fit a surface without respecting financial constraints.
- Penalizing violations in a loss may not guarantee constraint satisfaction.
- PDE residuals should be applied carefully; the Black-Scholes PDE holds in the continuation
  region, not as a plain equation everywhere.
- Automatic differentiation through a price network does not automatically produce reliable
  Delta or Gamma.
- Held-out regime tests matter more than random point splits for risk-surface generalization.

What can remain background only:

- Full theoretical analysis of PINN convergence.
- Advanced sampling strategies for high-dimensional PDEs.
- Deep PDE solver architectures unrelated to the American obstacle problem.
- Inverse problem PINNs, unless later market calibration is added.

How it supports the future paper contribution:

This layer helps us position the neural part as constrained acceleration, not as an unfocused
neural PDE solver. It also gives the future paper a reason to compare unconstrained or
penalty-only neural models against structure-preserving designs.

### 3.4 Deep Learning for American Options and Risk-Surface Applications

What this layer studies:

This layer studies neural methods specifically for American options and for the risk-surface
objects that matter in this project. Some methods learn exercise policies or continuation values.
Others use neural networks to represent the price, boundary, or both. In our project, the most
important question is not just price accuracy; it is whether neural tools can preserve obstacle
constraints, recover boundaries, support Greek diagnostics, and accelerate scenario evaluation.

Why it matters for our project:

This is the contribution-positioning layer. The methodology guide warns that a plain PINN for an
American put is no longer a strong contribution by itself. The distinctive project direction is a
validated, modular, structure-preserving workflow for price, boundary, and Delta diagnostics.

Key concepts:

- deep optimal stopping;
- learned continuation value;
- learned exercise policy;
- neural price surrogate;
- continuation-premium representation;
- direct boundary head;
- bounded Delta head;
- regime split;
- boundary-aware weighting;
- stress-regime testing;
- PSOR spot checks outside the training envelope;
- claim repair when a model fails.

Representative papers, topics, and keywords:

- Becker, Cheridito, and Jentzen on deep optimal stopping. The methodology guide lists their
  2019 Journal of Machine Learning Research paper.
- Becker, Cheridito, and Jentzen on pricing and hedging American-style options with deep
  learning. The methodology guide lists their 2020 Journal of Risk and Financial Management
  paper.
- Gatta, Schiano Di Cola, Giampaolo, Piccialli, and Cuomo on meshless methods for American
  option pricing through PINNs. The methodology guide lists a 2023 Engineering Analysis with
  Boundary Elements paper.
- Nwankwo, Umeorah, Ware, and Dai on a deep-learning free-boundary framework. The
  methodology guide lists a 2024 Computational Economics paper.
- The methodology guide also lists recent fractional Black-Scholes and jump-diffusion
  process-informed neural-network papers. These should be treated as reading leads until their
  details are checked directly.
- Useful search keywords: `deep optimal stopping American option`, `deep learning American
  option pricing`, `neural free boundary American option`, `American option risk surface`,
  `American option Greeks neural network`, `continuation premium neural surrogate`.

What we must understand before coding:

- Why the final workflow should be modular: price/premium surrogate, boundary diagnostic
  head, and bounded Delta head.
- Why predicting nonnegative continuation premium can enforce the payoff obstacle by
  construction.
- Why a boundary threshold proxy may be useful as a rough diagnostic but weak as a precise
  boundary method.
- Why Delta should be validated directly and may need a separate supervised head.
- Why neural results must be checked against fresh PSOR solves.

What can remain background only:

- Full architecture details from every deep American option paper.
- High-dimensional optimal stopping methods beyond the current one-dimensional
  Black-Scholes setup.
- Stochastic volatility, jump diffusion, rough volatility, and real market calibration, unless the
  project scope expands.

How it supports the future paper contribution:

This layer helps define the likely contribution: not "neural networks replace finite differences,"
but "a validated structure-preserving workflow for American option risk surfaces." It also gives
the future paper a disciplined claim structure: ready claims, restricted claims, rejected claims,
and evidence for each.

## 4. Cross-Layer Synthesis

The four layers connect in a staged way.

Layer 1 defines the financial problem: American options have early exercise, so price depends on
both payoff and continuation value. Layer 2 translates that financial problem into the
free-boundary, variational inequality, and LCP language needed for CN/PSOR. Layer 3 explains
why neural PDE methods are attractive but must be treated carefully because the American
option problem is constrained and non-smooth. Layer 4 shows how recent deep-learning work for
American options creates a space for a more careful risk-surface contribution.

The project gap is therefore not a missing price formula. In one-dimensional Black-Scholes
settings, American options can be priced with established numerical methods. The harder and
more interesting project is to generate reliable risk surfaces under changing regimes while
preserving financial structure:

- price must stay above payoff;
- the early-exercise boundary must be recovered and audited;
- Delta and Gamma must be diagnosed near kinks and exercise frontiers;
- neural acceleration must be faster without hiding constraint violations;
- claims must be narrowed when a model fails.

This points to a cautious contribution:

A validated CN/PSOR benchmark plus a constrained, modular neural acceleration workflow for
American option price, boundary, and Greek risk-surface diagnostics.

## 5. Recommended Reading Order

### Minimum reading path

Use this path if time is limited and the goal is to prepare for formulation and solver validation:

1. Project planning report.
2. Both project PDFs.
3. Introductory option pricing notes on calls, puts, payoff, intrinsic value, and time value.
4. One American option chapter from a computational finance textbook.
5. One explanation of the Black-Scholes PDE with dividends.
6. One explanation of variational inequalities or obstacle problems for American options.
7. CN/PSOR notes or examples for American puts.
8. Short overview of PINNs and DGM, mainly to understand why they are not the first step.

### Full reading path

Use this path to prepare for a formal literature review and later paper writing:

1. Minimum path above.
2. Brennan and Schwartz on finite-difference American put valuation.
3. Longstaff and Schwartz on least-squares Monte Carlo as a contrast method.
4. Haugh and Kogan on duality bounds as background.
5. Numerical treatment of LCPs, PSOR, and complementarity diagnostics.
6. Rannacher smoothing and finite-difference Greeks near non-smooth payoffs.
7. Raissi, Perdikaris, and Karniadakis on PINNs.
8. Sirignano and Spiliopoulos on DGM.
9. Becker, Cheridito, and Jentzen on deep optimal stopping.
10. Recent American-option PINN/free-boundary/process-informed papers listed in the
    methodology guide, after checking their bibliographic details directly.
11. Papers or notes on risk surfaces, stress maps, Greeks, and model validation in computational
    finance.

### What to read before `formulation_note.md`

Before creating `reports/02_math/formulation_note.md`, students should understand:

- American versus European exercise rights;
- payoff functions for calls and puts;
- the meaning of risk-neutral pricing at a high level;
- Black-Scholes PDE terms: diffusion, drift, discounting;
- time-to-maturity transformation;
- payoff obstacle;
- continuation region, exercise region, and free boundary;
- continuation premium as value minus payoff.

### What to read before solver implementation

Before coding CN/PSOR, students should understand:

- finite-difference grids over spot and time;
- boundary conditions for American puts and dividend-paying calls;
- Crank-Nicolson as a time-stepping method;
- why the American option discretization becomes an LCP;
- what PSOR is meant to enforce;
- obstacle violation and complementarity residual;
- grid refinement and domain sensitivity;
- Delta and Gamma finite-difference formulas;
- why Rannacher smoothing may matter near the payoff kink.

### What to read before neural surrogate modelling

Before neural modelling, students should understand:

- why labels must come from a validated solver;
- why random point splits can exaggerate performance;
- why held-out regime splits are more important;
- continuation-premium prediction and payoff reconstruction;
- direct boundary heads versus threshold boundary proxies;
- bounded Delta heads and Delta validation;
- obstacle, monotonicity, convexity, and Delta-bound diagnostics;
- why PSOR spot checks remain necessary outside the training envelope.

## 6. Knowledge Pickup Checklist

### Must understand before formulation

- Payoff of calls and puts.
- Strike, spot, maturity, volatility, rate, and dividend yield.
- Difference between European and American options.
- Basic idea of risk-neutral valuation.
- Why early exercise creates an exercise region and continuation region.
- What "free boundary" means in plain language.
- What the payoff obstacle means: American value cannot be below payoff.
- What Delta and Gamma measure.
- Why American option pricing is not just ordinary curve fitting.

### Must understand before CN/PSOR coding

- Black-Scholes PDE with dividends at a conceptual level.
- Forward time-to-maturity variable.
- Variational inequality and LCP meaning.
- Crank-Nicolson purpose.
- PSOR projection purpose.
- Boundary conditions for put and dividend-paying call cases.
- Continuation premium and boundary extraction.
- Obstacle and complementarity diagnostics.
- Grid refinement and domain sensitivity checks.
- Greek instability near payoff kinks and exercise boundaries.
- No-dividend American call validation check.

### Must understand before neural surrogate modelling

- Why a plain price network can violate payoff.
- Why a loss penalty may not guarantee financial consistency.
- Why a nonnegative continuation-premium output is structurally useful.
- Why price, boundary, and Delta may need separate model components.
- Why autograd Delta must be tested rather than assumed correct.
- Why Gamma labels near kinks and boundaries may need masks or special diagnostics.
- Why regime splits are essential.
- Why neural acceleration should be paired with PSOR audit checks.
- How to weaken or reject claims when diagnostics fail.

## 7. Search Keywords and Paper-Finding Strategy

High-priority keywords:

- `American option free boundary`
- `variational inequality Black-Scholes`
- `linear complementarity problem option pricing`
- `Crank-Nicolson PSOR American option`
- `American put finite difference PSOR`
- `dividend paying American call free boundary`
- `Rannacher smoothing option pricing`
- `American option Greeks finite difference`
- `PINN American option`
- `Deep Galerkin Method financial PDE`
- `deep optimal stopping`
- `American option Greeks risk surface`
- `neural free boundary American option`
- `continuation premium American option neural network`

Medium-priority keywords:

- `least squares Monte Carlo American options`
- `duality American option pricing`
- `obstacle problem neural network`
- `physics informed neural network variational inequality`
- `American option stress testing`
- `option risk surface Delta Gamma`
- `shape constrained neural network option pricing`

Lower-priority background keywords:

- `Heston American option free boundary`
- `jump diffusion American option neural network`
- `rough volatility American option`
- `market calibration American options`
- `high dimensional American option deep learning`

Paper-finding strategy:

1. Start with project-listed anchor papers, because they are already aligned with the methodology
   guide.
2. Use textbook chapters to learn concepts before reading dense papers.
3. For each paper, record what role it plays: finance foundation, solver method, neural PDE
   method, American-option neural method, or application/risk-surface support.
4. Do not collect papers only because they mention neural networks. Keep papers only if they help
   explain price, boundary, Greeks, constraints, stress regimes, or validation.
5. Before formal paper writing, verify every bibliographic detail directly from the paper or a
   trusted database.

## 8. Implementation Relevance

This literature map will guide later implementation without starting implementation now.

Solver design:

- The free-boundary and LCP literature explains why the solver must enforce the payoff obstacle.
- The classical American option literature identifies which cases should be used as sanity checks.

Validation checks:

- European closed-form comparisons check the finite-difference machinery.
- The no-dividend American call theorem checks early-exercise logic.
- Obstacle violation and complementarity residual check the American constraint.
- Grid refinement and domain sensitivity check numerical robustness.

Boundary extraction:

- The continuation-premium view suggests extracting the boundary from where value leaves the
  payoff.
- The literature warns that boundary claims should be audited directly, not inferred only from
  price plots.

Greek diagnostics:

- Delta and Gamma are central risk quantities, but they are least stable near payoff kinks and
  free boundaries.
- The map prepares us to report instability honestly instead of treating every spike as a coding
  bug.

Stress maps:

- Classical finance intuition tells us which regimes matter: high volatility, high dividend yield,
  different rates, and different maturities.
- Stress maps should answer financial questions, not only produce colorful plots.

Dataset construction:

- Labels should come only from a validated solver.
- Regime splits should withhold whole parameter combinations.
- Datasets should include price, premium, payoff, boundary, exercise indicator, Greek masks,
  boundary masks, and weights if the later design uses them.

Neural surrogate design:

- The SciML layer warns that PDE residuals and penalties are not enough by themselves.
- The American-option neural layer supports modular components: positive-premium price,
  direct boundary diagnostics, and bounded Delta diagnostics.

Claim audit:

- The literature map supports a conservative claim style.
- A successful paper can claim fast, structure-preserving risk-surface diagnostics.
- It should not claim that a single neural network universally solves American option pricing,
  boundaries, and Greeks.

## 9. Open Questions Before Implementation

The following questions should be clarified before coding starts:

1. Exact option universe: American puts only at first, or both puts and dividend-paying calls from
   the first solver report?
2. Parameter envelope: what ranges of volatility, rate, dividend yield, maturity, and spot should
   define the first study?
3. Strike normalization: should all experiments normalize to `K = 1` until market data is added?
4. Boundary extraction tolerance: what premium threshold or interpolation rule will define the
   reported boundary?
5. Greek masking rules: how wide should the masks be near the payoff kink and free boundary?
6. Validation thresholds: what error levels count as passing for price, obstacle violation,
   complementarity residual, boundary stability, and Greeks?
7. Grid/domain settings: what baseline grid and `Smax` should be used before sensitivity checks?
8. Rannacher smoothing: should smoothing be part of the default solver validation or a later
   comparison?
9. Dataset split design: which parameter regimes will be held out for testing?
10. Neural success criteria: what price RMSE, boundary MAE, Delta RMSE, speedup, and violation
    levels are required before neural results can be included in the main paper?
11. Claim audit format: how will ready, restricted, and rejected claims be documented?
12. Supervisor approval gate: what must be shown before moving from literature and formulation
    to solver implementation?

## 10. Limitations of This Literature Map

This document is not a complete final literature review. It does not prove that the listed papers
are sufficient for formal publication, and it does not verify every bibliographic detail beyond the
project methodology guide. Recent papers listed in the methodology guide should be checked
directly before they are used in a manuscript.

The map is also not a mathematical derivation. It intentionally explains difficult topics in
beginner-friendly language for accounting students with data analysis, Python, machine learning,
financial data analysis, and research-writing background. Formal definitions, notation, and
equations should be developed next in the formulation note.

Finally, this map does not decide final paper claims. It prepares the claim structure, but claims
must be earned later through solver validation, stress experiments, surrogate tests, and explicit
failure logs.

## 11. Next Recommended Action

The next recommended action is to create:

`reports/02_math/formulation_note.md`

That document should explain payoff, Black-Scholes dynamics, the PDE, the variational
inequality, the LCP, continuation premium, free-boundary meaning, and Greek definitions in
beginner-friendly language. It should be completed before any solver implementation begins.

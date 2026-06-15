# Ticket 01 Study Report: Black-Scholes Utilities

Date: 2026-06-15

## 1. Role of This Ticket in the Whole Project

Ticket 1 is the first coding ticket after the setup check. Its job is intentionally small:
build reliable payoff functions and European Black-Scholes closed-form pricing utilities before
the project starts finite-difference or CN/PSOR work.

This order matters. Later tickets will build a numerical solver that approximates option values on
a grid. A numerical solver is useful only if we can compare it against trusted reference values.
For European options under the Black-Scholes model with continuous dividends, closed-form formulas
already exist. That makes European calls and puts an ideal benchmark for the future solver's first
validation step.

In plain language, Ticket 1 gives us two kinds of building blocks:

- payoff functions, which describe what an option is worth if exercised at a given spot price;
- European closed-form prices, which give analytic benchmark values before any American early
  exercise logic is introduced.

The solver roadmap says Ticket 1 must come before finite differences because later failures should
be easy to diagnose. If a future Crank-Nicolson result disagrees with a European closed-form price,
we need confidence that the closed-form comparison is correct. Otherwise we would not know whether
the bug is in the numerical grid, the PDE operator, the boundary conditions, or the analytic
benchmark itself.

This ticket does not validate the future CN/PSOR solver. It prepares one trusted analytic layer
that the solver validation plan will use later, especially in the European closed-form check.

## 2. Finance and Mathematical Background

An option is a contract whose value depends on an underlying asset price, called `S`. The strike
price, called `K`, is the exercise price written into the contract.

A call option gives the holder the right to buy the asset for `K`. If the market spot price is
above `K`, the right to buy cheaply is valuable. If the spot is below `K`, exercising the call
would not help.

The call payoff is:

```text
Phi_call(S) = max(S - K, 0)
```

A put option gives the holder the right to sell the asset for `K`. If the market spot price is
below `K`, the right to sell at the higher strike is valuable. If the spot is above `K`, exercising
the put would not help.

The put payoff is:

```text
Phi_put(S) = max(K - S, 0)
```

A European option can be exercised only at maturity. In this ticket, `T` means time to maturity.
When `T = 0`, the option is already at maturity, so a European option price must equal its payoff.

The Black-Scholes model with continuous dividends uses:

- `S`: current spot price;
- `K`: strike price;
- `T`: time to maturity;
- `r`: continuously compounded risk-free interest rate;
- `q`: continuous dividend yield;
- `sigma`: volatility;
- `N(x)`: standard normal cumulative distribution function.

The continuous dividend yield `q` reduces the present value of the stock leg because dividends are
paid out of the underlying asset over time. The risk-free rate `r` discounts the future strike
payment or receipt back to present value.

For `T > 0` and `sigma > 0`, define:

```text
d1 = [ln(S / K) + (r - q + 0.5 * sigma^2) * T] / [sigma * sqrt(T)]
d2 = d1 - sigma * sqrt(T)
```

The European call price with continuous dividends is:

```text
C = S * exp(-qT) * N(d1) - K * exp(-rT) * N(d2)
```

The European put price with continuous dividends is:

```text
P = K * exp(-rT) * N(-d2) - S * exp(-qT) * N(-d1)
```

Plain-language interpretation:

- `S * exp(-qT)` is the dividend-adjusted present value of the stock part.
- `K * exp(-rT)` is the discounted strike part.
- `N(d1)` and `N(d2)` are probability-weighted terms from the Black-Scholes model.
- The call formula compares the value of receiving the asset with paying the strike.
- The put formula compares the value of receiving the strike with giving up the asset.

The call and put prices should also satisfy put-call parity with continuous dividends:

```text
C - P = S * exp(-qT) - K * exp(-rT)
```

This identity is a powerful consistency check. It says that a European call minus a European put
with the same `S`, `K`, `T`, `r`, `q`, and `sigma` should equal a simple discounted stock-minus-strike
position. If the implemented call and put formulas violate this identity, at least one formula is
wrong.

This ticket is not American option solving yet. American options can be exercised before maturity,
which creates the payoff obstacle, continuation region, exercise region, and free boundary
discussed in the formulation note. The functions in this ticket do not choose early exercise times
and do not solve a variational inequality or linear complementarity problem.

## 3. Design Decisions

The package namespace `src/american_risk_surfaces/` is used because Ticket 0 recommended a `src`
layout with one explicit project namespace. This makes imports clear:

```python
from american_risk_surfaces.solvers.black_scholes import european_call_price
```

This is safer than importing from a bare package name such as `solvers`, which could collide with
another package or script name later.

`unittest` is used instead of `pytest` because Ticket 0 found that `pytest` was not installed in
the checked Python runtimes. `unittest` is part of the Python standard library, so it lets the
project start testing without adding `pyproject.toml`, `requirements.txt`, or any dependency
installation step.

SciPy is avoided for now because Ticket 0 found that SciPy availability differs between runtimes.
The only SciPy-like thing needed in this ticket is the standard normal CDF `N(x)`. Python's
standard `math.erf` function is enough for this small utility step because:

- the normal CDF can be computed as `N(x) = 0.5 * [1 + erf(x / sqrt(2))]`;
- the tests here check identities and basic financial behavior, not high-performance statistics;
- avoiding SciPy keeps the first code ticket portable.

Scalar and NumPy array inputs are both supported. Scalar inputs are convenient for simple examples,
reports, and selected spot checks. Array inputs are important because later validation will compare
many spot values across grids or vectors. Returning a Python `float` for scalar `S` and a NumPy
array for array-like `S` makes the API natural in both situations.

The edge cases `T = 0` and `sigma = 0` are handled explicitly because the usual `d1` and `d2`
formulas divide by `sigma * sqrt(T)`. At maturity, the correct answer is exactly payoff. With zero
volatility and positive maturity, the stock path is deterministic under the model, so the option
price becomes a discounted deterministic payoff:

```text
call = max(S * exp(-qT) - K * exp(-rT), 0)
put  = max(K * exp(-rT) - S * exp(-qT), 0)
```

Handling these cases explicitly prevents divide-by-zero behavior and makes the finance meaning
clear.

## 4. Implementation Record

Four source/test files were created for Ticket 1:

- `src/american_risk_surfaces/__init__.py`
- `src/american_risk_surfaces/solvers/__init__.py`
- `src/american_risk_surfaces/solvers/black_scholes.py`
- `tests/test_black_scholes.py`

The top-level `__init__.py` marks `american_risk_surfaces` as the project package. The solvers
`__init__.py` marks the `solvers` subpackage. These files are small, but they make imports
predictable when tests run with `PYTHONPATH=src`.

`black_scholes.py` contains the actual payoff and European pricing utilities. It exposes four
public functions and keeps validation and helper logic private.

`tests/test_black_scholes.py` contains focused `unittest` tests for the public behavior.

### call_payoff(S, K)

Purpose: compute the immediate exercise payoff of a call option.

Inputs:

- `S`: nonnegative spot price, either a scalar or array-like object;
- `K`: positive strike price.

Output:

- `max(S - K, 0)`, returned as a float for scalar `S` or a NumPy array for array-like `S`.

Important edge cases:

- if `S` is below or equal to `K`, the payoff is zero;
- if `S` is negative, the function raises `ValueError`;
- if `K <= 0`, the function raises `ValueError`.

Later use: this payoff will define terminal values for calls and will also support future American
obstacle checks for call cases.

### put_payoff(S, K)

Purpose: compute the immediate exercise payoff of a put option.

Inputs:

- `S`: nonnegative spot price, either a scalar or array-like object;
- `K`: positive strike price.

Output:

- `max(K - S, 0)`, returned as a float for scalar `S` or a NumPy array for array-like `S`.

Important edge cases:

- if `S` is above or equal to `K`, the payoff is zero;
- if `S` is negative, the function raises `ValueError`;
- if `K <= 0`, the function raises `ValueError`.

Later use: this payoff will define terminal values for puts and will support the future American
put obstacle check `U >= Phi_put`.

### european_call_price(S, K, T, r, q, sigma)

Purpose: compute the European call price under the dividend-adjusted Black-Scholes formula.

Inputs:

- `S`: nonnegative spot price, scalar or array-like;
- `K`: positive strike;
- `T`: nonnegative time to maturity;
- `r`: continuously compounded risk-free rate;
- `q`: continuous dividend yield;
- `sigma`: nonnegative volatility.

Output:

- European call price, returned as a float for scalar `S` or a NumPy array for array-like `S`.

Important edge cases:

- `T = 0` returns `call_payoff(S, K)`;
- `sigma = 0` and `T > 0` returns the deterministic discounted call payoff;
- invalid `S`, `K`, `T`, or `sigma` values raise `ValueError`.

Later use: this function will be a benchmark for future European finite-difference call validation
and for the no-dividend American call check.

### european_put_price(S, K, T, r, q, sigma)

Purpose: compute the European put price under the dividend-adjusted Black-Scholes formula.

Inputs:

- `S`: nonnegative spot price, scalar or array-like;
- `K`: positive strike;
- `T`: nonnegative time to maturity;
- `r`: continuously compounded risk-free rate;
- `q`: continuous dividend yield;
- `sigma`: nonnegative volatility.

Output:

- European put price, returned as a float for scalar `S` or a NumPy array for array-like `S`.

Important edge cases:

- `T = 0` returns `put_payoff(S, K)`;
- `sigma = 0` and `T > 0` returns the deterministic discounted put payoff;
- invalid `S`, `K`, `T`, or `sigma` values raise `ValueError`.

Later use: this function will be the main benchmark for future European put finite-difference
validation, which matters because American puts are a core early-exercise case later.

## 5. Testing Strategy

The tests are deliberately focused on analytic utility behavior. They do not test a solver.

Payoff tests check call and put values below, at, and above the strike. These tests prove that the
basic option payoff signs are correct. They do not prove anything about time value, discounting, or
volatility.

Scalar and array behavior tests check that scalar inputs return Python floats and NumPy arrays
preserve shape. These tests prove that the functions are convenient for both one-point examples
and vectorized validation. They do not prove that every possible array-like object is ideal for
performance.

The `T = 0` payoff equality test checks that European prices equal payoff at maturity. This proves
that the maturity boundary is handled explicitly. It does not validate prices for positive maturity.

The `sigma = 0` deterministic discounted payoff test checks the no-randomness case. It proves that
the code avoids the `d1`/`d2` divide-by-zero problem and uses the correct discounted deterministic
logic. It does not test stochastic volatility behavior, because stochastic volatility is outside
this project stage.

The put-call parity test checks:

```text
C - P = S * exp(-qT) - K * exp(-rT)
```

This proves that the call and put formulas are internally consistent for continuous dividends. It
does not prove that every market convention or every model extension is supported.

Nonnegative price sanity checks verify that calls and puts are not negative for a simple vector
example. Negative European option values would be financially invalid. These checks do not prove
full arbitrage-free behavior in every possible parameter regime.

Monotonicity checks verify that, in a simple vector example, call prices increase with `S` and put
prices decrease with `S`. This matches basic financial intuition. These tests do not replace a full
mathematical proof of monotonicity.

Invalid input checks verify that clearly invalid values raise `ValueError`: negative spot,
nonpositive strike, negative maturity, and negative volatility. These tests prove that obvious
domain errors are not silently priced. They do not define a full production-grade validation
policy for every unusual input type.

## 6. Test Results and Interpretation

Exact test command:

```bash
PYTHONPYCACHEPREFIX=/private/tmp/codex_pycache PYTHONPATH=src python3 -m unittest discover -s tests
```

Final result:

```text
Ran 9 tests in 0.002s

OK
```

In this context, "Ran 9 tests OK" means the Ticket 1 unit tests passed for the implemented payoff
and European Black-Scholes utility functions. It means the tested payoff behavior, scalar/array
behavior, maturity edge case, zero-volatility edge case, put-call parity identity, basic sanity
checks, and invalid input checks all behaved as expected.

It does not mean the future finite-difference solver is validated. No spot grid, time grid,
Black-Scholes finite-difference operator, boundary adjustment, Crank-Nicolson time step, PSOR
projection, American obstacle enforcement, boundary extraction, or Greek calculation was tested in
Ticket 1.

The correct interpretation is narrow but important: the project now has analytic utilities that are
credible enough to use as benchmark functions in later validation tickets.

## 7. Limitations

This ticket only validates analytic utility behavior.

It does not include:

- finite-difference grid construction;
- Black-Scholes finite-difference operator setup;
- Crank-Nicolson time stepping;
- PSOR projection;
- American option obstacle enforcement;
- continuation premium calculation;
- boundary extraction;
- Delta or Gamma calculations;
- numerical solver validation;
- experiments, datasets, stress maps, or neural-network files.

The normal-CDF implementation uses `math.erf` through NumPy vectorization. That is suitable for this
small dependency-light utility layer, but it is not meant to be a high-performance replacement for
scientific statistics libraries in a larger production system.

The tests use a small number of representative cases. They are designed to catch common formula and
edge-case mistakes, not to certify every possible input regime.

## 8. Lessons Learned / Study Notes

The first lesson is that payoff is the foundation. Before discussing PDEs, free boundaries, or
neural networks, students should be able to draw and explain `max(S - K, 0)` for a call and
`max(K - S, 0)` for a put.

The second lesson is that European and American options are different validation tools. European
options have closed-form Black-Scholes prices in this model, so they are useful for checking a
numerical PDE solver. American options usually do not have the same simple closed-form solution
because early exercise creates an obstacle problem.

The third lesson is that dividends matter. The dividend yield `q` changes the stock leg through
`S * exp(-qT)` and appears in `d1` through `(r - q)`. A wrong sign on `q` is a common source of
incorrect option prices and later solver errors.

The fourth lesson is that edge cases are not side issues. At `T = 0`, price equals payoff. At
`sigma = 0`, the uncertain stock movement disappears. If code ignores those cases, it may produce
division-by-zero warnings or misleading values exactly where future validation needs clarity.

The fifth lesson is that a passing unit test suite should be interpreted carefully. Passing Ticket 1
tests is good evidence for these four analytic functions. It is not evidence that the future
CN/PSOR solver, boundary extraction, or Greeks are correct.

For beginner researchers, the habit to build here is disciplined layering: validate the simple
analytic layer before trusting more complicated numerical layers.

## 9. Link to Next Ticket

Ticket 1 prepares Ticket 2 by providing the payoff and closed-form benchmark utilities that later
finite-difference work will need.

Ticket 2 should build the finite-difference grid and Black-Scholes operator setup. That means future
code will begin representing spot points, time-to-maturity points, interior nodes, and the discrete
version of the Black-Scholes spatial operator.

The connection is:

- Ticket 1 payoff functions can define terminal payoff values on a future grid.
- Ticket 1 European price functions can benchmark future European finite-difference values.
- Ticket 2 should not need to redefine payoff or closed-form Black-Scholes formulas.

Ticket 2 still should not implement Crank-Nicolson, PSOR, American option solving, boundary
extraction, Greeks, experiments, datasets, stress maps, or neural networks unless a later reviewed
ticket explicitly asks for those pieces.

## 10. Reproducibility Record

Files changed for Ticket 1 implementation and this report revision:

- `src/american_risk_surfaces/__init__.py`
- `src/american_risk_surfaces/solvers/__init__.py`
- `src/american_risk_surfaces/solvers/black_scholes.py`
- `tests/test_black_scholes.py`
- `reports/03_solver/tickets/ticket_01_black_scholes_utilities.md`
- `reports/03_solver/tickets/ticket_01_black_scholes_utilities.pdf`

Source code change during this report revision: none.

Test command:

```bash
PYTHONPYCACHEPREFIX=/private/tmp/codex_pycache PYTHONPATH=src python3 -m unittest discover -s tests
```

Final test result:

```text
Ran 9 tests in 0.002s

OK
```

Compile command:

```bash
PYTHONPYCACHEPREFIX=/private/tmp/codex_pycache PYTHONPATH=src python3 -m compileall -q src tests
```

Compile result: exited with status 0.

PDF generation/check:

- Markdown report was regenerated as `reports/03_solver/tickets/ticket_01_black_scholes_utilities.pdf`.
- `pdfinfo` reported the PDF successfully.
- `pdftoppm` rendered the PDF pages for visual inspection.
- A Fontconfig warning about a missing default config may appear in this environment, but the render
  command can still complete successfully.

Git status summary at report revision time:

```text
?? reports/03_solver/tickets/ticket_01_black_scholes_utilities.md
?? reports/03_solver/tickets/ticket_01_black_scholes_utilities.pdf
?? src/
?? tests/
```

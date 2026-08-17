# Proposed Poster Content Blueprint (English)

This is a content proposal only. It does not edit or replace the LaTeX poster.

## Suggested title

**American-Option Risk Surfaces: Validated LCP Solvers and Structure-Aware Learning**

## 1. Motivation

American-option risk surfaces require more than accurate prices. A useful method should also preserve the payoff obstacle, locate the early-exercise boundary, and produce stable Delta and Gamma estimates. We therefore separate strict LCP solvers, high-accuracy numerical references, and approximate learning-based surrogates.

## 2. Problem and validation controls

For the discretized American obstacle problem,

\[
u-\Phi\ge 0,\qquad Au-b\ge 0,\qquad
(u-\Phi)^\top(Au-b)=0.
\]

European Black--Scholes prices and the no-dividend American-call identity are used as analytic controls. They are not substitutes for an American early-exercise benchmark.

## 3. Benchmark hierarchy

- **Basic / Original Classical Benchmark:** Crank--Nicolson + PSOR.
- **Strengthened benchmark 1:** Crank--Nicolson + Policy Iteration.
- **Strengthened benchmark 2:** Crank--Nicolson + option-directed Projected LU.
- **High-accuracy reference:** L-stable DIRK + Policy Iteration on a strike-concentrated sinh grid.

The first three methods solve the same discrete CN-LCP. The high-accuracy reference addresses temporal/spatial discretization and Greek stability rather than online solver speed.

## 4. Main strict-solver result

Scope: 67 held-out test/stress regimes, float64, one CPU thread on the same Mac, five warm-ups and thirty timed repetitions per method.

| Method | Median latency (s) | P95 latency (s) |
|---|---:|---:|
| CN + PSOR | 0.309707 | 1.43094 |
| CN + Policy Iteration | 0.0159997 | 0.0224767 |
| CN + Projected LU | 0.0102586 | 0.0144603 |
| CN + Penalty/Newton | 0.0207950 | 0.0622017 |

The three benchmark solvers passed the common residual gate on all 67 held-out cases. Projected LU had a paired median runtime ratio of 0.6776 relative to Policy Iteration. Its maximum full-trajectory difference from Policy Iteration was (2.62\times10^{-14}), and its maximum normalized LCP residual was (6.30\times10^{-16}). The frozen Penalty/Newton candidate passed only 40/67 common gates and had no overall speed advantage over Policy; it remains a failed comparator, not a benchmark.

**Required caveat:** Four held-out low-volatility no-dividend calls lie outside the standard M-matrix sufficient conditions. The method is therefore numerically certified on the frozen SURF domain, not claimed to be theorem-certified for arbitrary parameters.

## 5. Structure-aware learning evidence

- A direct price MLP produced payoff-obstacle violations, whereas a positive-premium representation reduced the measured obstacle violation to zero.
- A bounded Delta head achieved lower test error than differentiating the price network directly in the audited supervised workflow.
- POD experiments showed that price surfaces can be low rank, but RB-VI and basis-operator studies failed stricter boundary, Greek, or complementarity gates. Low price RMSE alone is not sufficient for an American risk-surface method.

These supervised and reduced-order results were obtained under protocols different from the strict solver timing experiment and should be shown in a separate panel.

An independent nine-regime American-put campaign also supported Policy Iteration over PSOR (median 0.00731 s versus 0.14061 s), but it did not access a blind holdout and must not be merged into the 67-regime Projected-LU timing table. Its obstacle-aware PINN, unrolling, quantum, and tensor/low-rank studies produced negative or diagnostic evidence rather than benchmark-beating methods.

## 6. PINN and operator-learning status

Formal Soft-LCP PINN, exact-terminal Fischer--Burmeister PINN, strict PINN-to-Policy hybrids, and Positive-Premium DeepONet have implementation protocols but no formal five-seed held-out results yet. They should be listed as ongoing work rather than reported as completed successes.

The M04 and M09 implementations and M10--M24 method packets likewise do not yet provide canonical numerical results. Primary-source audit documents for M22--M24 are research designs, not experimental outcomes.

## 7. Main findings

1. Policy Iteration substantially strengthens the Basic / Original CN+PSOR benchmark for the same CN-LCP.
2. Option-directed Projected LU provides a further strict speed improvement on the frozen one-dimensional SURF domain.
3. High-accuracy boundary and Greek assessment requires a separate discretization reference.
4. Structure-aware learning is necessary, but accurate prices do not guarantee an accurate exercise boundary, Greeks, or complementarity.

## Suggested figures

1. A log-scale latency bar chart for PSOR, Policy Iteration, Projected LU, and a red-starred failed Penalty/Newton comparator.
2. A compact solver-correctness badge: `67/67`, `max |LU-Policy|`, and `max LCP residual`.
3. One exercise-boundary surface/curve using the high-accuracy reference.
4. A small evidence matrix with rows `price`, `obstacle`, `boundary`, `Delta/Gamma`, and columns for strict solvers versus learned/reduced methods.

## Claims that require provenance before reuse

- The PINN relative errors currently printed on the poster.
- The American-put error relative to the binomial reference, including the binomial resolution and convergence audit.

# Final Paper Outline

This outline is a Stage 6 preparation artifact. It is not the final paper.

## Supported Story

The paper should present a modular American-option risk-surface workflow: validated CN/PSOR benchmark, v1 research dataset, positive-premium price component, direct boundary diagnostic component, and supervised Delta diagnostic component. Gamma remains blocked.

## Sections

1. **Introduction**
   - Purpose: Motivate American option risk surfaces and the free-boundary challenge.
   - Evidence: Planning report and methodology guide.
   - Figures/tables: Integrated workflow diagram
2. **Research question**
   - Purpose: State the modular structure-preserving surrogate question.
   - Evidence: Application strength gate and Stage 6 claim matrix.
   - Figures/tables: Claim evidence matrix
3. **Background and American option formulation**
   - Purpose: Define payoff obstacle, LCP, boundary, Delta, and Gamma caveats.
   - Evidence: Formulation note and references.
   - Figures/tables: Mathematical formulation table
4. **Numerical benchmark: CN/PSOR and LCP validation**
   - Purpose: Document the validated finite-difference benchmark.
   - Evidence: Ticket 12 synthesis and v1 QA.
   - Figures/tables: Solver validation table
5. **Dataset construction**
   - Purpose: Explain v1 regimes, splits, masks, and diagnostics.
   - Evidence: v1 small-grid report.
   - Figures/tables: Dataset QA table
6. **Price / premium surrogate**
   - Purpose: Show why positive premium is the retained price component.
   - Evidence: Stage 3 metrics and obstacle summary.
   - Figures/tables: Price and obstacle figures
7. **Boundary diagnostic component**
   - Purpose: Explain why direct boundary diagnostics are needed.
   - Evidence: Stage 4 boundary metrics.
   - Figures/tables: Boundary error and curve figures
8. **Delta diagnostic component**
   - Purpose: Explain why Delta needs a supervised diagnostic head.
   - Evidence: Stage 5 metrics and bounds summary.
   - Figures/tables: Delta error and bounds figures
9. **Integrated workflow**
   - Purpose: Assemble the component workflow and claim matrix.
   - Evidence: Stage 6 readiness matrices.
   - Figures/tables: Component overview and evidence map
10. **Limitations**
   - Purpose: State blocked claims and scope boundaries.
   - Evidence: Blocked claims matrix.
   - Figures/tables: Blocked claims table
11. **Conclusion**
   - Purpose: Summarize the cautious research contribution.
   - Evidence: Supported claim matrix.
   - Figures/tables: Final claims status
12. **Future work**
   - Purpose: Describe larger regimes, active learning, Gamma diagnostics, and production gaps.
   - Evidence: Handouts and blocked claims.
   - Figures/tables: Future work table

## Claims To Use

- C01: A validated baseline CN/PSOR benchmark can generate consistent American-option diagnostic surfaces within the approved one-dimensional Black-Scholes grid. (SUPPORTED_WITH_LIMITATIONS)
- C02: Positive continuation-premium prediction is better aligned with the American payoff obstacle than direct value prediction. (SUPPORTED_WITH_LIMITATIONS)
- C03: Boundary behavior should be handled by a boundary-focused diagnostic component rather than only thresholding price/premium predictions. (SUPPORTED_WITH_LIMITATIONS)
- C04: Delta behavior should be handled with a Delta-focused diagnostic component rather than assumed from price accuracy. (SUPPORTED_WITH_LIMITATIONS)
- C05: The project contribution is a modular research workflow, not a single universal neural solver. (SUPPORTED_WITH_LIMITATIONS)

## Claims To Block

- Production pricing or production risk-system readiness: The workflow is validated as a research diagnostic pipeline only.
- Exact analytical free-boundary accuracy: Boundary labels are threshold-based CN/PSOR diagnostics.
- Production Greek reliability: Delta is a finite-difference diagnostic label and Gamma remains fragile.
- Gamma-head or Gamma-surrogate claims: No Gamma head is trained in the compressed roadmap.
- Broad extrapolation outside the approved v1 grid: The dataset uses a fixed small Cartesian grid and regime-level splits.
- One surrogate model solves price, boundary, and Greeks together: Stages 4 and 5 show separate boundary and Delta components are needed.

## Recommended Next Step

Draft the final paper from these matrices without adding new model claims.

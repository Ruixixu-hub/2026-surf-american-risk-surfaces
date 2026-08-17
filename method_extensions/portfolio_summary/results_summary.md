# Portfolio Summary

## Main benchmark result

The unified poster rerun used the same Mac, frozen 121-by-121 CN-LCP, 67
held-out regimes, float64 arithmetic, one CPU thread, five warm-ups, and 30
timed repetitions per arm. Projected LU (strengthened benchmark 2) had a pooled
median of `0.0102586 s`, compared with `0.0159997 s` for Policy Iteration
(strengthened benchmark 1) and `0.309707 s` for CN+PSOR (the basic/original
classical benchmark). Policy and Projected LU passed strict certification on
67/67 regimes.

Penalty/Newton was included under the same protocol. It passed the common LCP
gate on only 40/67 regimes and had a pooled median of `0.0207950 s`; it is
therefore a failed candidate comparator, not a benchmark.

The Projected-LU claim is limited to the frozen SURF domain because four
held-out q=0 calls lie outside the classic M-matrix sufficient conditions.

## Main methodological finding

Low price-surface rank does not by itself guarantee reliable exercise
boundaries, Greeks, active sets, or complementarity. RB-VI, oracle
alignment/localization, and the positive-premium basis operator were therefore
stopped by their structure-aware validation gates.

## Neural status

PINN Arms C/D/E and Positive-Premium DeepONet have implementation code and
registered protocols but no formal results yet. Their smoke/development outputs
are not used in the decision matrix.

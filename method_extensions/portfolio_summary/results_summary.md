# Portfolio Summary

## Main benchmark result

Projected LU is strengthened benchmark 2. On the same Mac and the same frozen
CN-LCP, its pooled median was `0.00938379 s`, compared with `0.0138143 s` for
strengthened benchmark 1 (Policy Iteration) and `0.243615 s` for historical
PSOR. All 67 held-out trajectories passed the strict numerical certification.

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

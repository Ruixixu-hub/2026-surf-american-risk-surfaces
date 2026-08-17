# Benchmarks

This directory groups both speed benchmarks and the high-accuracy numerical
reference. The distinction is explicit: CN+PSOR, CN+Policy, and CN+Projected LU
compete on the same frozen CN-LCP, whereas the DIRK/nonuniform-grid route is
used to score discretization accuracy and Greeks.

The unified poster comparison, including the failed Penalty/Newton candidate,
is stored in [`../results/14_poster_unified_comparison/`](../results/14_poster_unified_comparison/).

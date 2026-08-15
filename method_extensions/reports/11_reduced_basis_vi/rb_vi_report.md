# SURF Primal/Dual Reduced-Basis VI Report

Final decision: **STOP_ACCURACY** at validation.

No test/stress high-accuracy labels were opened, and Experiment 40 was not run, because neither option family passed the frozen validation gate.

## Plain result

The best stable put basis by price used 24 dual generators. Its worst reduction RMSE was 6.91162e-05, but its worst boundary error was 0.1272; Delta and stable-mask Gamma reached 1.272x and 2.326x the CN+Policy reference errors.

The best stable call basis by price used 32 dual generators. Its worst reduction RMSE was 0.00112145, boundary error was 2.71812, and its Delta/Gamma error ratios were 8.945x/42.744x.

Put dimension 32 was rejected by the stability gate rather than regularized: the dual cone became numerically linearly dependent.

## Decision

The low price error confirms the value trajectories are compressible, but the global primal/dual cone does not preserve the moving exercise boundary and Greeks well enough. The next justified branch is a boundary-aligned/localized basis or a positive-premium basis operator/DeepONet; this RB-VI model must not be presented as an online winner.

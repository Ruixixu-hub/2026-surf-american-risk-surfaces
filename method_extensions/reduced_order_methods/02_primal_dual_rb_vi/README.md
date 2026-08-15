# Primal/Dual Reduced-Basis VI

This method reduces both the primal value space and the nonnegative dual
multiplier cone, with supremizer enrichment and a reduced active-set solve.

Formal status: **STOP_ACCURACY** at validation. Low price error did not imply
acceptable exercise-boundary or Greek accuracy. Test/stress labels remained
sealed.

- Experiments: [`36`](../../experiments/36_rb_vi_protocol_and_snapshots.py)
  through [`41`](../../experiments/41_rb_vi_synthesis.py)
- Decision: [`method_decision.json`](../../results/09_reduced_basis_vi/06_synthesis/method_decision.json)
- Report: [`rb_vi_report.md`](../../reports/11_reduced_basis_vi/rb_vi_report.md)

Full train snapshots and basis `.npz` files are regenerable and intentionally
excluded from Git.

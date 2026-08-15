# PINN Arms C/D/E

Registered methods:

- Arm C: vanilla Soft-LCP PINN;
- Arm D: exact-terminal, Fischer--Burmeister, curriculum/adaptive PINN;
- Arm E: Arm-D prediction followed by strict Policy Iteration.

Current status: **NO FORMAL RESULTS YET**. Code, tests, protocol, and Windows
GPU instructions are present. Tiny Mac smoke runs are development checks only
and are deliberately not included as method results.

The future formal report must compare PINN outputs against CN+PSOR,
strengthened benchmark 1 (CN+Policy), and strengthened benchmark 2
(CN+Projected LU). Arm E may still use Policy as its strict finishing solver;
that does not remove Projected LU from the speed comparison.

- Experiments: [`29`](../../experiments/29_pinn_protocol_and_operator_audit.py)
  through [`35`](../../experiments/35_pinn_gap_synthesis.py)
- Source: [`pinn/`](../../src/american_risk_surfaces/pinn/)
- Windows guide: [`windows_pinn_gpu.md`](windows_run_guide/windows_pinn_gpu.md)

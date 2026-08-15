# Positive-Premium DeepONet

The registered parameter-conditioned branch/trunk operator includes N0
(surface), N1 (structure-supervised), N2 (VI-regularized), and strict hybrid H.

Current status: **NO FORMAL RESULTS YET**. Code, tests, protocol, and Windows
GPU instructions are present. No development/smoke output is published as a
formal method result.

Any future formal benchmark must report CN+PSOR, strengthened benchmark 1
(CN+Policy), and strengthened benchmark 2 (CN+Projected LU) on the same
hardware/protocol. The high-accuracy DIRK+sinh route remains the scoring
reference rather than the online-speed competitor.

- Experiments: [`52`](../../experiments/52_deeponet_protocol_and_data_audit.py)
  through [`57`](../../experiments/57_deeponet_synthesis.py)
- Source: [`deeponet/`](../../src/american_risk_surfaces/deeponet/)
- Windows guide: [`windows_deeponet_gpu_CN.md`](windows_run_guide/windows_deeponet_gpu_CN.md)

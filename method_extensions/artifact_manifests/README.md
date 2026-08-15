# Excluded Regenerable Artifacts

Git does not contain the following bulky intermediate artifacts:

- 202 train-only full-grid RB/FOM snapshots;
- RB and localized-basis `.npz` artifacts;
- basis-operator development/five-seed checkpoints;
- PINN tiny-smoke checkpoints;
- DeepONet tiny-smoke/development checkpoints.

The checked-in protocol JSON, snapshot/basis manifests, checkpoint status
metadata, metrics, and experiment scripts document how these files were made.
Their exclusion does not remove a formal positive result: PINN and DeepONet do
not yet have formal results, while RB-VI and the basis operator stopped at
validation.

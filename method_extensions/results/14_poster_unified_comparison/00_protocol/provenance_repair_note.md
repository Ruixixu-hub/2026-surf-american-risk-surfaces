# Validation provenance repair

On 2026-08-17, an initial validation command exceeded the tool yield window
and continued while the Penalty/Newton iteration cap was being corrected.
That stale process later overwrote the validation manifest.  The final
validation artifacts were therefore regenerated deterministically from the
unchanged, pre-registered penalty ladder and validation-only selection rule.

No held-out metric was used to change the ladder, selected penalty, tolerance,
or decision rule.  The selected diagnostic penalty remained `1e8`, and the
formal 8,040 timing samples were not rerun or altered.

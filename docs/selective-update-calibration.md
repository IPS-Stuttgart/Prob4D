# Calibration after guarded Bayesian update selection

A Bayesian physical-twin guard can select a subset of otherwise calibrated
observation updates. Calibration of the proposal before selection does not imply
calibration of the accepted subset. `prob4d.selective_update_calibration` adds a
source-validation certificate that keeps four populations separate:

1. the candidate update evaluated on every validation object/session;
2. the candidate restricted to groups accepted by the frozen guard;
3. the unchanged physical fallback; and
4. the actually deployed policy, which uses the candidate on acceptance and the
   exact fallback on rejection.

The artifact is additive. It does not alter provider-v2 observations, the
BayesianPhysTwin guard, a frozen held-out promotion lock, or any target-access
rule.

## Statistical and information boundary

Complete physical objects or independently acquired object sessions are the
independent groups. Frames, points, tracks, cameras, and taxels are not
independent calibration units.

The certificate binds three disjoint source partitions:

- `guard_fit_group_ids`: groups used to fit guard parameters;
- `guard_calibration_group_ids`: groups used to choose or calibrate the frozen
  acceptance rule; and
- `validation_group_ids`: groups used only to certify post-selection behavior.

Every artifact declares `evidence_partition = "source-validation"` and
`uses_target_outcomes = false`. Target or confirmation groups must not enter any
of the three partitions. The proper score is explicitly declared and is always
interpreted as lower-is-better. Every width uses one declared unit.

## Group rows

Each `SelectiveUpdateGroupV1` row contains candidate, fallback, and deployed
coverage, width, and proper score, plus the frozen acceptance decision. The
constructor fails closed unless deployed values equal the candidate for an
accepted group or the fallback for a rejected group. This makes exact fallback a
validated semantic condition rather than a report annotation.

Coverage is one group-level coverage statistic at the frozen nominal level. Width
and proper score must use the same definition in every row. For example, one row
may contain equal-track 90% coverage, mean full interval width in millimetres, and
mean Gaussian negative log likelihood for one complete object/session.

## Reported gates

The deterministic report separates proposal, accepted-subset, fallback, and
complete-policy summaries. Its frozen criteria cover:

- accepted-subset coverage shortfall;
- complete deployed-policy coverage shortfall;
- the coverage change induced by selection;
- accepted-subset width relative to the matched fallback on those same groups;
- accepted-subset proper-score advantage over the matched fallback;
- harmful accepted updates;
- minimum accepted independent-group support; and
- worst-group deployed coverage shortfall.

The complete-policy score advantage is also reported, but the promotion criterion
uses the accepted-subset comparison. Otherwise many exact fallback rows could
dilute either the benefit or harm of the updates that were actually accepted.

Insufficient accepted groups are a valid failed source result. Do not loosen the
support threshold after inspecting the validation rows.

## Building a certificate

Prepare the raw JSON form shown in
[`examples/selective-update-calibration-input.json`](examples/selective-update-calibration-input.json),
then run:

```bash
python -m prob4d.selective_update_calibration build \
  docs/examples/selective-update-calibration-input.json \
  --output outputs/selective-update-calibration.json \
  --require-pass
```

A valid failed gate is still written and exits with status 3 when
`--require-pass` is supplied. Replay a retained artifact with:

```bash
python -m prob4d.selective_update_calibration verify \
  outputs/selective-update-calibration.json \
  --require-pass
```

The JSON artifact is content-addressed, rejects duplicate keys and non-finite
constants, and is published atomically without replacing different existing
bytes.

## Promotion boundary

A passing source-validation certificate authorizes neither target access nor a
scientific claim. It may be bound into a later target-free promotion lock together
with the exact guard, candidate, fallback, provider, calibration, and cohort
identities. The target must still be opened exactly once under its own
authorization, and provider competence remains separate from downstream physical
query benefit.

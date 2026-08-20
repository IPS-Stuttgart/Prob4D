# CUT3R source-only diagnostic strata

`prob4d prediction cut3r-strata` freezes and reports diagnostic strata for the
already frozen CUT3R native-versus-Prob4D comparison. The artifact is additive:
it does not alter the comparison arms, registered aggregate endpoints, source
roster, provider selection, or target-access decision. The surrounding
execution order and stop rules remain in the
[CUT3R source qualification runbook](cut3r-qualification-runbook.md).

The purpose is failure localization. A single drift slope or aggregate error can
hide whether degradation appears only late in a sequence, immediately after a
restarted-window boundary, after occlusion, under fast deformation, at novel
viewpoints, or under a poorly conditioned metric anchor.

## Arm-neutral frozen features

The strata lock contains exactly six diagnostics:

| Stratum | Feature | Frozen source |
| --- | --- | --- |
| Absolute prefix age | `frames_since_sequence_start` | source frame index |
| Restart-boundary phase | `frames_since_restart_boundary` | frozen window schedule |
| Occlusion/reappearance gap | `occlusion_reappearance_gap_frames` | one common input-visibility analysis |
| Normalized image motion | `normalized_image_motion` | one common input-motion analysis |
| Viewpoint novelty | `viewpoint_rotation_novelty_deg` | one frozen prefix-only camera-geometry analysis |
| Metric-anchor conditioning | `metric_anchor_log10_condition_number` | one frozen prefix-only anchor-geometry analysis |

Every feature is computed once for the paired observation and must be exactly
identical across `native-continuous`, `restarted-newest`, and
`restarted-prob4d-fused`. The report rejects arm-dependent binning. In
particular, it does not call the native recurrent state and a restarted state
comparable merely because each reports its own age; the restart-phase feature is
defined by the common frozen window schedule.

Features must declare that they use no truth, target outcome, or downstream
physical innovation. Bin edges and the content-addressed record-generation
definition are frozen from design, development, or calibration information
before source-evaluation scores are read. The final edge begins an open-ended bin.

## Freeze the strata

Start from the checked-in example and replace the bin edges, metric names, and
minimum independent-group count with the exact source protocol:

```bash
prob4d prediction cut3r-strata freeze \
  outputs/cut3r/source-comparison-lock.json \
  docs/examples/cut3r-diagnostic-strata-spec.json \
  --output outputs/cut3r/diagnostic-strata-lock.json

prob4d prediction cut3r-strata verify-lock \
  outputs/cut3r/source-comparison-lock.json \
  outputs/cut3r/diagnostic-strata-lock.json
```

The lock binds the exact comparison-lock identity, protocol name, complete
source-evaluation group roster, frozen random-seed roster, the SHA-256 identity
of the exact record-generation definition, metrics, bin edges, minimum groups per
bin, weighting, and reporting-only claim boundary. Publication is atomic and
no-clobber.

## Record contract

The report input is strict JSON. Freeze only metrics that are defined on every
retained paired record; endpoint, seam, and drift endpoints remain in the
aggregate provider report unless a separate record definition is registered.

The report input is:

```json
{
  "schema": "prob4d.cut3r-diagnostic-records",
  "schema_version": 1,
  "comparison_lock_id": "<sha256>",
  "strata_lock_id": "<sha256>",
  "record_definition_sha256": "<sha256>",
  "source_truth_used": true,
  "target_payloads_opened": false,
  "target_outcomes_opened": false,
  "records": [
    {
      "group_id": "object-01",
      "case_id": "object-01-session-01",
      "frame_index": 42,
      "random_seed": 7,
      "arm_id": "restarted-prob4d-fused",
      "features": {
        "frames_since_sequence_start": 42,
        "frames_since_restart_boundary": 8,
        "occlusion_reappearance_gap_frames": 0,
        "normalized_image_motion": 0.021,
        "viewpoint_rotation_novelty_deg": 11.4,
        "metric_anchor_log10_condition_number": 1.7
      },
      "metrics": {
        "point-error-m": 0.012,
        "proper-score": -4.3
      }
    }
  ]
}
```

Every `(group, case, frame, random seed)` must contain exactly the three enabled
causal, claim-eligible arms, and every frame must retain the complete random-seed
roster frozen by the comparison lock. The two schedule-derived features must
exactly match the source frame index and common restart schedule, and all other
features must agree across arms and seeds. Every frozen evaluation frame must be
retained; the frame must lie inside the comparison lock's frozen evaluation
interval, and every frozen source-evaluation group must be present. If a group has a technical failure or
lacks paired common support, retain that result in the authoritative source
competence report; do not silently delete the group to manufacture a strata
report.

## Build and verify the report

```bash
prob4d prediction cut3r-strata report \
  outputs/cut3r/source-comparison-lock.json \
  outputs/cut3r/diagnostic-strata-lock.json \
  outputs/cut3r/diagnostic-records.json \
  --output outputs/cut3r/diagnostic-strata-report.json

prob4d prediction cut3r-strata verify-report \
  outputs/cut3r/source-comparison-lock.json \
  outputs/cut3r/diagnostic-strata-lock.json \
  outputs/cut3r/diagnostic-records.json \
  outputs/cut3r/diagnostic-strata-report.json

prob4d prediction cut3r-strata summarize \
  outputs/cut3r/source-comparison-lock.json \
  outputs/cut3r/diagnostic-strata-lock.json \
  outputs/cut3r/diagnostic-records.json \
  outputs/cut3r/diagnostic-strata-report.json \
  --json
```

For each bin and arm, the report:

1. averages nested frame records inside each frozen random seed;
2. gives frozen seeds equal mass inside each case;
3. averages case means inside the complete object/session group;
4. averages complete groups with equal mass; and
5. computes registered arm contrasts as paired complete-group differences.

This prevents a lucky seed, long video, dense frame region, or object with many
cases from becoming many independent replicates. Bins below the frozen minimum independent
group count remain visible but are marked inadequately supported.

## Scientific boundary

The strata are reporting-only. They may identify the first plausible failure
boundary and motivate a separately versioned future source protocol, but they
cannot select the current provider, tune thresholds on the opened source groups,
reverse an aggregate negative, authorize target access, or rescue failed support,
mean, identity, gauge, covariance, or physical-query gates.

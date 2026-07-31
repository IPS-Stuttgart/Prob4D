# Held-out provider competence evaluation

`prob4d evaluate provider` is the first-class multi-case evaluator for dense
Prob4D prediction artifacts. It measures observation competence separately from
Bayesian-PhysTwin acceptance and downstream Causal4D prediction.

The evaluator is deliberately paired and fail-closed:

- every case must contain the same method set;
- every prediction archive must carry its fusion, covariance, and dependence
  semantics;
- one method cannot mix Prob4D revisions, MotionCrafter revisions or seed
  policies, gauge estimators, or calibration statuses across cases;
- historical archives without embedded semantics are rejected unless the run is
  explicitly labelled with `--allow-legacy-artifacts`;
- cases are first averaged within a registered `group_id`, then groups receive
  equal aggregate mass;
- one registered reference method defines paired within-case differences for every
  candidate;
- confidence intervals use a deterministic bootstrap over groups, not dense
  pixels or frames.

## Prediction artifact semantics

New benchmark exports store a versioned `prob4d.fused-prediction` descriptor in
the NPZ archive and retain point/flow fields in `float32` for evaluation. The
covariance field has one of three explicit meanings:

| Fusion rule | Covariance meaning | Dependence assumption |
| --- | --- | --- |
| uniform | Gaussian-mixture second moment | descriptive mixture, not a conditioned posterior |
| precision | independent-Gaussian posterior | contributors are treated as independent |
| covariance intersection | unknown-correlation consistency bound | unknown cross-contributor correlation |

These meanings are not interchangeable. The evaluator verifies that the
manifest method label agrees with the archive and that one method retains one
semantic signature across all held-out cases. Nonlegacy artifacts must also
declare exact Prob4D and MotionCrafter revisions, MotionCrafter seed policy,
gauge estimator, and uncertainty-calibration status.

Run `prob4d-benchmark` with `--include-covariance` to produce evaluable fused
archives. The benchmark manifest also records inference, loading, Prob4D fusion,
and artifact-export time separately.

## Evaluation manifest

Paths are resolved relative to the manifest unless they are absolute.

```json
{
  "schema_name": "prob4d.provider-evaluation",
  "schema_version": 1,
  "primary_mode": "metric",
  "reference_method": "prob4d_uniform",
  "cases": [
    {
      "case_id": "object-01-session-02",
      "group_id": "object-01",
      "truth": "truth/object-01-session-02.npz",
      "predictions": {
        "prob4d_uniform": "predictions/uniform/object-01-session-02.npz",
        "prob4d_ci": "predictions/ci/object-01-session-02.npz"
      },
      "boundary_frames": [25, 42, 59],
      "prefix_frame_stop_exclusive": 25
    }
  ],
  "metadata": {
    "split_id": "held-out-objects-v1",
    "calibration_id": "source-calibration-v1"
  }
}
```

`primary_mode` is one of:

- `metric`: use the prediction's declared metric frame without truth alignment;
- `prefix_aligned`: fit one scale and translation using only frames before each
  case's exclusive prefix stop, then evaluate the full sequence;
- `oracle_aligned`: fit on all evaluated truth and report a reconstruction-only
  diagnostic.

The evaluator always reports all available modes. Choosing `prefix_aligned` as
the primary mode requires a prefix stop for every case. `reference_method` must
identify one method present in every case. The report computes
`candidate - reference` differences within case, averages them within group, and
bootstraps groups.

## Command

```bash
prob4d evaluate provider protocols/provider-evaluation.json \
  --output-dir outputs/provider-evaluation \
  --bootstrap-resamples 2000 \
  --seed 7
```

The installed compatibility entry point is:

```bash
prob4d-evaluate-provider protocols/provider-evaluation.json \
  --output-dir outputs/provider-evaluation
```

Outputs are:

- `provider_evaluation.json`: nested case results, artifact semantics, equal-group
  aggregate intervals, and paired differences from the reference;
- `provider_evaluation.csv`: one flat row per case and method;
- `provider_evaluation.md`: a compact primary-mode table.

Reported metrics include metric and optionally aligned point RMSE, endpoint
error, drift slope, seam error, flow endpoint error, 95% ellipsoid coverage,
Gaussian negative log likelihood, and mean squared Mahalanobis error.

## Claim boundary

A passing provider report establishes only held-out observation competence under
the registered split and artifact semantics. It does not establish that a
Bayesian-PhysTwin update is identifiable or safe, that its baseline-relative
guard accepts the update, that physical prediction improves, or that an
accepted belief improves a Causal4D intervention forecast. Those remain separate
paired gates with exact physical-baseline fallback.

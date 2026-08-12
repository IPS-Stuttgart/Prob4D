# Provider readiness localization

Prob4D already separates support feasibility, source mean/identity competence,
gauge/dependence diagnostics, point covariance, query relevance, and exact
fallback. This note documents three additive tools that close the remaining
integration gaps without changing provider-v2 or fitting a new uncertainty model.

## 1. Contiguous support envelopes before residuals

`ProviderSupportFeasibilityV1` answers whether one frozen required-frame set is
supported. `ProviderSupportEnvelopeV1` additionally summarizes **all contiguous
frame intervals** that are supported by the same already-frozen camera/robot
metadata.

```bash
python -m prob4d.provider_support_envelope derive \
  --request support-request.json \
  --output support-envelope.json

python -m prob4d.provider_support_envelope verify \
  --artifact support-envelope.json
```

For each stream the artifact records:

- the frozen causal span;
- all contiguous admissible intervals `[start, stop_exclusive)`;
- earliest/latest admissible bounds and maximum contiguous length;
- required-frame support fraction;
- static intrinsics/extrinsics/metric-anchor completeness; and
- the original feasibility result and reason codes.

The envelope is derived without prediction payloads, residuals, or target
outcomes. It **does not modify** its source request. If a different prefix is to
be tried, freeze a new `ProviderSupportFeasibilityRequestV1` or a prospectively
declared `ProviderSupportDesignRequestV1` from the envelope before opening later
information. This is the safe route for avoiding another fixed-prefix support
failure without deleting cameras after outcomes are known.

## 2. Evidence-driven source covariance localization

`SourceCovarianceLocalizationV1` binds two existing source-only evidence types:

1. `SourceProviderCompetenceReportV1`, which separates observation-mean quality
   from identity/reliability quality; and
2. `prob4d.joint-covariance-diagnostics`, which separates shared/gauge-subspace
   residual energy from conditional-subspace residual energy.

A frozen policy declares acceptable bands and minimum independent-group pass
fractions. Evaluation obeys this stop order:

1. source-mean failure -> stop, no covariance development;
2. identity/association failure -> stop, no covariance development;
3. shared/gauge energy failure -> `gauge-or-dependence-negative`;
4. shared/gauge pass **and** conditional failure ->
   `point-covariance-localized`; and
5. both subspaces plus joint NEES pass -> `covariance-adequate`.

Only step 4 sets `authorize_point_uncertainty_development=true`. A joint-NEES
failure that cannot be assigned to the conditional subspace is conservatively
classified as gauge/dependence failure rather than being used to justify a more
complex point model.

The localizer also requires the joint diagnostic's factor-group IDs to match the
exact evaluable source object/session IDs in the source-competence report. This
prevents pixel-, frame-, or mismatched-cohort evidence from silently being used
as independent calibration groups.

Example policy:

```json
{
  "minimum_group_count": 8,
  "normalized_nees_lower": 0.7,
  "normalized_nees_upper": 1.3,
  "minimum_joint_pass_fraction": 0.8,
  "shared_energy_lower": 0.7,
  "shared_energy_upper": 1.3,
  "minimum_shared_pass_fraction": 0.8,
  "conditional_energy_lower": 0.7,
  "conditional_energy_upper": 1.3,
  "minimum_conditional_pass_fraction": 0.8,
  "require_shared_subspace": true
}
```

These numbers are examples only; a real study must freeze its policy from the
source/calibration protocol rather than copy them.

```bash
python -m prob4d.source_covariance_localization evaluate \
  --source-competence source-provider-competence.json \
  --joint-diagnostic joint-covariance.json \
  --policy covariance-localization-policy.json \
  --output covariance-localization.json
```

`readiness_gates()` converts the result directly into the existing
`gauge-dependence` and `point-covariance` `ReadinessGateV1` entries. No new
readiness classification is introduced.

## 3. Strict causal-prefix admission

Calibration transport already detects whether a causal target prefix lies in the
source-calibration feature regime. `ProviderPrefixAdmissionV1` makes the missing
conjunction explicit:

```text
admitted = support_feasible AND calibration_transport_accepted
```

The calibration-transport evidence must include the following metadata binding:

```json
{
  "provider_prefix_binding": {
    "provider_manifest_id": "<sha256>",
    "cohort_binding_id": "<sha256>",
    "target_prefix_id": "<sha256>",
    "causal_prefix_only": true,
    "target_residuals_used": false,
    "target_outcomes_used": false
  }
}
```

The builder rejects provider, cohort, or prefix mismatches as invalid evidence;
they are not converted into scientific negatives. A valid negative support or
transport result instead produces `admitted=false` and
`exact_fallback_required=true`.

```bash
python -m prob4d.provider_prefix_admission build \
  --support provider-support.json \
  --transport-model calibration-transport-model.json \
  --transport-evidence calibration-transport-evidence.json \
  --provider-manifest-id <sha256> \
  --target-prefix-id <sha256> \
  --output provider-prefix-admission.json
```

This certificate remains upstream of BayesianPhysTwin. Passing it means only
that Prob4D's provider prefix is geometrically feasible and inside the frozen
source-calibration support region. BayesianPhysTwin still owns physical-query
relevance, update admission, regret guards, and exact physical fallback;
Causal4D still consumes only the resulting accepted physical belief.

## Scientific stop rule

These additions deliberately do **not** implement `PointUncertaintyCalibrationV2`.
A richer point model is justified only after a real source/calibration execution
produces `point-covariance-localized`. A support, mean, identity, gauge/dependence,
transport, or query failure redirects the provider instead of increasing
covariance-model complexity.

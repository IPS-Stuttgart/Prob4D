# Provider-boundary examples

- `prediction-batch-preflight-policy.json` is the strict default scorer-batch policy.
- `provider-terminal-batch-incompatible.json` is an input specification for
  `python -m prob4d.provider_terminal_decision build`.

The example terminal specification intentionally contains no `artifact_id`; the
builder derives it from the canonical content. Replace every placeholder identity
with the exact content identity from the frozen provider protocol.

- `material-identity-weight-calibration-input.json` is a complete source-only
  group-cross-fitting example for `prob4d identity fit-calibration`.
- `material-identity-calibrated-mixture-config.json` applies the retained model
  through `prob4d identity calibrate-mixture`.
- `material-identity-mixture-config.json` remains the lower-level example for
  externally supplied calibrated log weights.
- `cut3r-source-competence-v2-spec.json` freezes exact common metric support,
  arm-neutral source-mean proper-score semantics, and paired source endpoints on
  top of an existing CUT3R source-competence v1 lock.
- `cut3r-source-competence-audit-spec.json` binds the exact arm-neutral score-
  reference artifact and enables independent reconstruction of every support
  digest from retained canonical rows.

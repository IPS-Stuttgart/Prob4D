# Preregistered provider-evaluation decisions

Provider-evaluation manifest version 2 adds a target-frozen decision policy to the
paired held-out evaluator. Manifest version 1 and report version 2 remain unchanged
for frozen reproduction.

The decision policy answers one narrow question: whether a registered Prob4D
observation method passes declared held-out provider-competence gates relative to the
manifest's reference method. It does not authorize a Bayesian-PhysTwin update and it
does not establish Causal4D intervention benefit.

## Manifest

A decision-bearing manifest uses schema version 2 and adds exactly one
`decision_policy` object:

```json
{
  "schema_name": "prob4d.provider-evaluation",
  "schema_version": 2,
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
    "target_access_seal": "<registered-seal-id>"
  },
  "decision_policy": {
    "policy_id": "prob4d-provider-gate-v1",
    "minimum_group_count": 9,
    "rules": [
      {
        "rule_id": "point-rmse-superiority",
        "candidate_method": "prob4d_ci",
        "metric": "metric_point_rmse",
        "direction": "lower",
        "criterion": "superiority",
        "margin": 0.0
      },
      {
        "rule_id": "coverage-noninferiority",
        "candidate_method": "prob4d_ci",
        "metric": "coverage_95",
        "direction": "higher",
        "criterion": "noninferiority",
        "margin": 0.03
      }
    ]
  }
}
```

All candidate methods must be present in every case and must differ from the
registered reference. Rule IDs are unique. `metric` is one unqualified metric field
from the selected primary evaluation mode. A decision-bearing manifest cannot use
`oracle_aligned`, because a truth-fitted full-sequence alignment is a reconstruction
diagnostic rather than a prospective observation gate.

## Bound interpretation

Every rule consumes the paired `candidate - reference` group-bootstrap summary. The
bound is selected by the rule semantics rather than by the observed result:

| Direction | Criterion | Passing condition |
| --- | --- | --- |
| lower is better | superiority | upper 95% bound <= `-margin` |
| lower is better | non-inferiority | upper 95% bound <= `margin` |
| higher is better | superiority | lower 95% bound >= `margin` |
| higher is better | non-inferiority | lower 95% bound >= `-margin` |

The complete policy passes only when every rule passes and the observed independent
`group_id` count is at least `minimum_group_count`. An insufficient group count is
reported as a failed gate; it does not discard the evaluated cases or suppress the
negative result.

Report schema version 3 binds the exact policy and emits:

- observed and required independent-group counts;
- every metric path, estimate, interval, selected decision bound, and threshold;
- per-rule pass/fail results;
- the complete policy pass/fail result; and
- the unchanged provider-only claim boundary.

## Fail-closed command use

The ordinary command always writes the report, including a failed decision:

```bash
prob4d evaluate provider protocols/provider-evaluation-v2.json \
  --output-dir outputs/provider-evaluation
```

For automation, require the registered policy to pass:

```bash
prob4d evaluate provider protocols/provider-evaluation-v2.json \
  --output-dir outputs/provider-evaluation \
  --require-decision-pass
```

The command returns:

- exit code `0` when the schema-v2 decision policy passes;
- exit code `3` when the report is valid but the group or metric gates fail; and
- argparse exit code `2` when `--require-decision-pass` is requested for a manifest
  without a decision policy.

The JSON, CSV, and Markdown outputs are written before exit code 3 is returned, so a
well-powered negative or underpowered result remains inspectable and archivable.

## Target-authorized physical-cohort execution

A claim-bearing physical-cohort evaluation must additionally supply the frozen
promotion lock and metadata-only target-provider admission:

```bash
prob4d evaluate provider protocols/provider-evaluation-v2.json \
  --promotion-lock promotion-lock.json \
  --target-provider-admission target-provider-admission.json \
  --bootstrap-resamples 10000 \
  --seed 20260805 \
  --output-dir outputs/provider-evaluation \
  --require-decision-pass
```

The provider-evaluation manifest SHA is itself frozen inside the promotion lock.
The target admission then binds that lock. Therefore the admission ID must **not**
be placed inside manifest metadata: that would create a circular identity in which
the manifest SHA determines the lock, the lock determines the admission, and the
admission would determine the manifest SHA.

The target-authorized route breaks this cycle with a separate content-addressed
`target_admission_authorization` receipt. Before target paths are resolved or
artifact contents are opened, it validates:

- exact provider-evaluation manifest bytes against the lock;
- the complete decision policy, target groups, method set, reference method, and
  minimum independent-group count;
- exact bootstrap resamples and seed;
- the admission against the promotion lock, cohort binding, provider revision,
  model set, run spec, and target groups; and
- the prohibition on legacy artifacts and target-outcome access during
  authorization.

Only after those checks pass does the evaluator materialize a private execution
manifest derived from the exact snapshotted bytes. Relative truth and prediction
paths are rewritten to absolute paths so their original manifest-directory
semantics survive the private snapshot. No estimator, metric, decision rule, or
scientific value is changed.

A target-authorized report uses schema version 4 and retains:

- `source_manifest_sha256` for the exact frozen input bytes;
- the unchanged manifest metadata, which must not contain the admission ID;
- the complete decision and case records; and
- the content-addressed `target_admission_authorization` receipt binding the lock,
  admission, cohort, manifest, registered groups and methods, case count,
  reference, decision minimum, bootstrap settings, and
  `target_outcomes_opened_during_authorization=false`.

The held-out promotion runner replays that receipt and separately requires the
BayesianPhysTwin query results to carry
`metadata.target_provider_admission_id`. Missing, circular, or mismatched result
custody fails closed before the two decisions are combined.

## Information boundary

The policy, split, grouping unit, reference method, metric directions, criteria,
margins, minimum group count, bootstrap configuration, model identities, and complete
prediction artifacts must be sealed before target outcomes are opened. Changing a rule
after scoring creates a new exploratory method rather than repairing the registered
one.

A passing provider decision remains only an observation-quality gate. Bayesian-PhysTwin
must separately evaluate identifiability, guarded acceptance, harmful accepted updates,
physical prediction, uncertainty width, and exact fallback. Causal4D may consume only
an accepted, content-bound twin belief and must not assimilate the same Prob4D evidence
a second time.

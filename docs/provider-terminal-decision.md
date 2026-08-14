# Provider terminal decision artifact

Provider experiments can stop at fundamentally different information
boundaries. An unsupported stream roster, an incompatible prediction batch, a
scorer crash, a valid scientific negative, and a completed positive target gate
must not be represented by the same generic `failed` flag.

`ProviderTerminalDecisionV1` is a small, content-addressed record that classifies
one completed protocol boundary and binds what information was accessed and what
inferences are or are not authorized.

## Classifications

| Classification | Meaning | Scientific result |
| --- | --- | --- |
| `support-negative` | Frozen support feasibility failed before payload access. | Support evidence only |
| `batch-incompatible` | Admitted prediction bytes are individually valid but violate the frozen scorer representation. | No |
| `technical-failure` | Execution terminated before a valid scientific outcome. | No |
| `scientific-negative` | A registered source or target outcome gate produced a valid negative. | Yes, within the declared inference |
| `completed-positive` | A registered target outcome gate produced a valid positive. | Yes, within the declared inference |

The artifact does not infer these classifications from a traceback. The protocol
owner supplies an exact specification, and construction replays the stop rules.

## Information-access fields

Every decision records four separate booleans:

- `source_payloads_accessed`;
- `source_outcomes_accessed`;
- `target_payloads_accessed`; and
- `target_outcomes_accessed`.

Outcome access requires the corresponding payload access. Target access also
requires source-payload access. A result that opened target outcomes can never
authorize a rerun of the same target protocol.

## Infrastructure nonclaim enforcement

`batch-incompatible` and `technical-failure` decisions must have an empty
`authorized_inferences` list. They must explicitly forbid all of:

```text
provider-competence
provider-calibration
bayesian-phystwin-benefit
causal4d-intervention-benefit
deployment-safety
state-of-the-art
```

This is validated by the artifact constructor and loader. Omitting one of these
boundaries is an invalid artifact rather than a weaker warning.

A `support-negative` decision must precede every source or target payload access
and may authorize only `provider-support-negative`.

## Building and verifying

Create a JSON specification with every artifact field except `artifact_id`, then
run:

```bash
python -m prob4d.provider_terminal_decision build \
  provider-terminal-specification.json \
  provider-terminal-decision.json
```

Verify the persisted bytes and replay all constraints with:

```bash
python -m prob4d.provider_terminal_decision verify \
  provider-terminal-decision.json
```

Writes are atomic and no-clobber. Repeating the exact artifact is idempotent;
trying to replace it with a different decision fails.

## Example batch-terminal decision

```json
{
  "schema": "prob4d.provider-terminal-decision",
  "schema_version": 1,
  "protocol_id": "fresh-provider-v7-source",
  "provider_manifest_id": "<64 hexadecimal characters>",
  "classification": "batch-incompatible",
  "failed_stage": "source-batch-preflight",
  "source_payloads_accessed": true,
  "source_outcomes_accessed": false,
  "target_payloads_accessed": false,
  "target_outcomes_accessed": false,
  "rerun_authorized": false,
  "successor_protocol_required": true,
  "evidence_ids": ["<prediction-batch-preflight artifact ID>"],
  "authorized_inferences": [],
  "forbidden_inferences": [
    "provider-competence",
    "provider-calibration",
    "bayesian-phystwin-benefit",
    "causal4d-intervention-benefit",
    "deployment-safety",
    "state-of-the-art"
  ],
  "summary": "The admitted source payloads cannot form the frozen scorer batch.",
  "metadata": {
    "future_prediction_payloads_opened": 0
  }
}
```

The builder adds the content-derived `artifact_id`.

## Repository boundary

Prob4D owns the prediction and provider-side execution boundary. BayesianPhysTwin
continues to own physical-query admission, regret guards, selected beliefs, and
exact physical fallback. Causal4D consumes only the selected BayesianPhysTwin
belief and owns intervention and counterfactual inference. A terminal decision
cannot bypass either downstream boundary.

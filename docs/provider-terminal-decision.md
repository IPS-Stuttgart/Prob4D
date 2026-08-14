# Provider terminal decision artifact

Provider experiments can stop at different information boundaries. Unsupported
geometry, an incompatible prediction batch, a scorer crash, a valid scientific
negative, and a completed target result must not share one generic `failed`
flag.

`ProviderTerminalDecisionV1` records the exact boundary, information access,
evidence identities, and authorized and forbidden inferences.

## Classifications

| Classification | Boundary |
| --- | --- |
| `support-negative` | Frozen support feasibility failed before payload access. |
| `batch-incompatible` | Valid admitted bytes violate the frozen scorer representation. |
| `technical-failure` | Execution terminated before a valid scientific outcome. |
| `scientific-negative` | A registered outcome gate produced a valid negative. |
| `completed-positive` | A registered target gate produced a valid positive. |

The four access flags separately record source payloads, source outcomes, target
payloads, and target outcomes. Outcome access requires payload access. Once
target outcomes have been opened, the same protocol cannot authorize a rerun.

## Infrastructure nonclaims

`batch-incompatible` and `technical-failure` authorize no scientific inference.
They must explicitly forbid:

```text
provider-competence
provider-calibration
bayesian-phystwin-benefit
causal4d-intervention-benefit
deployment-safety
state-of-the-art
```

A `support-negative` must precede all payload access and may authorize only
`provider-support-negative`.

## Commands

Create a specification containing every field except `artifact_id`, then run:

```bash
python -m prob4d.provider_terminal_decision build \
  provider-terminal-specification.json \
  provider-terminal-decision.json

python -m prob4d.provider_terminal_decision verify \
  provider-terminal-decision.json
```

Writes are atomic and no-clobber. Repeating identical bytes is idempotent;
replacing a sealed decision fails.

## Repository boundary

Prob4D owns provider-side execution evidence. BayesianPhysTwin continues to own
physical-query admission, guards, selected beliefs, and exact fallback.
Causal4D consumes only the selected BayesianPhysTwin belief and owns
interventional inference. This artifact bypasses neither boundary.

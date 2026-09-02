# Development history

## v1 sealed-submission hardening

- Separates challenge-owned truth, queries, losses, ambiguity sets, and fallback
  from provider-owned means, covariance, admissions, certificates, and actions.
- Adds exact challenge/submission roster and payload-hash binding.
- Distinguishes retrospective replay from prospectively sealed evaluation.
- Adds finite-query identifiability over quotient classes and explicitly reports
  local-nullspace admissions that fail the finite test.
- Charges communication cost to the provider submission rather than the joined
  truth-and-prediction replay payload.
- Adds adversarial tests for truth smuggling, provider-array smuggling, false
  finite-query admission, task/payload drift, and information-order relabelling.

## v1 controlled conformance candidate

- Defines a provider-neutral, multi-axis scorecard with no scalar overall score.
- Adds deterministic controls for accuracy/calibration rank reversal, joint
  dependence, finite-support query and decision semantics, exact fallback,
  communication parity, unsupported specificity, and shared provider bias.
- Adds fail-closed validation, focused tests, and deterministic CI reproduction.
- Opens no dataset and invokes no learned provider.

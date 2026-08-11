# BayesianPhysTwin evidence-decision consumer v1

Prob4D provides an independent validator for the closed
`bayesian_phystwin.evidence_decision` version-1 wire format. The validator lets
Prob4D verify that a downstream decision cites the exact Prob4D repository state
that participated in a result without adding a reverse runtime dependency on
BayesianPhysTwin.

The implementation is available from:

```python
from prob4d.api.evidence_decision_v1 import (
    load_evidence_decision_v1,
    require_authorized_evidence_decision_v1,
    require_prob4d_evidence_binding_v1,
)

decision = load_evidence_decision_v1("decision.json")
require_authorized_evidence_decision_v1(
    decision,
    claim_id="registered.claim",
    protocol_id="registered-protocol-v1",
    minimum_evidence_level=3,
)
prob4d_state = require_prob4d_evidence_binding_v1(
    decision,
    expected_revision="0123456789abcdef0123456789abcdef01234567",
)
```

## Validation boundary

The consumer validates the complete closed shape and semantic invariants of the
version-1 decision:

- canonical text, SHA-256 identities, exact Git revisions, and UTC timestamps;
- finite metric and metadata values;
- one primary repository and unique repository identities;
- deterministic repository ordering and content-address verification;
- claim authorization only for a passing confirmatory decision over clean
  repositories; and
- mandatory limitations for degraded or inconclusive outcomes.

`require_prob4d_evidence_binding_v1` additionally recognizes canonical and
historical Prob4D repository identities through the existing project-identity
boundary. It requires exactly one Prob4D binding and can lock its revision,
allowed role, and clean state.

## Source lock

This generated consumer binding is locked to:

- source repository `IPS-Stuttgart/BayesianPhysTwin`;
- source revision `4ee702f5130cfedbea7bce6be5e72483c92f63da`; and
- JSON Schema SHA-256
  `d5615258c6cf666d0ed9684a87930989adf91817fe99b0387e83a31479dcd465`.

The dedicated integration workflow checks both the schema bytes and a real
decision emitted by the pinned BayesianPhysTwin package. A semantic wire change
therefore requires a new consumer module rather than silently changing version
1.

## Command-line validation

The module can also be executed directly:

```bash
python -m prob4d.evidence_decision_v1 \
  decision.json \
  --require-authorized \
  --expected-prob4d-revision 0123456789abcdef0123456789abcdef01234567
```

## Scientific boundary

A valid envelope proves contract conformance, content identity, and declared
repository provenance. It does not establish that Prob4D observations are
accurate or calibrated, that BayesianPhysTwin improves a physical endpoint,
that Causal4D interventions are beneficial, or that a result transfers to a new
object or session. Those claims still require their registered physical and
statistical evidence.

# Provider support promotion authorization

`prob4d.provider_promotion_authorization` binds the existing target-free provider
support-feasibility result to the exact held-out promotion lock.

The authorization is created before calibration payloads, provider residuals, or
target payloads are opened. It is a gate, not empirical evidence.

## Required order

The strict prospective route is:

```text
freeze source, provider, model, loader, cohort, camera, and target roster
→ evaluate target-free support feasibility
→ create ProviderPromotionAuthorizationV2
→ run calibration and target-blind selection
→ open the sealed target evaluation
→ bind replay-complete evidence to the earlier authorization
```

Historical promotion evidence remains readable. New strict evidence can be
wrapped in `AuthorizedHeldoutProviderEvidenceV1` to prove that it belongs to the
earlier support authorization.

## Authorization checks

`authorize_provider_promotion()` fails closed unless:

- the support result is positive;
- prediction payloads, provider residuals, and target outcomes were not used;
- calibration payloads, target payloads, and provider residuals are still closed
  when authorization is created;
- the support request references the exact promotion-lock identity;
- source repository, source revision, and model-set identity match the lock;
- the request covers exactly the frozen target-group roster; and
- every target group has at least one supported stream.

The last condition is stricter than an aggregate stream-fraction threshold. A
cohort cannot authorize promotion while one complete target group lacks usable
support.

## Bound identities

The authorization retains the complete promotion lock and support result and
also exposes compact identities for:

- the support request and result;
- the complete stream roster and causal-prefix requirements;
- intrinsics, extrinsics, and metric anchors;
- the technical-exclusion policy;
- the target-group roster; and
- the set of target groups with actual support.

Any change to those values changes `authorization_id`.

## Usage

```python
from prob4d.provider_promotion_authorization import (
    authorize_provider_promotion,
    bind_authorized_heldout_provider_evidence,
)

authorization = authorize_provider_promotion(
    promotion_lock,
    support_feasibility,
)

authorized_evidence = bind_authorized_heldout_provider_evidence(
    authorization,
    heldout_provider_evidence,
)
```

Command-line use is available without adding another legacy executable:

```bash
python -m prob4d.provider_promotion_authorization authorize \
  --promotion-lock promotion-lock.json \
  --support-feasibility support-feasibility.json \
  --output promotion-authorization.json

python -m prob4d.provider_promotion_authorization verify \
  --artifact promotion-authorization.json

python -m prob4d.provider_promotion_authorization bind-evidence \
  --authorization promotion-authorization.json \
  --evidence heldout-provider-evidence.json \
  --output authorized-heldout-provider-evidence.json
```

Writers are atomic and refuse replacement by default. Loaders reject duplicate
JSON keys, non-finite values, schema drift, derived-identity drift, and content
identity changes.

## Scientific boundary

A positive authorization establishes only target-free technical support for the
exact frozen route. It does not establish provider competence, calibrated
uncertainty, BayesianPhysTwin benefit, Causal4D benefit, deployment safety,
generalization, or state of the art.

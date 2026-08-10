# Outcome-blind provider support design

`prob4d.provider_support_design` selects one support configuration from a finite
candidate set frozen before prediction payloads, provider residuals, or target
outcomes are opened.

The design step belongs before
[`ProviderSupportFeasibilityV1`](provider_support_feasibility.md). It prevents a
provider configuration from being chosen after an expensive residual or target
run has revealed which camera roster, causal prefix, or geometry route performs
well. A selected design remains only support infrastructure; it is not evidence
that the provider is accurate, calibrated, beneficial to BayesianPhysTwin or
Causal4D, or safe to deploy.

## Candidate boundary

Each candidate embeds one complete, content-addressed
`ProviderSupportFeasibilityRequestV1`. Candidates must share:

- source, provider, model-set, loader, cohort, and promotion-lock identities;
- coordinate semantics, admission rule, and technical-exclusion policy;
- the exact group/stream roster; and
- each stream's geometry-support threshold and requirements for intrinsics,
  extrinsics, and metric anchors.

Candidates may differ in their frozen causal interval, required and available
frames, geometry-supported frames, calibration or anchor identities, technical
support observations, and descriptive metadata. The complete finite set is
content-addressed before selection.

A design request explicitly records that prediction payloads were not opened
and neither provider residuals nor target outcomes were used. The embedded
feasibility requests enforce the same information boundary independently.

## Deterministic selection

A support-feasible candidate always outranks a support-negative candidate. Within
the same feasibility class, candidates are ordered by:

1. maximum minimum support fraction over complete frozen groups;
2. maximum number of groups with at least one supported stream;
3. maximum number of required camera/frame cells in fully supported streams;
4. maximum number of geometry-supported required camera/frame cells;
5. maximum supported-stream count;
6. minimum maximum causal-prefix span;
7. minimum total causal-prefix span; and
8. lexicographically smallest frozen candidate ID.

The result retains every candidate's complete replayed support-feasibility result
and ranking statistics. Loading the artifact recomputes the entire selection and
rejects any mismatch.

## Python API

```python
from pathlib import Path

from prob4d.provider_support_design import (
    ProviderSupportDesignCandidateV1,
    ProviderSupportDesignRequestV1,
    evaluate_provider_support_design,
    write_provider_support_design,
    write_provider_support_design_request,
    write_selected_provider_support_feasibility,
    write_selected_provider_support_request,
)

candidate = ProviderSupportDesignCandidateV1(
    candidate_id="prefix-0-8",
    feasibility_request=frozen_support_request,
    metadata={"window_geometry": "25-frame-overlap-8"},
)
request = ProviderSupportDesignRequestV1(
    protocol_id="fresh-provider-support-design-v1",
    candidates=(candidate,),
    prediction_payloads_opened=False,
    residuals_used=False,
    target_outcomes_used=False,
)

write_provider_support_design_request(
    Path("provider-support-design-request.json"),
    request,
)
result = evaluate_provider_support_design(request)
write_selected_provider_support_request(
    Path("selected-provider-support-request.json"),
    result,
)
write_selected_provider_support_feasibility(
    Path("selected-provider-support-result.json"),
    result,
)
write_provider_support_design(
    Path("provider-support-design-result.json"),
    result,
)
```

The selected request and feasibility result use the unchanged existing schemas.
They can therefore enter `ProviderPromotionAuthorizationV2` without rewriting
that authorization or any historical support artifact.

## Command-line selection and replay

The module is executable directly and does not add a legacy console-script alias:

```bash
python -m prob4d.provider_support_design select \
  --request provider-support-design-request.json \
  --output provider-support-design-result.json \
  --selected-request-output selected-provider-support-request.json \
  --selected-feasibility-output selected-provider-support-result.json

python -m prob4d.provider_support_design verify \
  --artifact provider-support-design-result.json
```

Both commands print a deterministic summary. Exit status `0` means the selected
candidate is support-feasible. Exit status `2` is a valid result in which every
frozen candidate is support-negative; the best negative remains selected and
fully reported. Malformed or tampered artifacts raise validation errors instead
of being relabelled as support-negative.

## Scientific boundary

This contract does not rescue the completed Deform360 source-support negative or
permit post-outcome camera deletion, prefix changes, threshold relaxation, or
partial-stream fitting. A later route must be a separately versioned prospective
provider experiment whose complete design candidate set is frozen before later
information is opened.

# Outcome-blind provider support feasibility

`prob4d.provider_support_feasibility` records whether a frozen prospective
provider cohort has the metadata, geometry, and metric support required for a
later provider-competence experiment.

This check belongs **before** prediction payloads, provider residuals, or target
outcomes are opened. A passing result is only a support-feasibility statement;
it is not evidence that the provider is accurate, calibrated, beneficial to
BayesianPhysTwin or Causal4D, or safe to deploy.

## What is bound

Each request is content-addressed and binds:

- the Prob4D, provider, model-set, loader, cohort, and promotion-lock identities;
- the exact group/stream roster and causal frame interval;
- required, available, and geometry-supported frame IDs;
- required camera intrinsics, camera extrinsics, and independent metric anchors;
- an `all-streams` or predeclared minimum-fraction admission rule; and
- the only technical exclusion codes and count permitted by the protocol.

The request must state that prediction payloads were not opened and that neither
provider residuals nor target outcomes were used. Any frame outside the frozen
causal interval is rejected during construction and replay.

## Python API

```python
from pathlib import Path

from prob4d.provider_support_feasibility import (
    ProviderSupportFeasibilityRequestV1,
    ProviderSupportStreamV1,
    evaluate_provider_support_feasibility,
    write_provider_support_feasibility,
    write_provider_support_feasibility_request,
)

stream = ProviderSupportStreamV1(
    group_id="target-object-00",
    stream_id="camera-0",
    causal_frame_start=0,
    causal_frame_stop_exclusive=8,
    required_frame_ids=tuple(range(8)),
    available_frame_ids=tuple(range(8)),
    geometry_supported_frame_ids=tuple(range(8)),
    minimum_geometry_support_fraction=1.0,
    intrinsics_required=True,
    intrinsics_id="1" * 64,
    extrinsics_required=True,
    extrinsics_id="2" * 64,
    metric_anchor_required=True,
    metric_anchor_id="3" * 64,
)

request = ProviderSupportFeasibilityRequestV1(
    protocol_id="fresh-provider-support-v1",
    source_repository="IPS-Stuttgart/Prob4D",
    source_revision="a" * 40,
    provider_family="external-4d-provider",
    provider_repository="example/provider",
    provider_revision="b" * 40,
    model_set_id="4" * 64,
    loader_id="5" * 64,
    cohort_binding_id="6" * 64,
    promotion_lock_id="7" * 64,
    coordinate_semantics="metric-world-frame",
    admission_rule="all-streams",
    minimum_supported_fraction=1.0,
    permitted_technical_exclusion_codes=(),
    maximum_technical_exclusions=0,
    prediction_payloads_opened=False,
    residuals_used=False,
    target_outcomes_used=False,
    streams=(stream,),
)

write_provider_support_feasibility_request(
    Path("provider-support-request.json"),
    request,
)
result = evaluate_provider_support_feasibility(request)
write_provider_support_feasibility(
    Path("provider-support-result.json"),
    result,
)
```

## Command-line replay

The module can be executed directly without introducing another historical
console-script alias:

```bash
python -m prob4d.provider_support_feasibility evaluate \
  --request provider-support-request.json \
  --output provider-support-result.json

python -m prob4d.provider_support_feasibility verify \
  --artifact provider-support-result.json
```

Both commands print a deterministic summary. They return exit status `0` for a
support-feasible result and `2` for a valid support-negative result. Invalid or
tampered artifacts fail validation instead of being treated as support-negative.

## Decision semantics

A stream is support-positive only when every required frame is available, its
geometry-supported fraction reaches the frozen threshold, all required
intrinsic/extrinsic/metric-anchor identities are present, and no technical
failure remains.

A predeclared technical failure may be removed from the admission denominator
only when its code is explicitly permitted and the frozen exclusion budget is
not exceeded. Undeclared failures remain in the denominator. Exceeding the
technical exclusion budget fails the complete cohort.

The result embeds the complete request and per-stream reason codes, then hashes
the canonical replayable representation. Loading the result recomputes the
entire decision and rejects any mismatch.

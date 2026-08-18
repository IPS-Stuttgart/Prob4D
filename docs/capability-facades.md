# Capability-specific Python façades

Prob4D's normative ecosystem boundary remains `prob4d.api.v2`. Five narrower,
collision-free version-1 preview façades now mirror curated subsets of that API
for consumers that need only one capability:

| Capability | Import |
| --- | --- |
| Similarity-transform geometry | `prob4d.api.geometry_v1` |
| Portable artifacts and streams | `prob4d.api.artifacts_v1` |
| Structured covariance queries | `prob4d.api.covariance_v1` |
| Calibration contracts | `prob4d.api.calibration_v1` |
| Provider admission and attestation | `prob4d.api.provider_v1` |

For example:

```python
from prob4d.api.covariance_v1 import project_observation_covariance
from prob4d.api.geometry_v1 import Sim3
```

Every exported object is the identical object exposed by `prob4d.api.v2`; the
façades contain no wrapper implementations and do not change serialization,
estimator, calibration, or fallback semantics. Importing them also retains the
lightweight runtime boundary and does not load Torch, Diffusers, or Decord.

## Lifecycle

Each module declares:

```python
FACADE_VERSION = 1
LIFECYCLE = "preview"
```

Preview means that the import split is available for integration testing but is
not yet listed as an independent compatibility surface in the content-addressed
public API manifest. `prob4d.api.v2` remains the required dependency boundary for
claim-bearing external integrations. Promotion should occur only after installed
wheel and three-repository consumer tests have exercised the narrower imports.

The explicit `__all__` inventory of every façade is regression-tested. A future
breaking change must use a new module version rather than silently changing a
promoted façade.

## Ownership boundary

The split is organizational rather than scientific:

- geometry owns `Sim(3)` representation and point Jacobians;
- artifacts owns portable records and persistence operations;
- covariance owns exact structured covariance actions and projections;
- calibration owns source-fitted calibration contracts; and
- provider owns strict admission, provider identity, and runtime attestation.

None of these import paths establishes provider competence, calibrated physical
uncertainty, BayesianPhysTwin benefit, Causal4D intervention benefit, deployment
safety, or state of the art.

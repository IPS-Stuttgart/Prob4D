# Public Python API

Prob4D separates evolving implementation modules from versioned ecosystem
interfaces. Downstream code should import the current façade explicitly:

```python
from prob4d.api import v2 as prob4d_api
```

## Minimal package root

Prob4D 0.5 removes the historical broad root façade. Importing `prob4d` exposes
only the installed version:

```python
import prob4d

print(prob4d.__version__)
```

Objects such as `Sim3`, observation contracts, calibration records, and factor
bundles are no longer available through `from prob4d import ...`. Use
`prob4d.api.v2` for supported downstream work or an owning implementation module
inside Prob4D itself. The packaged `prob4d/__init__.pyi` mirrors this minimal
runtime surface.

## Version 2 boundary

`prob4d.api.v2` is the current ecosystem surface for calibrated or explicitly
exploratory provider-v2 work. It exposes:

- strict claim-bearing observation loading and provider attestation;
- schema-v4 explicit-gauge factor bundles and append-only factor streams;
- sparse and tree-sparse execution contracts and strict loaders;
- portable causal gauge-tree priors and artifacts;
- matrix-free observation covariance actions and linear-query projections;
- source-only group-aware pathwise calibration and equal-group diagnostics;
- `Sim3`, project identity, and required serialization helpers; and
- explicit provider-v2 export and validation contracts.

Experiment runners, command dispatch, MotionCrafter internals,
provider-evaluation studies, and paper-specific evidence code remain outside the
façade. A breaking current ecosystem change requires a new façade such as
`prob4d.api.v3`; it must not silently alter `prob4d.api.v2`.

## Structured covariance queries and pathwise diagnostics

`project_observation_covariance` computes the exact requested
`A @ Sigma @ A.T` from dense, sparse, or tree-sparse observation factors without
materializing the full observation covariance. The result separates conditional
point uncertainty from shared gauge uncertainty.

`fit_pathwise_maximum_calibration` and `pathwise_uncertainty_diagnostics` require
one frozen physical-object or acquisition-session ID per trajectory. The
calibration maximum, conformal rank, target coverage, and Gaussian score are
computed at that independent-group level. The immutable calibration record binds
the complete trajectory-to-group assignment and rejects overlapping target group
IDs; track count cannot inflate the finite-sample sample size.

See [structured observation covariance queries](observation-covariance-queries.md)
for semantics, examples, and the BayesianPhysTwin/Causal4D ownership boundary.

## Historical provider-v1 artifacts

`prob4d.api.v1` and provider-v1 execution are not part of Prob4D 0.5. Pin the
exact Prob4D 0.4.1 wheel or source revision to reproduce a provider-v1 run.

The 0.5 package retains a narrow `prob4d.provider_v1` artifact compatibility
bridge for immutable v1 record types, manifests, serializers, validators, and
schema-v3 factor IO required by frozen evidence. It does not expose the old
provider estimator or exporters and is not a supported boundary for new code.

## Machine-readable inventory

Generate the content-addressed API manifest from the exact installed wheel:

```bash
python -m prob4d.public_api_manifest print
python -m prob4d.public_api_manifest build \
  --output public-api-manifest.json
python -m prob4d.public_api_manifest verify \
  public-api-manifest.json \
  --require-current
```

Manifest schema version 2 records the minimal package root and current v2
façade. See [public API manifest](public-api-manifest.md).

## Version reporting

Runtime version reporting is resolved from installed distribution metadata. An
uninstalled source tree reports `0+unknown`. Release versions are declared in
`pyproject.toml` and checked against `CITATION.cff` during release validation.

# Public Python API

Prob4D has two different compatibility concerns:

1. experiment and research modules evolve with the active scientific programme;
2. downstream observation contracts must remain stable for BayesianPhysTwin,
   Causal4D, and independent providers.

Downstream projects should therefore import the versioned façade:

```python
from prob4d.api import v1 as prob4d_api
```

The broad package root remains available for interactive use and historical
compatibility, but it is not the supported dependency boundary for new ecosystem
code. A breaking change to the façade requires a new module such as
`prob4d.api.v2`; it must not silently change `prob4d.api.v1`.

## Ownership boundary

`prob4d.api.v1` owns calibrated probabilistic 4D observations, portable
observation and factor contracts, covariance calibration artifacts, provider
manifests, and strict serialization helpers.

BayesianPhysTwin owns physical-prior fusion, guarded state updates, and exact
fallback decisions. Causal4D owns realized-intervention abduction and
counterfactual prediction. Neither downstream repository should reinterpret raw
Prob4D covariance or depend on experiment-specific Prob4D modules.

## Version reporting

Runtime version reporting is resolved from installed distribution metadata. An
uninstalled source tree reports the explicit sentinel `0+unknown` rather than a
second hard-coded release number. Release versions remain declared once in
`pyproject.toml` and are checked against `CITATION.cff` during release validation.

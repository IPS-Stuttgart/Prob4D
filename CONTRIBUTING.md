# Contributing to Prob4D

Prob4D is research software with versioned numerical and cross-repository contracts.
Changes should preserve the distinction between implementation evidence and empirical
scientific evidence.

## Development setup

```bash
python -m pip install -e ".[dev]"
python -m pytest
python -m ruff check src tests
```

The lightweight package must remain importable without Torch, Diffusers, Decord, or a
MotionCrafter checkout. GPU and model-loading dependencies belong behind lazy imports.

## Contract changes

`prob4d.provider_v1` is frozen for existing experiments. New claim-bearing development
uses `prob4d.provider_v2`. A breaking Python provider change requires a new provider
module; a changed artifact interpretation requires a new schema or causal-stream
contract version. Do not silently reinterpret historical artifacts.

Changes to covariance, reliability, source lineage, stochastic seed semantics, model
identity, or evidence admission require focused adversarial tests and corresponding
changelog documentation. Exact fallback and causal-prefix invariants must remain
fail-closed.

## Privileged validation

Ordinary pull-request workflows must remain on ephemeral GitHub-hosted runners. Do not
route pull-request source to a persistent self-hosted runner through a branch prefix,
changed workflow, or mutable runner-selection input.

When one explicitly reviewed revision genuinely requires the local GPU, large memory,
or approved data, use the protected `Trusted exact-head validation` workflow described
in `docs/trusted-self-hosted-validation.md`. The workflow must be dispatched from
`main`, bind the current open same-repository pull-request head by its full SHA, and
pause at the independently reviewed `trusted-self-hosted-validation` environment before
self-hosted checkout. A successful privileged run is implementation evidence only; it
does not authorize target access or promote a scientific claim.

## Pull requests

A pull request should describe:

- the implementation or contract change;
- the failure mode or research question it addresses;
- compatibility and claim boundaries;
- validation performed; and
- any calibration artifacts or frozen experiments that must be regenerated.

Generated predictions, model weights, large datasets, videos, and raw experiment
outputs must remain outside Git. Compact evidence belongs in the canonical project-notes
repository only after its source and claim boundary are frozen.

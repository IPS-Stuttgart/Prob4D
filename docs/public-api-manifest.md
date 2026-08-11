# Public API manifest

Prob4D exposes a content-addressed, machine-readable inventory of the Python
surfaces intended for downstream compatibility checks. The manifest records:

- the installed Prob4D version and transfer-safe project identity;
- the historical package-root export inventory and its lazy loading semantics;
- every export from `prob4d.api.v1`;
- every export from `prob4d.api.v2`; and
- the provider and provider-factor API versions attached to those façades.

Generate the manifest from the exact installed wheel or source distribution that
will be used downstream:

```bash
python -m prob4d.public_api_manifest print

python -m prob4d.public_api_manifest build \
  --output public-api-manifest.json
```

Validate persisted bytes structurally:

```bash
python -m prob4d.public_api_manifest verify public-api-manifest.json
```

For release or integration checks, also require equality with the executing
installation:

```bash
python -m prob4d.public_api_manifest verify \
  public-api-manifest.json \
  --require-current
```

The writer is no-clobber. Repeating an identical build is idempotent; an existing
different or malformed artifact is rejected instead of being overwritten.

## Compatibility interpretation

The broad `prob4d` package root is a historical compatibility surface. Importing
it is lightweight: implementation modules are loaded only when their exported
attributes are accessed. Its complete typing surface is retained in
`prob4d/__init__.pyi`.

New integrations should still use `prob4d.api.v1` or `prob4d.api.v2`. The
manifest makes accidental additions, removals, provider-version changes, and
packaging drift visible without promoting the broad root into the preferred
dependency boundary.

## Claim boundary

A valid manifest establishes exact installed Python export and version metadata.
It does not establish provider accuracy, uncertainty calibration, physical-query
improvement, BayesianPhysTwin admission, Causal4D intervention benefit,
deployment safety, or state of the art.

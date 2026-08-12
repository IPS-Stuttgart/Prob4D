# Public API manifest

Prob4D exposes a content-addressed machine-readable inventory of supported
Python surfaces. Manifest schema version 2 records:

- the installed Prob4D version and transfer-safe project identity;
- the minimal `prob4d` package root, which exports only `__version__`;
- every export from the current `prob4d.api.v2` façade; and
- provider and provider-factor API versions.

Generate the manifest from the exact installed wheel or source distribution:

```bash
python -m prob4d.public_api_manifest print
python -m prob4d.public_api_manifest build \
  --output public-api-manifest.json
```

Validate persisted bytes structurally, or require equality with the executing
installation:

```bash
python -m prob4d.public_api_manifest verify public-api-manifest.json
python -m prob4d.public_api_manifest verify \
  public-api-manifest.json \
  --require-current
```

The writer is no-clobber. Repeating an identical build is idempotent; an existing
different or malformed artifact is rejected.

## Compatibility interpretation

The package-root surface has version 2 and loading mode
`minimal-version-root-v1`. Its exact export roster is:

```json
["__version__"]
```

`prob4d.api.v2` is labelled `current`. `prob4d.api.v1` is absent from a 0.5
manifest. A schema-v1 manifest from Prob4D 0.4.x remains valid evidence about
that exact older installation, but a 0.5 installation does not accept it as its
current manifest. Historical manifests must not be rewritten.

## Claim boundary

A valid manifest establishes exact installed Python export and version metadata.
It does not establish provider accuracy, uncertainty calibration, physical-query
improvement, BayesianPhysTwin admission, Causal4D intervention benefit,
deployment safety, or state of the art.

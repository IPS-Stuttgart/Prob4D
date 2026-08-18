# Distribution boundaries

Prob4D publishes an installable Python distribution and separate scientific
evidence artifacts.

## Python distribution

The wheel and source distribution contain the library, typed contract data,
project metadata, documentation, and frozen protocol descriptions. They do not
contain GitHub workflows, generated evidence, CI environments, repository tests,
or one-off maintenance scripts.

The package includes `py.typed` and a deliberately minimal root
`__init__.py`/`__init__.pyi`. The supported ecosystem boundary is
`prob4d.api.v2`.

Every built wheel and source distribution also carries a generated
`prob4d/_build_identity.json`. It binds an exact clean source revision to a
SHA-256 manifest of the installed package bytes. A wheel rebuilt from a source
distribution inherits the original revision and recomputes the installed-package
manifest. See [release build identity](release-build-identity.md).

Prob4D 0.5 installs one executable:

```text
prob4d
```

It does not install standalone `prob4d-*` aliases. It also omits
`prob4d.api.v1`, `prob4d.legacy_cli`, and provider-v1 execution/export entry
points.

A narrow `prob4d.provider_v1` artifact compatibility bridge remains in the
library because immutable historical observation and schema-v3 factor artifacts
must still be inspectable by the installed-wheel three-repository capsule. It is
not an estimator or exporter. Pin Prob4D 0.4.1 for full v1 execution.

## Scientific evidence

Claim-bearing evidence must remain content-addressed and bind exact source
revisions, distribution hashes, provider/model identity, protocol identity,
calibration and fallback artifacts, dataset split, and all input/output digests.
Generated evidence is not bundled into the Python distribution merely because a
workflow produced it.

The embedded build identity is one input to that boundary. It detects package-byte
or source-cleanliness drift, but it does not replace signed release tags, external
distribution digests, an SBOM, build provenance, or the three-repository evidence
capsule.

## Source-distribution audit

`scripts/ci/check_sdist.py` verifies a single safe archive root, rejects links and
repository-only paths, requires the current release documentation and contract
data, installs the archive in an isolated environment, and smoke-tests:

- the minimal package root;
- the current `prob4d.api.v2` façade;
- the historical artifact compatibility bridge without v1 exporters;
- the content-addressed public API manifest; and
- the single grouped command surface.

The separate release-build-identity workflow verifies that both the ordinary
wheel and a wheel rebuilt from the source distribution attest the exact clean
checkout and reject installed-package tampering.

Distribution conformance is infrastructure evidence only.

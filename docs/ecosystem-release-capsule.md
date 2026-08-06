# Three-repository installed-wheel release capsule

Prob4D, BayesianPhysTwin, and Causal4D already exercise an installed-wheel golden
path owned by BayesianPhysTwin. Prob4D's `Ecosystem installed-wheel release
capsule` workflow wraps that path in a compact, content-addressed evidence
artifact.

The workflow checks out one exact revision of each repository, builds one wheel
from each clean source archive, installs only those wheels into an isolated
Python environment, and runs the existing three-repository integration tests.
After the tests pass, it records:

- the exact three repository revisions;
- each built wheel filename and SHA-256 digest;
- the provider and portable artifact contract versions;
- the Python and runner identities; and
- the GitHub Actions run identity.

The resulting `ecosystem-release-capsule.json` is deeply validated and carries a
`capsule_id` computed from canonical JSON. A changed revision, wheel byte stream,
contract version, or execution identity therefore produces a different capsule.
The workflow also retains the complete golden-path log beside the compact JSON.

## Running the workflow

Use the workflow's manual dispatch and select the BayesianPhysTwin and Causal4D
branch, tag, or commit to validate. The selected refs are resolved to exact commit
SHAs before the capsule is written. Prob4D is always the exact reviewed workflow
revision: a pull-request head on pull-request events and `github.sha` otherwise.

The workflow also runs weekly and whenever the release-capsule implementation or
one of Prob4D's stable provider/observation boundaries changes.

## Local capsule tooling

A previously retained golden-path log can be sealed with:

```bash
python scripts/ci/build_ecosystem_release_capsule.py build \
  --golden-path-log ecosystem-golden-path.log \
  --prob4d-revision <40-hex-commit> \
  --bayesian-phystwin-revision <40-hex-commit> \
  --causal4d-revision <40-hex-commit> \
  --python-version 3.12.11 \
  --runner-os Linux \
  --run-id 123456789 \
  --run-attempt 1 \
  --run-url https://github.com/IPS-Stuttgart/Prob4D/actions/runs/123456789 \
  --output ecosystem-release-capsule.json

python scripts/ci/build_ecosystem_release_capsule.py verify \
  ecosystem-release-capsule.json
```

The builder rejects missing or conflicting wheel hashes, noncanonical scalar
aliases, duplicate JSON keys, unknown fields, malformed repository revisions,
and a capsule whose identity does not reproduce from its declared content. It
writes atomically and refuses to replace a different existing capsule.

## Claim boundary

A passing capsule proves that the declared wheel bytes interoperate through the
retained golden path. It is infrastructure and provenance evidence only. It does
not establish Prob4D observation accuracy, calibrated uncertainty,
BayesianPhysTwin physical-query improvement, Causal4D intervention benefit,
deployment safety, or state of the art.

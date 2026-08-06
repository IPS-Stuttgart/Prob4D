# Security policy

## Supported versions

Security and integrity fixes are applied to the current `0.3.x` development line.
Frozen historical revisions remain reproducible but may not receive backports.

## Self-hosted workflow boundary

Ordinary pull-request workflows run on GitHub-hosted infrastructure. Pull-request
source, branch names, and workflow inputs must not select or trigger a persistent
self-hosted runner.

The only approved path for executing reviewed pull-request source on a self-hosted
runner is the manual `Trusted exact-head validation` workflow. It must be dispatched
from `main`, verify an open same-repository pull request and its exact current
40-character head SHA on a hosted runner, and use the protected
`trusted-self-hosted-validation` environment before checkout on the self-hosted host.
The environment must have an independent required reviewer and no attached secrets or
write credentials. See `docs/trusted-self-hosted-validation.md` for the complete
operational and host-hardening requirements.

Environment approval is not a sandbox. Suspected access to unrelated runner files,
credentials, datasets, services, or network resources is a security incident even when
repository and workflow permissions were read-only.

## Automated scanning

The repository runs pinned CodeQL analysis for Python and GitHub Actions plus a strict,
pinned `pip-audit` dependency scan on pull requests, default-branch pushes, a weekly
schedule, and explicit dispatch. Workflow-policy tests require immutable action pins,
disabled checkout credential persistence, and the protected self-hosted boundary.

## Reporting a vulnerability

Do not open a public issue for a suspected vulnerability involving credential exposure,
artifact path traversal, unsafe deserialization, workflow privilege, model-source
integrity, evidence tampering, or self-hosted runner access.

Use GitHub's private vulnerability-reporting or Security Advisory interface for this
repository. Include the affected revision, a minimal reproduction, the expected
security boundary, and whether any private data, token, model artifact, or evidence
bundle may have been exposed. If private reporting is unavailable, contact the
repository maintainers through a private institutional channel.

The maintainers will acknowledge the report, assess affected versions and artifacts,
and coordinate a fix and disclosure. Scientific correctness concerns without a
security impact should use an ordinary issue and retain the repository's explicit
claim boundaries.

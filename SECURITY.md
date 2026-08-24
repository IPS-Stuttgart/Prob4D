# Security policy

## Supported versions

Security and integrity fixes are applied to the current `0.5.x` development line.
Frozen historical revisions remain reproducible but may not receive backports.

## Legacy PhysTwin dataset boundary

The `PhysTwinCase` directory adapter still needs the official dataset's
historical `calibrate.pkl` and `processed_masks.pkl` files. Prob4D loads these
only inside that dedicated legacy adapter, rejects symbolic-link substitution,
and uses a restricted unpickler that admits primitive containers plus the
minimal NumPy array-reconstruction globals. Arbitrary Python globals fail
closed. Portable Prob4D prediction, calibration, observation, and evidence
artifacts never use pickle.

The restriction prevents ordinary pickle code execution through those adapter
paths; it is not a general sandbox against malformed or resource-exhausting
files. The legacy PhysTwin experiment and state diagnostics use the same
restricted loader. Use only locally verified official dataset files and retain
their hashes in claim-bearing run provenance.

A fail-closed source policy rejects new direct `pickle.load`, `pickle.loads`, and
NumPy `allow_pickle=True` calls. The sole retained exception is the exact
historical CUT3R Deform360 source-freeze-v1 builder. That builder is bound to a
frozen source-only protocol and revision and must not be copied into a new
claim-bearing path. A future execution that changes this boundary requires a
separately versioned protocol and safe non-object NPZ/JSON conversion before
claim-bearing execution; it may not silently reinterpret the frozen request.

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
operational and host-hardening requirements. Repository setup and acceptance testing are
tracked in issue #157.

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

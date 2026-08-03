# Security policy

## Supported versions

Security and integrity fixes are applied to the current `0.3.x` development line.
Frozen historical revisions remain reproducible but may not receive backports.

## Reporting a vulnerability

Do not open a public issue for a suspected vulnerability involving credential exposure,
artifact path traversal, unsafe deserialization, workflow privilege, model-source
integrity, or evidence tampering.

Use GitHub's private vulnerability-reporting or Security Advisory interface for this
repository. Include the affected revision, a minimal reproduction, the expected
security boundary, and whether any private data, token, model artifact, or evidence
bundle may have been exposed. If private reporting is unavailable, contact the
repository maintainers through a private institutional channel.

The maintainers will acknowledge the report, assess affected versions and artifacts,
and coordinate a fix and disclosure. Scientific correctness concerns without a
security impact should use an ordinary issue and retain the repository's explicit
claim boundaries.

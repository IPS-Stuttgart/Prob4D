# Executed workflow provenance

These workflow definitions are inert snapshots of the file-change-triggered
runs used to produce the source-only and held-out DEFORM DLO4/DLO5 evidence.
They are stored outside `.github/workflows` deliberately: the registered runs
are complete, the evaluation outcomes have been opened, and silently rerunning
or retuning the experiment must not become normal repository automation.

The permanent active check is
`.github/workflows/deform-dlo45-evidence-integrity.yml`. It validates formatting,
mechanism tests, action pins, result identities, the frozen source-to-evaluation
information order, exact fallback behavior, and the explicit covariance and
provider-competence limitations.

The authoritative execution identities are recorded in
`../summary.json` and `../../../docs/deform-dlo45-query-observability-heldout-v1.md`.

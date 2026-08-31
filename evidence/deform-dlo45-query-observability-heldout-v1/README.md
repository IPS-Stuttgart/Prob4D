# Held-out DEFORM DLO4/DLO5 evidence package

This directory is the machine-readable evidence package for the source-frozen
query-aware observability experiment documented in
`../../docs/deform-dlo45-query-observability-heldout-v1.md`.

- `summary.json` contains the retained aggregate held-out result, confidence
  intervals, criteria, limitations, execution identity, and claim boundary.
- `validation-manifest.json` declares the identities and checks that may not
  change without creating a new experiment.
- `ci-validation.json` records the successful merge-tree validation pass.
- `executed-workflows/` preserves the exact file-change-triggered execution
  definitions as inert provenance. They are intentionally not active Actions.

The official DLO4/DLO5 evaluation outcomes have been opened. No threshold,
support, query, prior, comparator, seed, or method may be retuned from this
package. The evidence concerns controlled gauges on held-out real geometry; it
does not establish learned visual-provider competence or calibrated accepted-
query covariance.

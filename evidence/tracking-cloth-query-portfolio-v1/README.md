# Tracking-Cloth query-portfolio study v1

This directory preserves the compact, machine-readable receipt for the hosted
real-trajectory query-portfolio experiment from Prob4D PR #402. The execution
used the official Tracking-Cloth Zenodo release and copied no raw trajectories
into the output artifact.

- empirical source revision: `3dfe9726b534f2dd556a706afeaaf4e69027ccf4`;
- workflow run: `33364480389`;
- hosted job: `99402284665`;
- artifact ID: `9747804017`;
- artifact archive SHA-256:
  `310775ea119a86e9728f2b78e82cb46e7b86fb9c657cfbb5a1237278586a8efc`;
- dataset DOI: `10.5281/zenodo.14644526`.

The archive covers 80 accepted cloth recordings, five recording-disjoint folds,
55 fold/portfolio evaluations, and 7,648 unique held-out windows. Exact
query-sufficient compression reproduced the full posterior with maximum relative
gain error `4.0804182725523995e-12`, maximum relative posterior-covariance error
`5.383026917575375e-13`, and maximum realized posterior-mean difference
`5.797456725464489e-13` metres.

Across the registered portfolios, the retained query rank ranged from 3 to 60.
The shared factor alone occupied 50.0% to 95.24% of the corresponding joint-cache
payload. Once the query-projection payload is included, however, none of these
finite portfolios was smaller than the joint cache. Cached online updates were
also about 13.5 to 17.0 times faster than reconstructing from compressed factors.
The result therefore validates exact query-sufficient posterior transport and
locates its finite-size storage/runtime break-even; it does not claim that factor
transport dominates a precomputed cache for the tested fixed portfolios.

`summary.json` retains the exact aggregate rows and SHA-256 identities of every
scientific artifact member. The separately queued local-mirror job is not claim
evidence and was not touched during archival.

## Claim boundary

This study establishes a recording-disjoint systems result on fixed queries and
real motion-capture geometry. It does not establish deployment calibration,
learned-provider quality, BayesianPhysTwin benefit, Causal4D benefit, or target
performance. The original frozen source files and hosted artifact remain the
authoritative empirical evidence; this directory is an immutable compact
archive, not a configuration input.

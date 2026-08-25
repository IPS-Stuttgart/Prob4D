# CUT3R source-comparison runtime amendment v1

The original development smoke terminated before source video verification or
decode because the launcher exposed the CUT3R repository root but not its
internal `src` package root. The retained result records one attempt, zero
decoded frames, zero CUT3R inference, zero predictions, zero output members, and
no source-truth or target access. That case remains permanently non-retriable.

The v1.1 amendment changes only runtime import/bootstrap and artifact-custody
code. It preserves the original source roster, CUT3R revision and checkpoint,
window schedule, three compared arms, random seeds, alignment, gauge fusion,
uncertainty construction, information boundary, and no-replacement rule.

The amended execution plan deterministically registers the first
lexicographically ordered development case whose SHA-256 identity differs from
the retained failed case. The runner rejects every other smoke case. Exactly one
attempt is permitted. A successful run must then pass the independent custody
verifier, with no decoded images retained, before either source shard can be
authorized. A second technical failure is retained and closes this route.

This amendment is not a scientific result and does not authorize source truth,
target data, BayesianPhysTwin, Causal4D, or scoring. It only creates a guarded
opportunity to obtain the first valid CUT3R prediction artifact under the frozen
comparison.

The server-runtime plan has content identity
`ab460acf8ba85d8e5470126e6e9e2fc445d16ad506612b10f1a926a614c60f98`
and file SHA-256
`d4eceb6a44f154901227f1e2ac0e832874869179e55f74ce18f10d6a352d6b00`.
It binds implementation revision
`83ce1c546d4c7d0ebca740334a8ad969666a1d0c` and records only the SHA-256
identity of the registered replacement smoke in public summaries.

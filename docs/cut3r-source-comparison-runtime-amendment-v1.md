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

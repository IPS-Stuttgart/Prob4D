# CUT3R source-comparison runtime amendment v1.2

The v1.1 execution plan was independently rejected before execution because its
successful-smoke custody requirement and one-attempt limit were declarative.
The runner could select a full source shard without receiving or revalidating a
successful smoke receipt, and a new output root could bypass the registered
attempt count. Zero v1.1 smoke, source-shard, source-outcome, or target
executions were authorized or performed.

The v1.2 plan preserves the frozen 40-case source roster, CUT3R revision and
checkpoint, window schedule, compared arms, seeds, alignment, fusion,
uncertainty construction, replacement development case, and information
boundary. It changes only execution custody:

- the plan binds one exact smoke output root and one independent attempt-ledger
  path by canonical path SHA-256;
- the runner atomically consumes the write-once ledger before CUT3R runtime
  initialization and rejects every other output root or ledger;
- a second launch is rejected even when a fresh output root is supplied; and
- full source shards require the registered attempt record plus an independently
  published ordinary-success custody receipt that the runner recomputes from the
  complete retained smoke tree and shard report.

The corrected plan has content identity
`1c74105994cb32885c8ab57af5a2d1feb296ce63c5e9a3fc39665fb2094bad3a`
and file SHA-256
`172e725e3ecfe12dac7c0e402f01883dd92d6898179e89079f21f2b3c05e680c`.
It binds implementation revision
`06f7a19c2016b291f85bf17634a515ffc4890ac1` and records only path and case
SHA-256 identities in public summaries.

This amendment authorizes no source truth, source scoring, target payload,
BayesianPhysTwin, Causal4D, deployment, or SOTA claim. The sole different
development smoke remains unauthorized until independent review passes the
new implementation and plan.

# Recursive task-sufficient compression status

Status: controlled mechanism implementation on research branch; no real-data or paper promotion yet.

Completed on this branch:

- exact audited LTI task-state closure kernel;
- adversarial recursive parity test separating current-query sufficiency from recursive sufficiency;
- controlled protocol freezing the expected dimensions, ranks, parity gates, and failure control;
- deterministic science script producing content-addressable result JSON; and
- explicit prior-art and next-experiment boundaries.

The controlled design has state dimension 20, current-task dimension 3, recursive closure dimension
4, observation dimension 12, and supplied shared-factor rank 8. The registered mechanism expects
closure-aware factor rank 4 and current-query-only factor rank 3. The current query is required to
match at the first update, while the rank-3 control must subsequently diverge because an omitted
closure coordinate drives a future task coordinate. Closure-aware recursion must remain numerically
identical to the full task-state posterior.

No claim should be moved into the ICRA manuscript until the branch passes repository CI and the
controlled study is executed at an immutable revision. A real/closed-loop claim requires the
separately preregistered experiment described in
`docs/recursive-task-sufficient-compression-next-experiment.md`.

# Exact CUT3R safe-directory replacement

The retained CUT3R source-freeze workflow reached the configured external CUT3R
checkout on `workstation2`, but Git rejected revision resolution before the
source-freeze builder could publish either a support-positive or support-negative
decision. The latest retained failure evidence is bound to workflow run
`32771242880`, attempt `3`, execute job `97709306705`, and artifact digest
`sha256:68f22308ab86190d68c196b747ffcf6f217e670a07da15377da35b7bbb61b57e`.
The earlier attempt-2 diagnostic is also admitted by exact ID, size, and digest;
any additional artifact fails closed.

Attempt 3 proved that pre-populating an attempt-specific home is insufficient:
the historical target workflow exports a newly isolated `HOME` before invoking
the builder. The replacement therefore changes only the current reviewed control
plane. It resolves the configured CUT3R checkout canonically and passes Git's
protected configuration directly to the single historical driver invocation via
`GIT_CONFIG_COUNT`, with `safe.directory` equal to that exact path. It writes no
system or global configuration, uses no `sudo`, and permits no wildcard trust.

The historical execution revision, request ID, source protocol, source/target
roster, checkpoint, wheel, windowing, and information boundary remain frozen.
One exact issue-49 command may dispatch a fresh current-main workflow run. The
hosted dispatcher revalidates the attempt-3 run, complete job and execute-step
rosters, both diagnostic artifacts, historical request authorization, and the
scoped-trust workflow text. Any newer target retry produces a no-op receipt
rather than another dispatch.

This stage remains source-only and target-closed. It does not itself execute
CUT3R, open source predictions or residuals, open confirmation data, authorize
target access, or support a scientific performance claim.

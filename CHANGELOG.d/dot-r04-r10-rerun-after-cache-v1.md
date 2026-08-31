# Fail-closed recovery after DOT R01–R10 cache prewarm

Add a hosted recovery workflow for the exact frozen R04–R10 CUT3R confirmation run. After the checksum-bound cache-only prewarm completes successfully, the workflow independently verifies its receipt and inspects the frozen target run.

It requests `rerun-failed-jobs` only when the target has completed as a technical failure or cancellation, remains below the bounded attempt ceiling, and has no terminal scientific result artifact. Active or successful target runs are left untouched. No repository content, protocol, request, provider identity, cohort, threshold, marker order, prediction, or outcome is modified.

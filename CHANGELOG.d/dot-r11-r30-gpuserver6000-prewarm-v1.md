# DOT R11–R30 gpuserver6000 prewarm

Add a fail-closed, request-triggered operational workflow that prepares the exact frozen CUT3R runtime and downloads the official compressed `R11-20.zip` and `R21-30.zip` archives on `gpuserver6000`.

The workflow verifies publisher checksums, keeps the archives unopened, never reads normal-view images or marker payloads, constructs no predictions, and performs no scientific evaluation. It only removes runtime and transfer latency from the separately frozen R11–R30 query-selective experiment; it does not change that experiment's protocol, cohort, methods, thresholds, or decision rules.

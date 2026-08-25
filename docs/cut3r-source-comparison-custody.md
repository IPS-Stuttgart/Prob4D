# CUT3R source-comparison custody gate

The frozen CUT3R executor writes one content-addressed case directory at a time
and one shard report after all selected cases terminate. Before an output may be
uploaded, aggregated, scored, or used to authorize another information stage,
validate the retained bytes with the independent custody verifier:

```bash
python scripts/science/verify_cut3r_source_comparison_artifacts.py shard \
  --output-root /path/to/source-comparison-output \
  --report /path/to/source-comparison-output/shards/smoke-case.json \
  --receipt /path/to/source-comparison-output/custody/smoke-case.json \
  --expected-plan-id 0dbb6b3a46e2c895259fd5f4a4691c1d6d3c43b0e71774171bbfb3a20239953c
```

The default shard command requires every referenced case to be an ordinary
success. `--allow-technical-failures` is available only for retaining a valid
negative custody record; such a receipt does not satisfy the execution barrier.

The verifier independently checks:

- the case and shard content identities;
- an exact, duplicate-free member roster;
- every member SHA-256 and byte count;
- regular-file and path-confinement semantics;
- absence of symbolic links and undeclared files;
- exact source/target and downstream-execution boundary flags;
- agreement between shard counts and retained case statuses; and
- absence of decoded source images from the publishable artifact.

A previously existing case is not accepted merely because its manifest names the
right plan and case. All bytes are rehashed. This closes the resume-integrity gap
in which a stale, truncated, or externally modified case directory could
otherwise be treated as complete.

## Execution order

1. Run exactly one frozen development smoke case.
2. Require an ordinary-success custody receipt with zero decoded frames retained.
3. Only then run the two frozen source shards.
4. Validate each shard before publication.
5. Aggregate or score only the case artifact IDs named by valid shard receipts.

A custody receipt establishes byte integrity and information-boundary compliance
only. It does not establish provider accuracy, source competence, covariance
calibration, BayesianPhysTwin value, Causal4D value, deployment safety, or state
of the art. A retained technical failure stops the source route unless the
already frozen protocol explicitly authorizes a replacement; the custody
verifier itself never grants such authorization.

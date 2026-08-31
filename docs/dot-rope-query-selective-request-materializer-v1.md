# DOT query-selective request materializer

This helper closes the operational gap between the frozen R04--R10 confirmation
and the already reviewed R11--R30 query-selective protocol. It does not open DOT
payloads or make a scientific decision.

After the R04--R10 workflow has completed, save the GitHub artifact metadata and
extract its `result.json` and `marker-support.json`. Then run:

```bash
python scripts/science/materialize_dot_rope_query_selective_request.py \
  --protocol-git-blob-sha "$(git rev-parse HEAD:protocols/dot-rope-query-selective-heldout-v1.json)" \
  --result /path/to/result.json \
  --marker-support /path/to/marker-support.json \
  --artifact-metadata /path/to/artifact-metadata.json
```

The helper independently invokes
`scripts/science/verify_dot_rope_cut3r_heldout_result.py`. It emits no request
unless the result is exactly `heldout-strong-positive`, the marker-support record
is valid, the artifact is unexpired and content-addressed, and all frozen
protocol identities agree. It refuses to overwrite an existing request.

Committing the generated file must remain a separate main-branch change. The
existing protected workflow requires that it be the only changed path before any
R11--R30 normal-view image or marker payload may be opened. R31--R70 remain
reserved.

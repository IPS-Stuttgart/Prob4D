# Exact retry invocation for the retained CUT3R source freeze

After this change is merged, dispatch
`Execute retained CUT3R source freeze automatically v2` from `main` with:

```text
execution_sha: 8b923e8cd67ca65f09312cffe305e36852f36fbb
request_id: 8f3c9fba12f8a16895edce89d7a92e4806a43cb2f34b5a05faff71945809b63e
```

The workflow independently proves that the revision is historical merged main,
that the retained request is byte-identical at the historical and current
revisions, and that the registered protocol blob is unchanged. These literals are
operational pointers to the cancelled zero-evidence request, not mutable defaults
or a new scientific registration.

Before dispatch, a matching `[self-hosted, Linux, X64, nvidia-smi]` runner should
be online and the four required repository variables should be configured. The
hosted preflight exposes variable names only. If no runner accepts the job within
20 minutes, the run is cancelled with a target-closed receipt and can be retried
with the same two literals.

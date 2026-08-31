# DOT R04–R10 hosted archive relay v1

This is an operational fallback for the already frozen DOT R04–R10 CUT3R confirmation. It is not a new scientific protocol and cannot be triggered while the direct `gpuserver6000` archive recovery is still active or successful.

## Why it exists

The frozen provider run `33434695566` stopped before image inference because the publisher archive connection ended about 1.9 MiB before the exact 1,408,905,061-byte payload was complete. It emitted no provider artifact and opened no marker payload. A direct resumable recovery is running separately as `33442397966`.

The hosted relay may be authorized only if that direct recovery terminates without success and the original provider run remains in its exact pre-inference failure state.

## Bounded route

1. A GitHub-hosted job resolves `R01-10.zip` through the official Dataverse metadata API.
2. It requires byte count `1408905061` and publisher MD5 `ca546ff5f22c0279123ccb18509858ee`.
3. It downloads the archive with bounded resumable retries, verifies it, and uploads checksum-bound chunks as a one-day transient artifact.
4. A read-only `gpuserver6000` job reconstructs and verifies the archive, then atomically installs it at the cache path already used by the frozen provider workflow.
5. A final GitHub-hosted job deletes the raw relay artifact and reruns only the exact failed provider job, allowing its existing dependent evaluator and independent verifier to continue.

## Information boundary

The relay does not decode images, inspect provider predictions, open 2-D or 3-D markers, change any scientific input, or create a new confirmation attempt. The self-hosted job has no repository or Actions write permission. The raw archive is not retained in the bounded receipt and is deleted from GitHub after installation.

No execution request is included with the control plane. A later main-branch commit may create only `protocols/execution_requests/dot_r04_r10_hosted_archive_relay_v1.json`, and only after the direct recovery has terminally failed.

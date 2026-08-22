# Retained CUT3R source-freeze lock publication

The source-freeze workflow runs on protected retained storage and uploads one
checksummed target-closed artifact. Claim-bearing source execution must consume
reviewed repository bytes rather than a mutable Actions download. This publication
stage converts the exact successful workflow payload into an ordinary lock pull
request without recomputing any scientific value.

## Request

`protocols/publication_requests/cut3r_deform360_source_freeze_v2.json` binds:

- source request ID
  `34eb4065a5be84ab6939595b38e53721d4555793a3d75eeb4cfc38170847b166`;
- the corresponding Actions artifact-name prefix;
- the required `source-support-freeze-ready` decision;
- ten source object-session groups;
- twelve forbidden confirmation groups; and
- the exact destination paths under `protocols/locks/`.

The request keeps source RGB decoding, source residual/truth access, target
payload access, and target outcome access false.

## Artifact verification

The hosted publication job lists nonexpired Actions artifacts and considers only
names bound to the exact source request. For each candidate, it:

1. downloads the ZIP through the GitHub Actions API;
2. rejects path traversal and symbolic links;
3. verifies every retained `SHA256SUMS` entry;
4. requires one exact execution summary, source freeze, comparison specification,
   comparison lock, and comparison summary;
5. verifies the source request, decision, group counts, and target-closed flags;
6. independently replays the comparison-lock validator and summary command; and
7. copies the original JSON bytes without normalization or rewriting.

A publication receipt binds the Actions artifact ID and source workflow run ID,
the source-freeze artifact ID, and the SHA-256 and byte count of every published
file.

## Review boundary

The workflow creates a normal pull request from a hosted runner. The retained
comparison workflow may start only after this exact-byte PR passes ordinary
repository review and is merged. The source-freeze publisher never runs on the
self-hosted retained-data runner and never receives protected filesystem paths.

## Scientific boundary

Publication establishes custody and replayability only. It does not execute
CUT3R, measure provider accuracy, fit uncertainty, open confirmation or target
payloads, run BayesianPhysTwin, run Causal4D, authorize deployment, or establish
state of the art.

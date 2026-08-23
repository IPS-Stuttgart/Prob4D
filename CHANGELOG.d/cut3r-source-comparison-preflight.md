Added an outcome-blind CUT3R source-comparison preflight that validates the
canonical request, source freeze, source-case locators, comparison specification,
and regenerated comparison lock. It verifies every frozen source video and
sidecar identity, confines metadata inventory to explicitly frozen source
episodes, checks the retained CUT3R origin/revision/worktree/checkpoint surface,
and publishes reports atomically without decoding frames, executing inference,
or opening source, confirmation, or target outcomes.

The preflight revalidates the source-freeze information boundary and exact
recurrent-online provider mode, verifies video identity before `ffprobe`, executes
only a uniquely resolved tracked `demo.py`, excludes raw remote URLs and absolute
retained paths, and stores command diagnostics only as redacted content digests.
The initial workflow is hosted-only and cannot enter the retained self-hosted
runner before the exact source locks are published.

Added an outcome-blind CUT3R source-comparison preflight that validates the
canonical request, source freeze, source-case locators, comparison specification,
and regenerated comparison lock. It verifies every frozen source video and
sidecar identity, confines metadata inventory to explicitly frozen source
episodes, checks the retained CUT3R origin/revision/worktree/checkpoint surface,
and publishes reports atomically without decoding frames, executing inference,
or opening source, confirmation, or target outcomes. The initial workflow is
hosted-only and cannot enter the retained self-hosted runner before the exact
source locks are published.

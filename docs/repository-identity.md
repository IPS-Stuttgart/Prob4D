# Repository identity and transfer compatibility

The canonical Prob4D repository is now `IPS-Stuttgart/Prob4D`. GitHub repository
names are mutable display coordinates: a repository may be transferred without
changing the project or its commit history. New integrations should therefore use
Prob4D's stable project identity rather than treating the current owner/name pair
as the sole identity.

Print the machine-readable descriptor with:

```bash
prob4d project identity --compact
```

The descriptor currently contains:

- stable project ID `github-repository-id:1295794737`;
- canonical repository `IPS-Stuttgart/Prob4D`;
- historical alias `FlorianPfaff/Prob4D`; and
- the repository string retained by frozen observation artifacts.

Repository aliases are matched case-insensitively but are otherwise exact.
Leading or trailing whitespace, string subclasses, and objects that merely
provide a string representation fail closed rather than being normalized.
Descriptor validation also requires exact JSON primitive types, so Boolean
schema versions and tuple-for-array substitutions cannot compare equal to the
canonical descriptor through Python coercion rules.

## Frozen artifact boundary

Provider-v1 artifacts, causal-stream-v1/v2 artifacts, conformance vectors, and
historical provider-v2 attestations may contain
`source_repository = "FlorianPfaff/Prob4D"`. That field is part of their frozen
content-addressed semantics. It must not be rewritten merely because the GitHub
repository was transferred.

The stable project-identity API is additive. It does not change any existing
artifact schema, provider manifest, artifact ID, calibration ID, or causal source
digest. Consumers can accept both current and historical repository aliases for
repository discovery while continuing to validate the exact legacy string when a
frozen schema requires it.

## Consumer migration rule

For new, non-frozen orchestration metadata:

1. bind `project_id` as the durable identity;
2. record the canonical repository for human navigation;
3. retain the exact repository string already embedded in frozen artifacts;
4. never rewrite old content-addressed artifacts to normalize a repository name;
5. update cross-repository fixtures before making a future schema require the
   stable project ID.

The Python helpers `canonical_prob4d_repository`, `is_prob4d_repository`, and
`validate_prob4d_project_identity` provide the same rules to programmatic
consumers.

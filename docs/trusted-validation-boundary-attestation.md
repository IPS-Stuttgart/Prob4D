# Trusted validation boundary attestation

The protected `Trusted exact-head validation` workflow controls which reviewed
pull-request revision may execute on `workstation2`. Repository code alone cannot
prove the external boundary around that workflow. In particular, a checked box,
Markdown statement, workflow file, or repository-local JSON document is not proof
of the current GitHub environment settings, runner-host state, dataset namespace,
or executed positive and negative controls.

`prob4d.trusted_validation_attestation` provides a strict, content-addressed
record for those four independently evidenced boundaries.

## Four separate evidence sections

A ready artifact requires all assertions in all four sections:

1. **GitHub environment policy** — independent review, no environment secrets,
   main-only workflow definition, self-approval handling, and read-only workflow
   permissions.
2. **Runner-host hardening** — a dedicated non-administrator account, absence of
   personal or write credentials, restricted unrelated resources and network
   destinations, and an incident rebuild or rotation procedure.
3. **Dataset namespace isolation** — only approved datasets exposed, read-only
   mounts where possible, unopened target cohorts outside the ordinary runner
   namespace, and an independently checked namespace inventory.
4. **Exact-head acceptance tests** — a successful exact-head run and observed
   environment pause, plus stale-SHA and non-`main` controls rejected before any
   self-hosted checkout. Retained evidence must bind the pull request, head, base,
   runner, and selected profile.

The sections are intentionally independent. A passing GitHub environment query
cannot substitute for a runner-host audit, and a successful exact-head run cannot
substitute for dataset isolation.

## Create an explicitly unverified draft

Bind the draft to the exact repository revision and workflow bytes under review:

```bash
python -m prob4d.trusted_validation_attestation template \
  --source-revision "$(git rev-parse HEAD)" \
  --workflow-sha256 "$(sha256sum \
    .github/workflows/trusted-self-hosted-validation.yml | cut -d' ' -f1)" \
  --output trusted-validation-attestation.draft.json
```

The generated draft is structurally valid but every section is `unverified`,
every assertion is `null`, and it cannot authorize trusted execution. Complete
one section only after an independent verifier has collected external evidence.
A completed section records:

- `verification_status`: `verified` or `failed`;
- the independent verifier and UTC verification time;
- an allowed external evidence method;
- an external HTTPS locator or `urn:prob4d:external-audit:...` locator;
- the SHA-256 digest of the retained external evidence snapshot; and
- every section-specific assertion as an explicit Boolean.

Repository-relative paths, GitHub repository blob/tree URLs, raw repository files,
and the evidence method `repository-file` are rejected. A `verified` section may
not contain a failed assertion. A completed audit with any failed assertion must
use status `failed` and keeps the overall boundary closed.

## Seal and verify

After completing the independently evidenced draft:

```bash
python -m prob4d.trusted_validation_attestation seal \
  --draft trusted-validation-attestation.draft.json \
  --output trusted-validation-attestation.json

python -m prob4d.trusted_validation_attestation verify \
  --artifact trusted-validation-attestation.json \
  --require-ready
```

Sealing recomputes the readiness decision, complete failure-reason set, and
content identity. The command returns status `2` when `--require-ready` is used
and any section remains unverified or failed. Loading a sealed artifact also
recomputes all derived fields, rejects duplicate JSON keys and non-finite values,
and detects content or identity tampering. Writers are atomic and no-clobber by
default.

## Operational use

A ready attestation should be retained beside the corresponding environment
settings snapshot, host audit, dataset-namespace inventory, and Actions-run
records. The workflow revision and source revision in the artifact must match the
revision whose trusted-execution boundary is being approved. A later environment,
host, dataset, workflow, or runner change requires a new attestation rather than
editing an old sealed artifact.

This artifact records an independently evidenced operational boundary. It does
not itself configure GitHub, sandbox the runner, grant protected target access,
establish provider accuracy or uncertainty calibration, establish
BayesianPhysTwin or Causal4D benefit, approve deployment, or constitute a
scientific result.

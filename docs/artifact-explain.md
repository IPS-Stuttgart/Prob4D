# Artifact explanation

`prob4d artifact explain` gives collaborators a compact, human-readable view of
one Prob4D artifact without weakening its validation boundary.

```bash
prob4d artifact explain outputs/sequence/observation_belief.npz
prob4d artifact explain outputs/sequence/observation_belief.npz --json
prob4d artifact explain outputs/sequence/observation_belief.npz --arrays
```

The command automatically recognizes these current artifact families:

- `ObservationBeliefV1` NPZ files;
- portable sparse gauge-tree prior manifests and their NPY sidecars;
- tree-sparse observation manifests, their NPY members, and their bound sparse
  gauge-tree prior;
- content-addressed prediction-bundle stores; and
- strict finite JSON objects that do not yet have a registered explanation
  handler.

Known artifacts are opened through their existing strict loader. The explanation
then reports the schema, content identity, causal/source context, and a compact
artifact-specific summary. It never substitutes a new validation path for the
normative loader.

## Strict versus structural output

A recognized artifact that passes its normative loader reports:

```text
Status: valid
Validation: strict-schema-and-content-address
```

An unrecognized JSON or NPZ container can still be inspected, but it reports:

```text
Status: structural-only
```

For structural-only JSON, duplicate keys and non-finite values are rejected, but
no artifact-specific schema or digest is claimed as verified. For structural-only
NPZ, the archive is opened with `allow_pickle=False`; no semantic or content-ID
claim is made.

Use `--require-strict` in automated or claim-bearing workflows:

```bash
prob4d artifact explain artifact.json --require-strict --json
```

The command exits with status 2 when no registered strict loader matches or when
the matched loader rejects the artifact.

## Array inventory

NPZ array values are not summarized by default. `--arrays` adds only member
names, dtypes, shapes, and uncompressed byte counts:

```bash
prob4d artifact explain observation_belief.npz --arrays
```

This option does not print observations, target outcomes, calibration residuals,
or other payload values.

## Scientific boundary

An explanation is an inspection aid. A `valid` explanation establishes only that
the existing strict loader accepted the artifact and its content identity. It does
not establish provider competence, calibrated uncertainty, BayesianPhysTwin
benefit, Causal4D intervention benefit, deployment safety, or state of the art.

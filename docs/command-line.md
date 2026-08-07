# Command-line interface

Prob4D exposes one grouped executable, `prob4d`. Historical `prob4d-*`
entry points remain installed for reproduction and scripted compatibility, but
every installed legacy command now has a canonical grouped replacement.

## Discover commands

```bash
prob4d --help
prob4d commands list
prob4d commands list --lifecycle stable
prob4d commands list --json
prob4d commands describe observation-export-calibrated
prob4d commands describe prob4d-target-admit --json
prob4d commands validate
```

The registry records, for every command:

- its stable command ID and grouped route;
- lifecycle (`stable`, `experimental`, `diagnostic`, or `archived`);
- implementation target and owning contract or subsystem;
- historical executable aliases and previous grouped routes;
- operational runtime requirements;
- whether it requires a GPU; and
- whether the command can produce or adjudicate claim-bearing evidence.

Registry output is deterministic JSON when `--json` is supplied. Automation
should use command IDs and registry metadata rather than scraping human help
text.

## Migrate historical commands

```bash
prob4d commands migrate prob4d-validate-observation
# prob4d observation validate

prob4d commands migrate prob4d-target-admit --json
```

Migration succeeds only for a historical executable alias or a previous grouped
route. A current command ID is not treated as a migration source.

## Newly grouped compatibility routes

The following tools previously existed only as standalone executables and are
now available under `prob4d`:

```text
prob4d diagnostic finite-sample-preflight
prob4d diagnostic provider-v2-gauge-ablation
prob4d diagnostic visual-bias-calibration
prob4d observation validate
prob4d provider target-admit
prob4d provider target-verify
```

The old executables remain unchanged. Frozen tags and environments can therefore
continue to reproduce their historical command lines while new workflows use a
single discoverable interface.

## Lifecycle meanings

**Stable** commands implement supported contracts and ordinary producer,
validation, storage, or evaluation workflows.

**Experimental** commands run current research protocols whose scientific result
must be interpreted through a separately frozen evidence contract.

**Diagnostic** commands run audits, ablations, stress tests, or exploratory
exports. A successful diagnostic run is not automatically promotable evidence.

**Archived** commands preserve a frozen compatibility or reproduction surface.
New claim-bearing workflows should use the stated stable replacement.

Lifecycle describes the command surface, not the scientific result. Artifact
provenance, target-access controls, calibration, technical-failure accounting,
and registered decision rules remain independently mandatory.

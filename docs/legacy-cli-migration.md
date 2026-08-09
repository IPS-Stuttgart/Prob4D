# Legacy executable migration

Prob4D has one canonical command surface:

```text
prob4d <grouped route>
```

Historical `prob4d-*` executables remain installed for compatibility, but they
are no longer independent dispatch definitions. Every alias is now routed
through `prob4d.legacy_cli`, which resolves its target and replacement from the
canonical command registry.

## Runtime behavior

An invocation such as

```bash
prob4d-validate-observation --help
```

continues to execute the unchanged validation command, but first writes a
migration message to standard error:

```text
DEPRECATION: 'prob4d-validate-observation' is a legacy compatibility executable; use 'prob4d observation validate'. Legacy executables are compatibility aliases and may be removed in the next documented incompatible pre-1.0 release.
```

The warning does not change the command arguments, standard output, or exit
status. A historical target returning no status is normalized to successful
status `0`; malformed non-integer target statuses fail closed.

## Machine-readable replacements

The existing registry commands remain the public migration interface:

```bash
prob4d commands list --json
prob4d commands describe prob4d-validate-observation --json
prob4d commands migrate prob4d-validate-observation
```

Python callers can also obtain the complete deterministic mapping:

```python
from prob4d.legacy_cli import (
    legacy_migration_descriptor,
    legacy_migration_table,
)

migrations = legacy_migration_table()
descriptor = legacy_migration_descriptor()
```

The descriptor contains a schema/version, removal policy, and every registered
legacy-to-grouped replacement. Its key set is checked against
`EXPECTED_LEGACY_ALIASES`, and package metadata is tested to ensure every legacy
entry point uses the central wrapper module.

## Compatibility policy

Legacy executables remain compatibility aliases during the current pre-1.0
line. New documentation, automation, and scientific protocols should use only
the grouped command. Removal requires a separately documented incompatible
release; aliases are not silently redirected to a different estimator or
protocol.

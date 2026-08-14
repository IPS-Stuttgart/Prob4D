# MotionCrafter stochastic seed policy

MotionCrafter can be run in deterministic or diffusion mode, but the same
adapter code creates several logically distinct prediction products:

- the disjoint-window baseline;
- the latent overlap baseline; and
- every independently decoded overlap window used by Prob4D.

Before this policy was introduced, every call received the same configured seed.
That behavior is kept as an explicit compatibility mode rather than silently
changing frozen runs.

## Policies

`prob4d motioncrafter` and `prob4d benchmark` accept:

```text
--seed-policy legacy-common
--seed-policy derived-per-call
```

### `legacy-common`

Every inference call receives the root `--seed` unchanged. This exactly
preserves historical behavior and may create common-random-number dependence
between products. Manifests created before the policy field existed are
interpreted as `legacy-common`.

### `derived-per-call`

Every call receives a deterministic 32-bit seed obtained from a versioned hash
of:

```text
(root seed, stable call identity)
```

The call identity binds the prediction product and, for an independently
decoded window, its window ID and exclusive source-frame interval. Changing the
window schedule therefore changes the effective seed schedule. Distinct seeds
avoid an implicit shared seed; they do not prove statistical independence of
predictions produced by a shared model and checkpoint.

## Manifest contract

New prediction manifests contain:

```json
{
  "config": {
    "seed": 42,
    "seed_policy": "derived-per-call"
  },
  "stochastic_seed_schedule": {
    "schema": "prob4d.motioncrafter-seed-schedule.v1",
    "policy": "derived-per-call",
    "root_seed": 42,
    "calls": [
      {
        "call_id": "baseline-disjoint",
        "product": "disjoint_baseline",
        "effective_seed": 1402652322
      },
      {
        "call_id": "baseline-latent-linear",
        "product": "latent_linear_baseline",
        "effective_seed": 4164824660
      },
      {
        "call_id": "overlap-window:window_0000:0:25",
        "product": "independently_decoded_overlap_window",
        "window_id": "window_0000",
        "source_frame_start": 0,
        "source_frame_stop_exclusive": 25,
        "effective_seed": 659168348
      }
    ]
  }
}
```

The example seeds above are the actual values produced by the version-1 schedule
for root seed 42 and the displayed call identities. The validator recomputes
every effective seed and rejects:

- unsupported policies or schedule schemas;
- a root seed or policy that differs from `config`;
- missing, reordered, duplicated, or source-inconsistent calls;
- an incorrect effective seed; and
- a derived-per-call seed collision.

A new `derived-per-call` manifest without a schedule fails closed. A legacy
manifest without a schedule remains admissible only as implicit
`legacy-common` evidence.

## Calibration compatibility

The historical common-seed behavior retains the version-1 MotionCrafter model
identifier, including when `legacy-common` is written explicitly. This keeps
frozen calibration artifacts reproducible.

`derived-per-call` uses `prob4d.motioncrafter-model.v2` and includes the policy
in the canonical model identifier. Gauge and point covariance calibrations
therefore cannot be reused silently across the two stochastic semantics.

## Recommended use

Use `legacy-common` only for exact reproduction of an existing run. For new
stochastic experiments, use `derived-per-call`, regenerate all covariance and
source-reliability calibration artifacts, and treat distinct root seeds as
separate sequence-level replicates. Do not count windows from one root-seed run
as independent Monte Carlo replicates.

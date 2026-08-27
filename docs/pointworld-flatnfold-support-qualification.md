# PointWorld--Flat'n'Fold support qualification

The proposed held-out PointWorld experiment must begin with an inventory-only
support decision. Prediction payloads, provider residuals, target geometry, and
BayesianPhysTwin innovations are prohibited at this stage.

The executable helper is:

```python
from prob4d.pointworld_flatnfold_support import (
    build_pointworld_flatnfold_support_request,
    scaffold_pointworld_flatnfold_support_inventory,
)
```

Create an intentionally incomplete inventory:

```bash
python -m prob4d.pointworld_flatnfold_support scaffold \
  outputs/pointworld-flatnfold/source-support-inventory.json
```

The scaffold contains one demonstration with the three required camera streams.
It is deliberately invalid until every `REPLACE_WITH_...` identity is replaced
and the complete source/inventory garment roster is entered.

After the exact PointWorld checkpoint, loader, dataset bytes, camera geometry,
action sequence, frame schedule, cohort binding, and promotion lock are frozen,
evaluate it with:

```bash
python -m prob4d.pointworld_flatnfold_support evaluate \
  outputs/pointworld-flatnfold/source-support-inventory.json \
  outputs/pointworld-flatnfold/provider-support-request.json \
  outputs/pointworld-flatnfold/provider-support-result.json
```

The command writes the existing replayable
`ProviderSupportFeasibilityRequestV1` and `ProviderSupportFeasibilityV1`
artifacts. Exit status `0` means support-feasible; status `2` is a completed
support-negative result. Invalid or tampered inputs raise instead of being
relabelled as a negative result.

## Inventory invariants

Version 1 enforces:

- exactly three declared camera IDs;
- every retained demonstration has all three camera streams;
- all cameras of a demonstration bind the same action-sequence digest;
- all cameras of a demonstration use the same causal frame schedule;
- each stream binds intrinsic, extrinsic, and metric-anchor digests;
- complete garment identity is the top-level support group;
- stream IDs are demonstration/camera pairs nested under a garment;
- prediction payloads, residuals, and target outcomes are all unopened; and
- technical exclusions are only those frozen in the inventory.

Frames, cameras, and demonstrations are nested support observations. They are
not promoted to independent target replicates. The later provider and
BayesianPhysTwin analysis must cluster at complete physical garment identity.

## Decision order

1. Freeze exact source revisions and byte identities.
2. Freeze the complete inventory/source garment roster.
3. Record camera/action/frame support without reading PointWorld predictions.
4. Build and retain the support-feasibility result.
5. Stop the provider version if support-negative.
6. Only after a positive result, run source-only PointWorld exports through the
   sparse persistent-point contract.
7. Calibrate point uncertainty, dependence, reliability, and downstream
   admission on source/calibration garments only.
8. Freeze a held-out target protocol before target outcomes are opened.

A passing inventory proves only that the declared experiment is technically
supportable. It does not establish PointWorld accuracy, useful Prob4D fusion,
calibrated uncertainty, BayesianPhysTwin benefit, Causal4D benefit, or safety.

# PointWorld checkpoint-only model-load smoke

This source-side smoke removes the PointWorld runtime as a confound before any
Flat'n'Fold robot demonstration is opened. It is intentionally dataset-free.

The run binds and verifies:

- Prob4D execution revision and request identity;
- PointWorld revision `05484826dfef74cbe278a3974179a5a16705d35d`;
- DINOv3 submodule revision `54694f7627fd815f62a5dcc82944ffa6153bbb76`;
- the exact 1,213,050,671-byte DINOv3 ViT-L/16 file with SHA-256
  `8aa4cbddda325040fc78db2c272754af6ebe8ff2c55f6ec4f1964d8890f66035`;
- the selected `large-droid+behavior/model-best.pt` bytes, whose SHA-256 is
  measured and recorded before model loading; and
- Python, PyTorch, CUDA, GPU, parameter-count, and peak-memory information.

The trusted self-hosted job copies the pinned PointWorld checkout into an
isolated runner-temporary workspace and links the read-only external DINOv3
weights into that copy. It creates an isolated Python 3.10 environment from the
pinned PointWorld requirements, loads the complete PointWorld checkpoint and
DINOv3 backbone on CUDA, and then stops. No data loader and no model forward pass
are constructed.

## Trigger

The workflow is triggered only by a non-forced push to `main` that changes:

```text
protocols/execution_requests/pointworld_model_load_smoke_v1.json
```

The request must bind the Git blob of
`protocols/pointworld-model-load-smoke-v1.json` and carry a canonical request ID.
Dataset access, prediction execution, provider residuals, and target outcomes
must all remain explicitly unauthorized.

## Staged paths on gpuserver6000

The frozen runner alias is:

```text
/home/github-runner/.cache/datasets/pointworld-flatnfold-v1
```

Expected relative entries are:

```text
code/PointWorld
models/pointworld/large-droid+behavior/model-best.pt
models/dinov3/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth
```

A missing path, revision mismatch, changed DINOv3 byte count/hash, unavailable
CUDA runtime, dependency-installation failure, or model-load failure produces a
sanitized negative artifact and a failed execution job. A hosted reporting job
posts the result to issue #333; the self-hosted job receives no issue-write or
repository-write permission.

## Claim boundary

A passing result means only that these exact model assets can be initialized on
gpuserver6000. It is not PointWorld validation on Flat'n'Fold and carries no
Prob4D, BayesianPhysTwin, Causal4D, accuracy, calibration, or safety claim.

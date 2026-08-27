# Real source smoke checklist

Complete this checklist for the first predeclared source garment before any
calibration or target garment is opened.

## Byte and environment identity

- [ ] Prob4D revision recorded.
- [ ] PointWorld revision recorded.
- [ ] PointWorld checkpoint SHA-256 recorded.
- [ ] DINOv3 and all auxiliary model SHA-256 values recorded.
- [ ] Loader/configuration content ID recorded.
- [ ] Python, PyTorch, CUDA, driver, GPU, and deterministic settings recorded.
- [ ] Flat'n'Fold archive-member SHA-256 manifest recorded.

## Geometry and action

- [ ] All three RGB-D camera streams present for the complete demonstration.
- [ ] Camera intrinsics IDs recorded.
- [ ] Camera-to-Baxter-base extrinsics IDs recorded.
- [ ] Depth units and metric scale verified without target residuals.
- [ ] Baxter/gripper pose convention verified.
- [ ] Robot point-flow construction uses only released geometry and actions.
- [ ] Action-sequence content ID recorded and identical across camera streams.

## PointWorld sparse export

- [ ] One context frame and the declared forecast horizon are fixed.
- [ ] Absolute output frame IDs are fixed.
- [ ] Context scene-point order is retained.
- [ ] Context displacement is exact zero.
- [ ] Context validity equals source support.
- [ ] Inactive context points do not become valid later.
- [ ] Positions are constructed only from context coordinates plus declared
      displacement-from-context.
- [ ] Native log variance is exported unchanged.
- [ ] Context uncertainty is marked invalid.
- [ ] No target truth, rasterization, or nearest-neighbor repair is used.
- [ ] Canonical sparse artifact verifies and its content ID is recorded.

## Decision

- [ ] `representation_positive`: proceed to source mean/identity diagnostics.
- [ ] `representation_negative`: stop this provider version and retain the exact
      failure artifact and logs.

Passing this checklist is not provider competence or paper evidence by itself.

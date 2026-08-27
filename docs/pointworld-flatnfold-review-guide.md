# Review guide for the PointWorld--Flat'n'Fold branch

Review this branch in three independent layers.

## 1. Sparse artifact contract

Check that the contract cannot silently:

- duplicate or mutate point IDs;
- reinterpret invalid points;
- change dense precision;
- pair uncertainty with the wrong validity mask;
- treat absent uncertainty as present;
- accept unknown archive fields; or
- overwrite an existing evidence artifact.

## 2. PointWorld adapter

Check that the adapter:

- retains original active context-array indices;
- requires exact zero context displacement;
- forbids inactive-point resurrection;
- reconstructs positions only by addition to context coordinates;
- marks context uncertainty invalid;
- preserves native log variance unchanged;
- binds repository, revision, checkpoint, loader, camera, action, and source bytes;
- uses no rasterization or target truth; and
- verifies the written canonical artifact.

## 3. Flat'n'Fold support gate

Check that the inventory:

- is evaluated before prediction/residual/target access;
- requires exactly three cameras;
- retains all cameras for each demonstration;
- uses one action digest and frame schedule per demonstration;
- groups all streams by complete garment identity;
- binds geometry and metric-anchor IDs; and
- retains valid support-negative outcomes.

A review approval confirms software and protocol boundaries only. It does not
approve a provider, target experiment, paper claim, or deployment.

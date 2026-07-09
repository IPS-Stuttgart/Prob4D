# Prob4D

Probabilistic recursive fusion for long-horizon 4D reconstruction from
overlapping MotionCrafter windows.

The core estimator treats each window's global gauge as an uncertain
`Sim(3)` state. It aligns duplicate predictions for the same frame and pixel,
calibrates a compact along-ray/lateral uncertainty model, and fuses correlated
window estimates conservatively.

The initial experiment is organized around seven variants:

1. disjoint 25-frame MotionCrafter windows;
2. MotionCrafter's latent-space linear overlap blend;
3. decoded-space `Sim(3)` alignment and uniform averaging;
4. naive precision-weighted fusion;
5. covariance intersection;
6. covariance intersection with fixed-lag gauge smoothing; and
7. the smoothed estimator with sparse metric anchors.

The package consumes window files containing MotionCrafter's `point_map`,
`valid_mask`, `scene_flow`, and `deform_mask` arrays. Each file additionally
stores its absolute `frame_indices`, which prevents accidental fusion of local
frame numbers from different clips.

## Development

```bash
python -m pip install -e ".[dev]"
pytest
```

Large videos, model weights, predictions, and result artifacts are kept out of
git. Reproducible experiment commands will write those files below `outputs/`.


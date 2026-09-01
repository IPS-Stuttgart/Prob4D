# DOT R04–R10 shared-3-D multiview support audit

**Decision:** `multiview-support-available`

The original fixed-`cam001` DOT experiment remains a terminal support-negative result. This diagnostic asks a narrower question on the already-open R04–R10 cohort: whether a camera-union observation rule would satisfy the frozen marker-count support requirements before any new R11–R70 protocol is registered.

## Dataset layout correction

A ZIP-central-directory census showed that DOT V29 does not contain one 3-D coordinate file per camera. For every audited sequence and frame it contains:

- camera-specific 2-D coordinate carriers;
- normal-view images for multiple cameras; and
- exactly one shared 3-D coordinate carrier labelled `cam001`.

Across the five cameras present in every R04–R10 sequence (`cam001`–`cam005`), all **245** camera-frame pairs have equal 2-D and shared-3-D row counts. The corrected audit therefore uses row-index identity against the shared 3-D carrier and rejects any future row-count mismatch instead of silently truncating.

## Structural support result

No individual camera is feasible across every sequence. Under the union of the five common views, however, the ordinary pixel-zero-based interpretation passes every registered support requirement on all seven sequences:

| Sequence | Fit A | Overlap | Nonempty overlap frames | Fit B |
| --- | ---: | ---: | ---: | ---: |
| R04 | 10 | 15 | 3 | 10 |
| R05 | 12 | 18 | 3 | 13 |
| R06 | 14 | 21 | 3 | 14 |
| R07 | 20 | 30 | 3 | 20 |
| R08 | 14 | 21 | 3 | 14 |
| R09 | 18 | 26 | 3 | 13 |
| R10 | 10 | 15 | 3 | 10 |

The frozen minima were 6 markers in each fit partition, 6 overlap observations, and two nonempty overlap frames. The five-view union therefore has nontrivial margin above the support boundary.

## Consequence

A separately versioned **multiview** source-gated DOT protocol is structurally admissible for registration. This does not reopen, replace, or reinterpret the fixed-camera negative result. It also does not authorize opening R11–R70: the new observation rule, coordinate convention, provider, covariance handling, thresholds, and target custody must first be frozen.

## Provenance

- layout census run: `33512929079`
- layout census artifact: `9802375802`
- census ID: `74f090a99d6740ac3388c43493531ea5168291e4c5a709fb74344e45b46b4f19`
- corrected support run: `33513389851`
- corrected support job: `99874305523`
- corrected support artifact: `9802559019`
- corrected artifact SHA-256: `e31e07ececc6cbe4897eff59f14f43a8c0b1a3c5a6d51cca8daf1c6c8b8a20fd`
- adapted audit ID: `d66d20e2ec156d098d0c79324a32504829c48a7c616da6571be54b31ddf46db7`
- machine-readable record: [`result.json`](result.json)

## Boundary

No R11–R70 archive was opened. No provider prediction, covariance score, marker error, or other performance metric was computed. The result establishes observation support only—not provider competence, decision value, calibration, safety, or state of the art.

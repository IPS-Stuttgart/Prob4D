"""Convert official CUT3R recurrent-online outputs into neutral predictions.

The adapter consumes the deterministic file layout written by CUT3R's recurrent
``demo.py`` path. It never imports or executes CUT3R, Torch, OpenCV, or model
checkpoints. Exact code, checkpoint, input-video, and generated-source identities
are bound into the resulting provider manifest.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from pathlib import Path
from typing import Final

import numpy as np

from ._cut3r_limits import Cut3RImportLimits, DEFAULT_CUT3R_IMPORT_LIMITS
from ._cut3r_source import (
    _adapter_implementation_sha256,
    _file_sha256,
    _record_id,
    _verify_source_descriptors,
)
from ._cut3r_window import _canonical_window, _relative_member, _write_window_atomically
from ._strict_json import require_exact_integer, require_revision, require_sha256
from .data import DENSE_STORAGE_DTYPES, DenseStorageDType
from .prediction_provider_manifest import (
    PredictionFrameLineageV1,
    PredictionPayloadDescriptorV1,
    PredictionProviderManifestV1,
    save_prediction_provider_manifest,
    verify_prediction_provider_manifest,
)

CUT3R_OFFICIAL_REPOSITORY: Final = "CUT3R/CUT3R"
CUT3R_ONLINE_SOURCE_LAYOUT: Final = "cut3r-demo-recurrent-depth-conf-camera-v1"
_ADAPTER_DOMAIN: Final = "prob4d.cut3r-online-provider-adapter.v1"
_MODEL_SET_DOMAIN: Final = "prob4d.cut3r-online-model-set.v1"
_SOURCE_BUNDLE_DOMAIN: Final = "prob4d.cut3r-online-source-bundle.v1"
_RUN_DOMAIN: Final = "prob4d.cut3r-online-run.v1"
_MEMBER_DOMAIN: Final = "prob4d.cut3r-online-member.v1"


def import_cut3r_online_prediction_manifest(
    source_root: str | Path,
    output_manifest_path: str | Path,
    *,
    sequence_id: str,
    cut3r_revision: str,
    checkpoint_sha256: str,
    input_video_sha256: str,
    input_video_byte_count: int,
    frame_start: int = 0,
    view_id: str = "camera-0",
    window_id: str = "cut3r-online",
    confidence_threshold: float = 1.5,
    storage_dtype: DenseStorageDType = "float32",
    limits: Cut3RImportLimits | None = None,
) -> PredictionProviderManifestV1:
    """Import one exact recurrent-online CUT3R output tree.

    The caller declares that the source tree was generated with CUT3R's recurrent
    online path, one forward pass, no revisit, and no global alignment. The
    adapter binds that declaration and per-frame prefix lineage but cannot prove
    how an external process was launched.
    """

    revision = require_revision(cut3r_revision, name="CUT3R revision")
    checkpoint = require_sha256(checkpoint_sha256, name="CUT3R checkpoint SHA-256")
    video_sha256 = require_sha256(input_video_sha256, name="input video SHA-256")
    video_bytes = require_exact_integer(
        input_video_byte_count,
        name="input video byte count",
        minimum=1,
    )
    start = require_exact_integer(frame_start, name="frame_start", minimum=0)
    if storage_dtype not in DENSE_STORAGE_DTYPES:
        raise ValueError("storage_dtype must be one of " + ", ".join(DENSE_STORAGE_DTYPES))
    if type(sequence_id) is not str or not sequence_id:
        raise ValueError("sequence_id must be a nonempty string")
    if type(view_id) is not str or not view_id:
        raise ValueError("view_id must be a nonempty string")
    if type(window_id) is not str or not window_id:
        raise ValueError("window_id must be a nonempty string")
    if type(confidence_threshold) not in {int, float}:
        raise TypeError("confidence_threshold must be an int or float")
    threshold = float(confidence_threshold)
    if not np.isfinite(threshold) or threshold < 0.0:
        raise ValueError("confidence_threshold must be finite and non-negative")
    effective_limits = DEFAULT_CUT3R_IMPORT_LIMITS if limits is None else limits
    if not isinstance(effective_limits, Cut3RImportLimits):
        raise TypeError("limits must be a Cut3RImportLimits instance")

    source = Path(source_root)
    output_path = Path(output_manifest_path)
    manifest_root = output_path.parent.resolve()
    sequence_token = hashlib.sha256(sequence_id.encode("utf-8")).hexdigest()[:16]
    payload_path = manifest_root / "payloads" / f"{sequence_token}-cut3r-online.npz"

    with _canonical_window(
        source,
        frame_start=start,
        window_id=window_id,
        confidence_threshold=threshold,
        storage_dtype=storage_dtype,
        limits=effective_limits,
    ) as (window, source_members, source_member_bytes, dense_array_bytes):
        _verify_source_descriptors(source, source_members)

        loader_id = _adapter_implementation_sha256()
        model_set_id = _record_id(
            _MODEL_SET_DOMAIN,
            {
                "official_repository": CUT3R_OFFICIAL_REPOSITORY,
                "cut3r_revision": revision,
                "checkpoint_sha256": checkpoint,
            },
        )
        source_bundle_id = _record_id(
            _SOURCE_BUNDLE_DOMAIN,
            {
                "layout": CUT3R_ONLINE_SOURCE_LAYOUT,
                "members": list(source_members),
            },
        )
        provider_run_id = _record_id(
            _RUN_DOMAIN,
            {
                "model_set_id": model_set_id,
                "loader_id": loader_id,
                "source_bundle_id": source_bundle_id,
                "input_video_sha256": video_sha256,
                "input_video_byte_count": video_bytes,
                "frame_start": start,
                "frame_count": int(window.shape[0]),
                "confidence_threshold": threshold,
                "execution_mode": "recurrent-online",
                "revisit_count": 1,
                "global_alignment": False,
            },
        )
        stochastic_member_id = _record_id(
            _MEMBER_DOMAIN,
            {
                "model_set_id": model_set_id,
                "provider_run_id": provider_run_id,
            },
        )

        _write_window_atomically(payload_path, window)
        payload = PredictionPayloadDescriptorV1(
            product_role="external-sequence",
            window_id=window.window_id,
            path=_relative_member(payload_path, root=manifest_root),
            sha256=_file_sha256(payload_path),
            byte_count=int(payload_path.stat().st_size),
            view_id=view_id,
            stochastic_member_id=stochastic_member_id,
            dependence_group_ids=(
                f"input-video:{video_sha256}",
                f"model-set:{model_set_id}",
                f"provider-run:{provider_run_id}",
            ),
            dense_storage_dtype=storage_dtype,
            has_scene_flow=False,
            has_ray_directions=False,
            frame_lineage=tuple(
                PredictionFrameLineageV1(
                    output_frame_id=int(frame_id),
                    source_frame_start=start,
                    source_frame_stop_exclusive=int(frame_id) + 1,
                    contributor_ids=(provider_run_id,),
                )
                for frame_id in window.frame_indices
            ),
        )
        manifest = PredictionProviderManifestV1(
            sequence_id=sequence_id,
            provider_family="CUT3R-online",
            provider_repository=CUT3R_OFFICIAL_REPOSITORY,
            provider_revision=revision,
            provider_run_id=provider_run_id,
            model_set_id=model_set_id,
            loader_id=loader_id,
            coordinate_semantics="sequence-local-sim3",
            point_semantics="dense-point-map",
            flow_semantics="absent",
            ray_semantics="absent",
            payloads=(payload,),
            metadata={
                "source_adapter": _ADAPTER_DOMAIN,
                "source_adapter_sha256": loader_id,
                "source_layout": CUT3R_ONLINE_SOURCE_LAYOUT,
                "source_bundle_id": source_bundle_id,
                "source_member_count": len(source_members),
                "source_member_total_bytes": source_member_bytes,
                "dense_array_byte_count": dense_array_bytes,
                "canonicalization_backend": "frame-streamed-npy-memmap-v1",
                "sequence_wide_dense_stack_avoided": True,
                "input_video_sha256": video_sha256,
                "input_video_byte_count": video_bytes,
                "checkpoint_sha256": checkpoint,
                "execution_mode": "recurrent-online",
                "online_prefix_only": True,
                "revisit_count": 1,
                "global_alignment": False,
                "confidence_threshold": threshold,
                "confidence_is_support_not_reliability": True,
                "metric_scale_claimed": False,
                "uses_truth": False,
                "uses_downstream_physical_innovation": False,
            },
        )
        save_prediction_provider_manifest(output_path, manifest)
        verify_prediction_provider_manifest(output_path)
        return manifest


def main(argv: Sequence[str] | None = None) -> int:
    """Dispatch the grouped CUT3R import command without eager CLI imports."""

    from ._cut3r_cli import main as cli_main

    return cli_main(argv)


__all__ = [
    "CUT3R_OFFICIAL_REPOSITORY",
    "CUT3R_ONLINE_SOURCE_LAYOUT",
    "Cut3RImportLimits",
    "DEFAULT_CUT3R_IMPORT_LIMITS",
    "import_cut3r_online_prediction_manifest",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())

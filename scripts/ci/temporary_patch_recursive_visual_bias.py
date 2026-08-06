#!/usr/bin/env python3
"""Apply the reviewed one-shot recursive visual-bias repair."""

from __future__ import annotations

import re
from pathlib import Path

SOURCE_PATH = Path("src/prob4d/visual_bias_stream.py")
TEST_PATH = Path("tests/test_visual_bias_stream.py")


def _replace_once(source: str, old: str, new: str, *, label: str) -> str:
    if new in source:
        return source
    if old not in source:
        raise RuntimeError(f"expected source fragment is missing for {label}")
    return source.replace(old, new, 1)


def _patch_source() -> None:
    source = SOURCE_PATH.read_text(encoding="utf-8")
    source = _replace_once(
        source,
        "from typing import Any, Final, TypeAlias\n",
        "from typing import Any, Final, TypeAlias, cast\n",
        label="cast import",
    )
    source = _replace_once(
        source,
        "np.savez_compressed(stream, **dict(arrays))\n",
        "np.savez_compressed(stream, **dict(arrays))  # type: ignore[arg-type]\n",
        label="NPZ typing boundary",
    )
    source = _replace_once(
        source,
        ") -> np.ndarray:\n    covariance = np.asarray(value)\n",
        ") -> FloatArray:\n    covariance = np.asarray(value)\n",
        label="joint covariance return type",
    )
    source = _replace_once(
        source,
        "    return covariance\n\n\ndef _bias_model_record(",
        "    return cast(FloatArray, covariance)\n\n\ndef _bias_model_record(",
        label="joint covariance return cast",
    )

    method_start = source.index("    @classmethod\n    def from_record(")
    method_end = source.index(
        "\n\n\n@dataclass(frozen=True)\nclass VisualBiasNuisanceStreamV1:",
        method_start,
    )
    typed_method = '''    @classmethod
    def from_record(cls, value: object) -> VisualBiasStreamUpdateV1:
        mapping = require_mapping(value, name="visual-bias stream update")
        require_exact_fields(mapping, _UPDATE_FIELDS, name="visual-bias stream update")
        if mapping["schema"] != VISUAL_BIAS_STREAM_UPDATE_SCHEMA:
            raise ValueError("unsupported visual-bias stream update schema")
        previous_value = mapping["previous_update_id"]
        previous_update_id = (
            None
            if previous_value is None
            else require_sha256(previous_value, name="previous_update_id")
        )
        return cls(
            bias_model_id=require_sha256(mapping["bias_model_id"], name="bias_model_id"),
            observation_stream_update_id=require_sha256(
                mapping["observation_stream_update_id"],
                name="observation_stream_update_id",
            ),
            visual_bias_artifact_id=require_sha256(
                mapping["visual_bias_artifact_id"],
                name="visual_bias_artifact_id",
            ),
            observation_artifact_id=require_sha256(
                mapping["observation_artifact_id"],
                name="observation_artifact_id",
            ),
            observation_identity_sha256=require_sha256(
                mapping["observation_identity_sha256"],
                name="observation_identity_sha256",
            ),
            frame_start=require_exact_integer(
                mapping["frame_start"],
                name="frame_start",
                minimum=0,
            ),
            frame_stop_exclusive=require_exact_integer(
                mapping["frame_stop_exclusive"],
                name="frame_stop_exclusive",
                minimum=1,
            ),
            row_start=require_exact_integer(
                mapping["row_start"],
                name="row_start",
                minimum=0,
            ),
            row_stop_exclusive=require_exact_integer(
                mapping["row_stop_exclusive"],
                name="row_stop_exclusive",
                minimum=1,
            ),
            maximum_gauge_projection=_finite_nonnegative_real(
                mapping["maximum_gauge_projection"],
                name="maximum_gauge_projection",
            ),
            previous_update_id=previous_update_id,
            update_id=require_sha256(mapping["update_id"], name="update_id"),
        )'''
    source = source[:method_start] + typed_method + source[method_end:]

    source = _replace_once(
        source,
        '        bias_ids = require_string_sequence(self.bias_ids, name="bias_ids")\n'
        '        basis_names = require_string_sequence(self.basis_names, name="basis_names")\n',
        '        validated_bias_ids: tuple[str, ...] = require_string_sequence(\n'
        '            self.bias_ids,\n'
        '            name="bias_ids",\n'
        '        )\n'
        '        validated_basis_names: tuple[str, ...] = require_string_sequence(\n'
        '            self.basis_names,\n'
        '            name="basis_names",\n'
        '        )\n',
        label="validated bias identifiers",
    )
    replacements = (
        (
            "len(set(bias_ids)) != len(bias_ids)",
            "len(set(validated_bias_ids)) != len(validated_bias_ids)",
            "bias ID uniqueness",
        ),
        (
            "len(set(basis_names)) != len(basis_names)",
            "len(set(validated_basis_names)) != len(validated_basis_names)",
            "basis-name uniqueness",
        ),
        (
            "row_bias >= len(bias_ids)",
            "row_bias >= len(validated_bias_ids)",
            "row bias bounds",
        ),
        (
            "expected_jacobian = (row_count, 3, len(basis_names))",
            "expected_jacobian = (row_count, 3, len(validated_basis_names))",
            "Jacobian shape",
        ),
        (
            "latent_dimension = len(bias_ids) * len(basis_names)",
            "latent_dimension = len(validated_bias_ids) * len(validated_basis_names)",
            "latent dimension",
        ),
        (
            "            bias_ids=bias_ids,\n            basis_names=basis_names,\n",
            "            bias_ids=validated_bias_ids,\n"
            "            basis_names=validated_basis_names,\n",
            "model-record names",
        ),
        (
            'object.__setattr__(self, "bias_ids", bias_ids)',
            'object.__setattr__(self, "bias_ids", validated_bias_ids)',
            "stored bias IDs",
        ),
        (
            'object.__setattr__(self, "basis_names", basis_names)',
            'object.__setattr__(self, "basis_names", validated_basis_names)',
            "stored basis names",
        ),
    )
    for old, new, label in replacements:
        source = _replace_once(source, old, new, label=label)

    source = _replace_once(
        source,
        "        model_metadata = frozen_finite_json_mapping(\n",
        "        validated_model_metadata: Mapping[str, Any] = "
        "frozen_finite_json_mapping(\n",
        label="model metadata type",
    )
    source = _replace_once(
        source,
        "        metadata = frozen_finite_json_mapping(\n",
        "        validated_metadata: Mapping[str, Any] = frozen_finite_json_mapping(\n",
        label="stream metadata type",
    )
    source = _replace_once(
        source,
        "            model_metadata=model_metadata,\n",
        "            model_metadata=validated_model_metadata,\n",
        label="model metadata use",
    )
    source = _replace_once(
        source,
        'object.__setattr__(self, "model_metadata", model_metadata)',
        'object.__setattr__(self, "model_metadata", validated_model_metadata)',
        label="stored model metadata",
    )
    source = _replace_once(
        source,
        'object.__setattr__(self, "metadata", metadata)',
        'object.__setattr__(self, "metadata", validated_metadata)',
        label="stored stream metadata",
    )

    source = _replace_once(
        source,
        '        for attribute in (\n'
        '            "observation_stream_update_id",\n'
        '            "visual_bias_artifact_id",\n'
        '        ):\n',
        '        for attribute in (\n'
        '            "observation_stream_update_id",\n'
        '            "visual_bias_artifact_id",\n'
        '            "observation_artifact_id",\n'
        '            "observation_identity_sha256",\n'
        '        ):\n',
        label="replayed-evidence rejection",
    )

    expected_pattern = re.compile(
        r"            expected(?:: IntArray)? = np\.full\(\n"
        r"(?P<body>.*?)"
        r"            \)\n"
        r"            if not np\.array_equal\(\n",
        re.DOTALL,
    )
    if "expected = cast(IntArray, np.full(" not in source:
        match = expected_pattern.search(source)
        if match is None:
            raise RuntimeError("expected row-update construction is missing")
        replacement = (
            "            expected = cast(IntArray, np.full(\n"
            + match.group("body")
            + "            ))\n"
            + "            if not np.array_equal(\n"
        )
        source = source[: match.start()] + replacement + source[match.end() :]

    source = _replace_once(
        source,
        "    row_updates: list[np.ndarray] = []\n"
        "    row_bias: list[np.ndarray] = []\n"
        "    jacobians: list[np.ndarray] = []\n",
        "    row_updates: list[IntArray] = []\n"
        "    row_bias: list[IntArray] = []\n"
        "    jacobians: list[FloatArray] = []\n",
        label="typed row buffers",
    )
    source = _replace_once(
        source,
        "        row_updates.append(\n"
        "            np.full(nuisance.observation_count, index, dtype=np.int64)\n"
        "        )\n"
        "        row_bias.append(np.asarray(nuisance.row_bias_indices))\n"
        "        jacobians.append(np.asarray(nuisance.bias_jacobian))\n"
        "        row_start = row_stop\n"
        "        previous_update_id = update.update_id\n",
        "        row_updates.append(\n"
        "            cast(\n"
        "                IntArray,\n"
        "                np.full(nuisance.observation_count, index, dtype=np.int64),\n"
        "            )\n"
        "        )\n"
        "        row_bias.append(\n"
        "            cast(IntArray, np.asarray(nuisance.row_bias_indices, dtype=np.int64))\n"
        "        )\n"
        "        jacobians.append(\n"
        "            cast(FloatArray, np.asarray(nuisance.bias_jacobian, dtype=np.float64))\n"
        "        )\n"
        "        row_start = row_stop\n"
        "        if update.update_id is None:\n"
        '            raise AssertionError("validated visual-bias update lacks an ID")\n'
        "        previous_update_id = update.update_id\n",
        label="typed recursive rows",
    )

    source = _replace_once(
        source,
        "    stream = VisualBiasNuisanceStreamV1(\n"
        '        stream_key=record["stream_key"],\n',
        '    stream_key = require_exact_string(record["stream_key"], name="stream_key")\n'
        "    orthogonalization_semantics = require_exact_string(\n"
        '        record["orthogonalization_semantics"],\n'
        '        name="orthogonalization_semantics",\n'
        "    )\n"
        "    gauge_projection_tolerance = _positive_real(\n"
        '        record["gauge_projection_tolerance"],\n'
        '        name="gauge_projection_tolerance",\n'
        "    )\n"
        '    bias_model_id = require_sha256(record["bias_model_id"], name="bias_model_id")\n'
        '    artifact_id = require_sha256(record["artifact_id"], name="artifact_id")\n'
        "    stream = VisualBiasNuisanceStreamV1(\n"
        "        stream_key=stream_key,\n",
        label="typed manifest values",
    )
    source = _replace_once(
        source,
        '        orthogonalization_semantics=record["orthogonalization_semantics"],\n'
        '        gauge_projection_tolerance=record["gauge_projection_tolerance"],\n',
        "        orthogonalization_semantics=orthogonalization_semantics,\n"
        "        gauge_projection_tolerance=gauge_projection_tolerance,\n",
        label="typed manifest semantics",
    )
    source = _replace_once(
        source,
        '        bias_model_id=record["bias_model_id"],\n'
        '        artifact_id=record["artifact_id"],\n',
        "        bias_model_id=bias_model_id,\n"
        "        artifact_id=artifact_id,\n",
        label="typed manifest IDs",
    )
    SOURCE_PATH.write_text(source, encoding="utf-8")


def _patch_tests() -> None:
    tests = TEST_PATH.read_text(encoding="utf-8")
    if "def test_builder_rejects_replayed_observation_evidence()" in tests:
        return
    marker = "\ndef test_append_preserves_retained_update_chain() -> None:\n"
    regression = '''

def test_builder_rejects_replayed_observation_evidence() -> None:
    first = _nuisance(observation_character="a", identity_character="b")
    repeated_artifact = _nuisance(
        observation_character="a",
        identity_character="d",
    )
    with pytest.raises(ValueError, match="observation_artifact_id values must be unique"):
        build_visual_bias_nuisance_stream(
            stream_key="replayed-artifact",
            nuisances=(first, repeated_artifact),
            observation_stream_update_ids=(_sha("1"), _sha("2")),
            frame_intervals=((0, 5), (5, 10)),
        )

    repeated_identity = _nuisance(
        observation_character="c",
        identity_character="b",
    )
    with pytest.raises(
        ValueError,
        match="observation_identity_sha256 values must be unique",
    ):
        build_visual_bias_nuisance_stream(
            stream_key="replayed-identity",
            nuisances=(first, repeated_identity),
            observation_stream_update_ids=(_sha("1"), _sha("2")),
            frame_intervals=((0, 5), (5, 10)),
        )
'''
    if marker not in tests:
        raise RuntimeError("test insertion marker is missing")
    TEST_PATH.write_text(tests.replace(marker, regression + marker, 1), encoding="utf-8")


def main() -> int:
    _patch_source()
    _patch_tests()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

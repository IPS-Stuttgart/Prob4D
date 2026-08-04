from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any, cast

import pytest

from prob4d._immutable_json import (
    FrozenJsonDict,
    FrozenJsonList,
    frozen_finite_json_mapping,
    plain_json,
)


def test_plain_json_copies_nested_containers_and_converts_tuples() -> None:
    original = {
        "nested": {
            "values": (1, 2),
        },
        "flag": True,
    }

    normalized = plain_json(original)

    assert normalized == {
        "nested": {
            "values": [1, 2],
        },
        "flag": True,
    }
    assert normalized is not original
    assert normalized["nested"] is not original["nested"]
    assert normalized["nested"]["values"] is not original["nested"]["values"]


def test_plain_json_rejects_a_top_level_non_string_key() -> None:
    payload: dict[object, object] = {1: "value"}

    with pytest.raises(
        TypeError,
        match=r"JSON object keys must be strings at \$; got int",
    ):
        plain_json(payload)


def test_plain_json_reports_the_nested_mapping_path() -> None:
    payload: dict[str, object] = {
        "outer": [
            {
                "inner": {
                    2: "value",
                },
            }
        ]
    }

    with pytest.raises(TypeError) as caught:
        plain_json(payload)

    assert str(caught.value) == (
        "JSON object keys must be strings at $['outer'][0]['inner']; got int"
    )


def test_plain_json_rejects_integer_string_key_collisions() -> None:
    payload: dict[object, object] = {
        1: "integer-key",
        "1": "string-key",
    }

    with pytest.raises(TypeError, match="JSON object keys must be strings"):
        plain_json(payload)


def test_frozen_mapping_wraps_non_string_keys_without_losing_the_cause() -> None:
    payload: dict[object, object] = {
        "nested": {
            1: "value",
        }
    }

    with pytest.raises(
        ValueError,
        match="metadata must be finite JSON data",
    ) as caught:
        frozen_finite_json_mapping(cast(Mapping[str, Any], payload))

    assert isinstance(caught.value.__cause__, TypeError)
    assert "JSON object keys must be strings" in str(caught.value.__cause__)


def test_frozen_mapping_remains_recursively_immutable_and_copyable() -> None:
    frozen = cast(
        FrozenJsonDict,
        frozen_finite_json_mapping(
            {
                "nested": {
                    "values": [1, 2],
                }
            }
        ),
    )
    nested = cast(FrozenJsonDict, frozen["nested"])
    values = cast(FrozenJsonList, nested["values"])

    with pytest.raises(TypeError, match="metadata is immutable"):
        frozen["new"] = "value"
    with pytest.raises(TypeError, match="metadata is immutable"):
        values.append(3)

    thawed = copy.deepcopy(frozen)
    thawed["nested"]["values"].append(3)

    assert thawed == {"nested": {"values": [1, 2, 3]}}
    assert plain_json(frozen) == {"nested": {"values": [1, 2]}}

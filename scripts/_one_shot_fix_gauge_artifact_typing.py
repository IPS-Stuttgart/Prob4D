#!/usr/bin/env python3
"""Apply the exact MyPy-only sparse-artifact repairs."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COMMON = ROOT / "src/prob4d/_gauge_tree_artifact_common.py"
IO = ROOT / "src/prob4d/_gauge_tree_artifact_io.py"


def _replace(path: Path, old: str, new: str, *, expected_count: int = 1) -> None:
    text = path.read_text(encoding="utf-8")
    count = text.count(old)
    if count != expected_count:
        raise SystemExit(
            f"{path}: expected {expected_count} occurrence(s), found {count}: {old!r}"
        )
    path.write_text(text.replace(old, new), encoding="utf-8")


def main() -> None:
    _replace(
        COMMON,
        "        dtype = np.dtype(text)\n",
        "        dtype: np.dtype[Any] = np.dtype(text)\n",
    )
    _replace(
        COMMON,
        '''        return cls(
            path=value.get("path"),
            byte_count=value.get("byte_count"),
            file_sha256=value.get("file_sha256"),
            dtype=value.get("dtype"),
            shape=value.get("shape"),
            content_sha256=value.get("content_sha256"),
        )
''',
        '''        return cls(
            path=_validate_member_path(value["path"]),
            byte_count=require_positive_integer(
                value["byte_count"],
                name="array member byte_count",
            ),
            file_sha256=validate_digest(
                value["file_sha256"],
                name="array member file_sha256",
            ),
            dtype=_validate_dtype(
                value["dtype"],
                name="array member dtype",
            ),
            shape=_require_shape(
                value["shape"],
                name="array member shape",
            ),
            content_sha256=validate_digest(
                value["content_sha256"],
                name="array member content_sha256",
            ),
        )
''',
    )
    _replace(
        IO,
        '''        if version == (1, 0):
            shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(
                buffer,
                max_header_size=MAX_NPY_HEADER_BYTES,
            )
        elif version == (2, 0):
            shape, fortran_order, dtype = np.lib.format.read_array_header_2_0(
                buffer,
                max_header_size=MAX_NPY_HEADER_BYTES,
            )
        else:
            raise ValueError(f"unsupported NPY format version {version}")
''',
        '''        if version == (1, 0):
            header_length_bytes = 2
            header_reader = np.lib.format.read_array_header_1_0
        elif version == (2, 0):
            header_length_bytes = 4
            header_reader = np.lib.format.read_array_header_2_0
        else:
            raise ValueError(f"unsupported NPY format version {version}")
        length_start = buffer.tell()
        length_stop = length_start + header_length_bytes
        if len(payload) < length_stop:
            raise ValueError("truncated NPY header length")
        header_length = int.from_bytes(
            payload[length_start:length_stop],
            byteorder="little",
            signed=False,
        )
        if header_length > MAX_NPY_HEADER_BYTES:
            raise ValueError("NPY header length exceeds its bound")
        shape, fortran_order, dtype = header_reader(buffer)
''',
    )


if __name__ == "__main__":
    main()

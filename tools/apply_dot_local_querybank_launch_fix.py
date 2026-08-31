from __future__ import annotations

from pathlib import Path


PATH = Path(
    ".github/workflows/run-dot-measured-querybank-gpuserver4090-20260901-v1.yml"
)
text = PATH.read_text(encoding="utf-8")

replacements = [
    (
        "protocols/execution_requests/dot-measured-querybank-gpuserver4090-20260901-v1.json",
        "protocols/execution_requests/dot-measured-querybank-gpuserver4090-20260901-v2.json",
        2,
    ),
    (
        '"request_id": "dot-measured-querybank-gpuserver4090-20260901-v1",',
        '"request_id": "dot-measured-querybank-gpuserver4090-20260901-v2",',
        1,
    ),
    (
        "uses: actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1",
        "uses: actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1 # v7.0.1",
        2,
    ),
    (
        "uses: actions/setup-python@5fda3b95a4ea91299a34e894583c3862153e4b97",
        "uses: actions/setup-python@5fda3b95a4ea91299a34e894583c3862153e4b97 # v7.0.0",
        2,
    ),
    (
        "uses: actions/download-artifact@d3f86a106a0bac45b974a628896c90dbdf5c8093",
        "uses: actions/download-artifact@d3f86a106a0bac45b974a628896c90dbdf5c8093 # v4.3.0",
        1,
    ),
    (
        "uses: actions/upload-artifact@ea165f8d65b6e75b540449e92b4886f43607fa02",
        "uses: actions/upload-artifact@ea165f8d65b6e75b540449e92b4886f43607fa02 # v4.6.2",
        1,
    ),
]

for old, new, expected_count in replacements:
    count = text.count(old)
    if count != expected_count:
        raise SystemExit(
            f"expected {expected_count} matches for {old!r}; observed {count}"
        )
    text = text.replace(old, new)

PATH.write_text(text, encoding="utf-8")

from __future__ import annotations

import base64
import hashlib
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PAYLOAD_DIR = ROOT / ".tmp" / "provider-v2-root"

FILES = {
    "src/prob4d/observation_export.py": (
        "observation_export",
        "065b63c632d77f3a00590d7c2edfa814b0b13c6be93bd76b6f5f9b7dded3b34b",
    ),
    "tests/test_observation_export.py": (
        "test_observation_export",
        "ebf1d77fde0c2f2c1ffe8c603f6530e8f4257d15ac9de913ee2c01d6713f18ed",
    ),
}

for target, (prefix, expected_sha256) in FILES.items():
    parts = sorted(PAYLOAD_DIR.glob(f"{prefix}.*.txt"))
    if not parts:
        raise RuntimeError(f"no payload chunks found for {target}")
    encoded = "".join(part.read_text(encoding="utf-8") for part in parts)
    decoded = zlib.decompress(base64.b85decode(encoded.encode("ascii")))
    actual_sha256 = hashlib.sha256(decoded).hexdigest()
    if actual_sha256 != expected_sha256:
        raise RuntimeError(
            f"decoded {target} has SHA-256 {actual_sha256}, expected {expected_sha256}"
        )
    destination = ROOT / target
    destination.write_bytes(decoded)
    print(f"restored {target} ({len(decoded)} bytes, {actual_sha256})")

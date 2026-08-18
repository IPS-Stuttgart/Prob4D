from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATCH_ROOT = ROOT / ".agent-patch"
CHUNK_ROOT = PATCH_ROOT / "chunks"


def main() -> None:
    manifest = json.loads((PATCH_ROOT / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("version") != 1:
        raise ValueError("unsupported patch manifest")
    for entry in manifest["files"]:
        encoded = "".join(
            (CHUNK_ROOT / name).read_text(encoding="ascii").strip()
            for name in entry["chunks"]
        )
        data = base64.b64decode(encoded, validate=True)
        if len(data) != entry["bytes"]:
            raise ValueError(f"byte count changed for {entry['path']}")
        if hashlib.sha256(data).hexdigest() != entry["sha256"]:
            raise ValueError(f"SHA-256 changed for {entry['path']}")
        destination = ROOT / entry["path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)

    for path in sorted(PATCH_ROOT.rglob("*"), reverse=True):
        if path.is_file():
            path.unlink()
        elif path.is_dir():
            path.rmdir()
    PATCH_ROOT.rmdir()
    workflow = ROOT / ".github/workflows/agent-assemble-fused-artifacts.yml"
    workflow.unlink()


if __name__ == "__main__":
    main()

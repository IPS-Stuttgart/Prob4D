#!/usr/bin/env python3
"""Run a separately versioned CUT3R source comparison with native RoPE enforced."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import run_cut3r_source_comparison as legacy

from prob4d.cut3r_runtime_contract import require_compiled_cut3r_rope


class _NativeRopeCut3RRuntime(legacy._Cut3RRuntime):
    """Legacy executor runtime with an additional native-RoPE admission gate."""

    def __init__(self, checkout: Path, checkpoint: Path, *, device: str) -> None:
        self.runtime_receipt = require_compiled_cut3r_rope(checkout)
        super().__init__(checkout, checkpoint, device=device)


def main(argv: Sequence[str] | None = None) -> int:
    """Delegate to the frozen executor after replacing only its runtime constructor."""

    legacy._Cut3RRuntime = _NativeRopeCut3RRuntime
    return legacy.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())

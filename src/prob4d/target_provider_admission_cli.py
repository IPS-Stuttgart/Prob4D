"""Installed command wrappers for held-out target provider admission."""

from __future__ import annotations

import sys
from collections.abc import Sequence

from .target_provider_admission import admit_cli, verify_cli


def main_admit(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    return admit_cli(arguments)


def main_verify(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    return verify_cli(arguments)


__all__ = ["main_admit", "main_verify"]

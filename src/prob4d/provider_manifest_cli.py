"""Print an installed Prob4D observation-provider manifest."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Sequence
from pathlib import Path

ManifestFactory = Callable[..., dict[str, object]]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--api-version",
        type=int,
        choices=(1, 2),
        default=1,
        help="provider API descriptor to emit; version 1 remains the frozen default",
    )
    parser.add_argument(
        "--provider-revision",
        help="explicit source revision; otherwise use installed VCS metadata",
    )
    parser.add_argument("--output", type=Path)
    return parser


def _manifest_factory(api_version: int) -> ManifestFactory:
    if api_version == 1:
        from prob4d.provider_manifest import prob4d_provider_manifest

        return prob4d_provider_manifest
    from prob4d.provider_v2 import prob4d_provider_manifest

    return prob4d_provider_manifest


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = _manifest_factory(args.api_version)(
        provider_revision=args.provider_revision
    )
    text = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Grouped provider-neutral prediction command dispatcher."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from .prediction_provider_manifest import main as legacy_prediction_main


def _help_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="prob4d prediction",
        description="import, validate, and execute provider-neutral predictions",
    )
    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser(
        "import-motioncrafter",
        help="convert an integrity-bound MotionCrafter prediction bundle",
    )
    subparsers.add_parser(
        "import-vggt",
        help="convert an integrity-bound official VGGT sample",
    )
    subparsers.add_parser(
        "import-cut3r-online",
        help="convert depth-reprojected recurrent-online CUT3R outputs",
    )
    subparsers.add_parser(
        "import-cut3r-direct",
        help="convert direct recurrent-online CUT3R XYZ point maps",
    )
    subparsers.add_parser(
        "cut3r-fidelity",
        help="audit direct-pointmap fidelity and recurrent causal-prefix closure",
    )
    subparsers.add_parser(
        "cut3r-comparison",
        help="freeze or inspect a group-aware native-versus-fused CUT3R comparison",
    )
    subparsers.add_parser(
        "cut3r-strata",
        help="freeze and report source-only CUT3R diagnostic strata",
    )
    subparsers.add_parser(
        "cut3r-source-competence",
        help="aggregate complete paired CUT3R source records into competence evidence",
    )
    subparsers.add_parser(
        "compatibility",
        help="build or compare additive semantic-compatibility manifests",
    )
    subparsers.add_parser(
        "import-generic",
        help="import external canonical PredictionWindow payloads",
    )
    subparsers.add_parser(
        "scaffold-generic",
        help="create a no-clobber external-provider import scaffold",
    )
    subparsers.add_parser(
        "validate",
        help="strictly validate a neutral manifest and its payloads",
    )
    subparsers.add_parser(
        "runtime",
        help="causally load, inspect, or exploratorily fuse neutral payloads",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments:
        _help_parser().print_help()
        return 0
    if arguments[0] in {"-h", "--help"}:
        _help_parser().parse_args(arguments)
        return 0
    if arguments[0] == "import-vggt":
        from .vggt_provider_adapter import main as vggt_main

        return int(vggt_main(arguments[1:]))
    if arguments[0] == "import-cut3r-online":
        from .cut3r_provider_adapter import main as cut3r_main

        return int(cut3r_main(arguments[1:]))
    if arguments[0] == "import-cut3r-direct":
        from .cut3r_direct_provider_adapter import main as cut3r_direct_main

        return int(cut3r_direct_main(arguments[1:]))
    if arguments[0] == "cut3r-fidelity":
        from .cut3r_pointmap_fidelity import main as cut3r_fidelity_main

        return int(cut3r_fidelity_main(arguments[1:]))
    if arguments[0] == "cut3r-comparison":
        from .cut3r_comparison import main as cut3r_comparison_main

        return int(cut3r_comparison_main(arguments[1:]))
    if arguments[0] == "cut3r-strata":
        from .cut3r_diagnostic_strata import main as cut3r_strata_main

        return int(cut3r_strata_main(arguments[1:]))
    if arguments[0] == "cut3r-source-competence":
        from .cut3r_source_competence import main as cut3r_source_competence_main

        return int(cut3r_source_competence_main(arguments[1:]))
    if arguments[0] == "compatibility":
        from .semantic_compatibility import main as compatibility_main

        return int(compatibility_main(arguments[1:]))
    if arguments[0] == "import-generic":
        from .prediction_provider_import import main as generic_main

        return int(generic_main(arguments[1:]))
    if arguments[0] == "scaffold-generic":
        from .prediction_provider_scaffold import main as scaffold_main

        return int(scaffold_main(arguments[1:]))
    if arguments[0] == "runtime":
        from .provider_runtime import main as runtime_main

        return int(runtime_main(arguments[1:]))
    return int(legacy_prediction_main(arguments))


if __name__ == "__main__":
    raise SystemExit(main())

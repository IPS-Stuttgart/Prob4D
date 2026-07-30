from __future__ import annotations

from importlib import resources
from importlib.metadata import version

import prob4d


def test_runtime_version_matches_installed_distribution() -> None:
    assert prob4d.__version__ == version("prob4d")


def test_distribution_marks_inline_types() -> None:
    assert resources.files("prob4d").joinpath("py.typed").is_file()

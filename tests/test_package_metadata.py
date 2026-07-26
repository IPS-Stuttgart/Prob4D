from __future__ import annotations

from importlib.metadata import version

import prob4d


def test_runtime_version_matches_installed_distribution() -> None:
    assert prob4d.__version__ == version("prob4d")

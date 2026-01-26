"""Coverage startup hook for subprocesses."""

from __future__ import annotations

import os


def _start_coverage() -> None:
    if not os.environ.get("COVERAGE_PROCESS_START"):
        return
    try:
        import coverage
    except ImportError:
        return
    coverage.process_startup()


_start_coverage()

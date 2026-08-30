"""Event-loop factories for runtime dependencies with platform constraints."""

from __future__ import annotations

import asyncio
import sys


def psycopg_compatible_loop() -> asyncio.AbstractEventLoop:
    """Use Selector on Windows; use the platform default everywhere else."""
    if sys.platform == "win32":
        return asyncio.SelectorEventLoop()
    return asyncio.new_event_loop()

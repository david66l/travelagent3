"""Process-global rate limiter shared by *all* AMap callers.

AMap personal keys are limited to ~3 QPS. The app hits AMap from several places
during a single plan — the distance-matrix builder, the POI collector (paginated
search), and the supplement/restaurant fetches. Each component limiting itself is
not enough: when they overlap they collectively burst past the QPS ceiling and
every call returns ``CUQPS_HAS_EXCEEDED_THE_LIMIT``, which silently drops travel
times back to straight-line haversine and lets the planner zig-zag.

Routing every AMap request through this single gate guarantees the whole process
stays under the key's QPS limit regardless of which features fire concurrently.
"""

from __future__ import annotations

import asyncio

# ~2.5 QPS: comfortably under a personal key's ~3 QPS ceiling.
_MIN_INTERVAL = 0.4

_lock = asyncio.Lock()
_last_ts = 0.0


async def amap_rate_gate() -> None:
    """Block until at least ``_MIN_INTERVAL`` has elapsed since the last request."""
    global _last_ts
    loop = asyncio.get_running_loop()
    async with _lock:
        now = loop.time()
        wait = _last_ts + _MIN_INTERVAL - now
        if wait > 0:
            await asyncio.sleep(wait)
        _last_ts = loop.time()

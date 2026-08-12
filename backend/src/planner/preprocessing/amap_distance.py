"""Real road-network travel times via the AMap (高德) Distance API.

Replaces the straight-line haversine estimate used by ``TransportSelector`` with
real driving distance + duration. This is the single highest-impact fix for the
"行程跨城乱跳 + 15 分钟瞬移" problem: the CP-SAT solver already minimises total
travel time, but it was being fed travel times ~2x too small (haversine ÷ ideal
speed), so crossing the river looked "free". Real durations make zig-zags cost
real day-budget, which forces geographically compact days.

We return a **coordinate-keyed edge map** (``"lat,lng|lat,lng" -> minutes``)
rather than a positional matrix, because the solver mutates its POI list before
building the matrix (reservation filtering removes POIs, meal-break dummy nodes
are appended). A coord-keyed lookup is robust to those changes and is JSON-safe
for the VRP microservice; unknown pairs (e.g. meal dummy nodes with no coords)
fall back to the haversine estimate inside ``TransportSelector``.

Returns ``None`` on any failure so the caller falls back entirely to haversine.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Optional

import httpx

from core.amap_rate import amap_rate_gate

logger = logging.getLogger(__name__)

_DISTANCE_URL = "https://restapi.amap.com/v3/distance"
_MAX_POIS = 50  # guardrail: N origins per call, N calls total
_HTTP_TIMEOUT = 8.0
# AMap personal keys are limited to ~3 QPS. Firing N column calls in a burst
# trips CUQPS_HAS_EXCEEDED_THE_LIMIT, so every road time silently falls back to
# haversine — which makes crossing the river look "free" and lets the solver
# zig-zag across the city. We therefore (a) gate request *starts* to a steady
# rate below the QPS ceiling, (b) retry several times with exponential backoff
# on a rate-limit hit, and (c) persist results to the edge cache so the cost is
# paid only once per city.
_MAX_CONCURRENCY = 1
_QPS_ERROR = "CUQPS_HAS_EXCEEDED_THE_LIMIT"
_MAX_RETRIES = 6
_RETRY_BACKOFF = (0.6, 1.2, 2.0, 3.0, 4.0, 5.0)

# L1: process-local cache keyed by the rounded coordinate set, so identical
# re-plans (reject → re-solve, modify) short-circuit instantly.
_CACHE: dict[tuple, dict[str, int]] = {}
_CACHE_MAX = 128

# L2: per-edge cache (Redis-backed + process-local). Same-city POIs recur across
# requests, so caching individual road-time edges gives high hit rates even when
# the POI set differs slightly, survives restarts, and is shared across requests.
_EDGE_CACHE: dict[str, int] = {}
_EDGE_CACHE_MAX = 50_000
_EDGE_TTL = 30 * 24 * 3600  # road durations are stable; refresh monthly
_REDIS_PREFIX = "amap:edge:"


async def _load_cached_edges(edge_keys: list[str]) -> dict[str, int]:
    """Return {edge_key: minutes} for edges already cached (process-local then Redis)."""
    out: dict[str, int] = {}
    missing: list[str] = []
    for ek in edge_keys:
        cached = _EDGE_CACHE.get(ek)
        if cached is not None:
            out[ek] = cached
        else:
            missing.append(ek)
    if missing:
        try:
            from core.redis_client import redis_cache_client

            vals = await redis_cache_client.mget([_REDIS_PREFIX + k for k in missing])
            for ek, v in zip(missing, vals):
                if v is None:
                    continue
                try:
                    minutes = int(v)
                except (TypeError, ValueError):
                    continue
                out[ek] = minutes
                if len(_EDGE_CACHE) < _EDGE_CACHE_MAX:
                    _EDGE_CACHE[ek] = minutes
        except Exception:
            pass  # Redis unavailable → process-local cache only
    return out


async def _store_edges(edges: dict[str, int]) -> None:
    """Persist freshly fetched edges to the process-local + Redis cache (best-effort)."""
    if not edges:
        return
    for ek, minutes in edges.items():
        if len(_EDGE_CACHE) < _EDGE_CACHE_MAX:
            _EDGE_CACHE[ek] = minutes
    try:
        from core.redis_client import redis_cache_client

        await redis_cache_client.mset_ttl(
            {_REDIS_PREFIX + ek: str(m) for ek, m in edges.items()},
            ttl=_EDGE_TTL,
        )
    except Exception:
        pass


def coord_key(lat: float, lng: float) -> str:
    return f"{round(lat, 5)},{round(lng, 5)}"


def edge_key(lat1: float, lng1: float, lat2: float, lng2: float) -> str:
    return f"{coord_key(lat1, lng1)}|{coord_key(lat2, lng2)}"


def _cache_key(pois: list[Any]) -> tuple:
    return tuple((round(p.lat, 5), round(p.lng, 5)) for p in pois)


def _has_real_coords(pois: list[Any]) -> bool:
    coords = {(round(p.lat, 4), round(p.lng, 4)) for p in pois if p.lat and p.lng}
    return len(coords) >= 2


async def _fetch_column(
    client: httpx.AsyncClient,
    key: str,
    origins: str,
    dest: Any,
    pois: list[Any],
    semaphore: asyncio.Semaphore,
) -> Optional[dict[str, int]]:
    """One AMap call: all origins → a single destination. Returns edge→minutes."""
    params = {
        "key": key,
        "origins": origins,
        "destination": f"{dest.lng},{dest.lat}",
        "type": "1",  # 1 = driving (real road network)
        "output": "json",
    }
    data: dict[str, Any] | None = None
    for attempt in range(_MAX_RETRIES):
        async with semaphore:
            await amap_rate_gate()
            try:
                resp = await client.get(_DISTANCE_URL, params=params)
                data = resp.json()
            except Exception as exc:
                logger.warning("AMap distance call failed: %s", exc)
                return None
        if data.get("status") == "1":
            break
        # Back off and retry when we trip the QPS limit; the rate gate plus
        # exponential backoff lets a personal key fill the whole matrix.
        if data.get("info") == _QPS_ERROR and attempt < _MAX_RETRIES - 1:
            await asyncio.sleep(_RETRY_BACKOFF[attempt])
            continue
        logger.warning("AMap distance non-OK: %s", data.get("info"))
        return None

    if not data or data.get("status") != "1":
        return None

    out: dict[str, int] = {}
    for r in data.get("results") or []:
        try:
            idx = int(r.get("origin_id", 0)) - 1  # AMap origin_id is 1-based
            seconds = float(r.get("duration", 0) or 0)
        except (ValueError, TypeError):
            continue
        if 0 <= idx < len(pois):
            origin = pois[idx]
            out[edge_key(origin.lat, origin.lng, dest.lat, dest.lng)] = max(0, round(seconds / 60))
    return out


async def build_amap_minutes_map(
    pois: list[Any],
    api_key: str,
) -> Optional[dict[str, int]]:
    """Build a coord-keyed driving-time map (minutes) for the given POIs.

    Returns None when AMap is unavailable, coords are missing, or N is too large.
    """
    n = len(pois)
    if not api_key or n < 2 or n > _MAX_POIS or not _has_real_coords(pois):
        return None

    ck = _cache_key(pois)
    if ck in _CACHE:
        return _CACHE[ck]

    def _has_coord(p: Any) -> bool:
        return bool(p.lat or p.lng)

    # L2: seed from the per-edge cache, then only fetch destination columns that
    # still have at least one missing origin→dest edge.
    all_edge_keys = [
        edge_key(o.lat, o.lng, d.lat, d.lng)
        for d in pois
        if _has_coord(d)
        for o in pois
        if _has_coord(o)
    ]
    edge_map: dict[str, int] = await _load_cached_edges(all_edge_keys)

    cols_to_fetch = [
        j
        for j, d in enumerate(pois)
        if _has_coord(d)
        and any(
            _has_coord(o)
            and edge_key(o.lat, o.lng, d.lat, d.lng) not in edge_map
            for o in pois
        )
    ]

    if cols_to_fetch:
        origins = "|".join(f"{p.lng},{p.lat}" for p in pois)
        semaphore = asyncio.Semaphore(_MAX_CONCURRENCY)
        columns: list[Optional[dict[str, int]]] = []
        try:
            async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT) as client:
                columns = await asyncio.gather(
                    *(
                        _fetch_column(client, api_key, origins, pois[j], pois, semaphore)
                        for j in cols_to_fetch
                    )
                )
        except Exception as exc:
            logger.warning("AMap matrix build failed: %s", exc)

        fetched: dict[str, int] = {}
        for column in columns:
            if column:
                fetched.update(column)
        if fetched:
            edge_map.update(fetched)
            await _store_edges(fetched)
    else:
        logger.info("AMap matrix fully served from edge cache (%d edges)", len(edge_map))

    # Keep partial results: the solver falls back to haversine per missing edge,
    # so a few failed columns degrade gracefully instead of discarding everything.
    if not edge_map:
        return None

    if len(_CACHE) < _CACHE_MAX:
        _CACHE[ck] = edge_map
    return edge_map

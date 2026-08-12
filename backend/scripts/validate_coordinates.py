"""Validate (and optionally correct) attraction coordinates against AMap.

Wrong seed coordinates break geographic planning regardless of how accurate the
distance API is: e.g. 上海野生动物园 stored at (31.25, 121.464) — central Shanghai
— instead of its real location in 南汇 (~30.91, 121.76), ~40 km away. The solver
then schedules it with downtown POIs and the plan looks like it zig-zags.

This script looks up each attraction by name on AMap (the authoritative POI
location), compares it to the stored coordinate, and flags any that differ by
more than a threshold. It is **dry-run by default**; pass --apply to write
corrections back to the DB.

Usage:
  python scripts/validate_coordinates.py                 # report all cities
  python scripts/validate_coordinates.py --city 上海      # one city
  python scripts/validate_coordinates.py --threshold 2   # km mismatch threshold
  python scripts/validate_coordinates.py --apply         # write corrections
"""

from __future__ import annotations

import argparse
import asyncio
import math
import sys
from pathlib import Path

import httpx

# Allow running as a standalone script (python scripts/validate_coordinates.py).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from sqlalchemy import text  # noqa: E402

from core.amap_rate import amap_rate_gate  # noqa: E402
from core.database import async_session_maker  # noqa: E402
from core.settings import settings  # noqa: E402

_PLACE_URL = "https://restapi.amap.com/v3/place/text"


def _haversine_km(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    r1, g1, r2, g2 = map(math.radians, (lat1, lng1, lat2, lng2))
    dlat, dlng = r2 - r1, g2 - g1
    s = math.sin(dlat / 2) ** 2 + math.cos(r1) * math.cos(r2) * math.sin(dlng / 2) ** 2
    return 6371 * 2 * math.asin(math.sqrt(s))


def _name_matches(query: str, candidate: str) -> bool:
    """Loose name match so we only trust a clearly-corresponding POI."""
    q, c = query.strip(), candidate.strip()
    return bool(q) and bool(c) and (q == c or q in c or c in q)


async def _amap_lookup(
    client: httpx.AsyncClient, name: str, city: str
) -> tuple[float, float, str] | None:
    """Return (lat, lng, matched_name) for the best AMap match, or None."""
    params = {
        "key": settings.amap_key,
        "keywords": name,
        "city": city,
        "citylimit": "true",
        "offset": 10,
        "page": 1,
        "output": "json",
    }
    await amap_rate_gate()
    try:
        data = (await client.get(_PLACE_URL, params=params)).json()
    except Exception as exc:  # noqa: BLE001
        print(f"  ! AMap lookup failed for {name}: {exc}")
        return None
    if data.get("status") != "1":
        return None
    pois = data.get("pois") or []
    if not pois:
        return None
    # Prefer an exact/loose name match; otherwise fall back to the top hit.
    best = next((p for p in pois if _name_matches(name, p.get("name", ""))), pois[0])
    loc = best.get("location", "")
    try:
        lng_s, lat_s = loc.split(",")
        return float(lat_s), float(lng_s), best.get("name", "")
    except (ValueError, AttributeError):
        return None


async def main() -> None:
    parser = argparse.ArgumentParser(description="Validate attraction coordinates against AMap")
    parser.add_argument("--city", default=None, help="Only check this city")
    parser.add_argument("--threshold", type=float, default=3.0, help="km mismatch threshold")
    parser.add_argument("--limit", type=int, default=0, help="Max rows to check (0 = all)")
    parser.add_argument("--apply", action="store_true", help="Write corrections to DB")
    args = parser.parse_args()

    if not settings.amap_key:
        print("AMAP_KEY not configured; cannot validate.")
        return

    sql = "SELECT id, name, city, lat, lng FROM attractions WHERE status != 'deprecated'"
    params: dict = {}
    if args.city:
        sql += " AND city = :city"
        params["city"] = args.city
    sql += " ORDER BY city, name"
    if args.limit:
        sql += f" LIMIT {int(args.limit)}"

    async with async_session_maker() as db:
        rows = (await db.execute(text(sql), params)).mappings().all()

    print(f"Checking {len(rows)} attractions (threshold {args.threshold} km, "
          f"{'APPLY' if args.apply else 'DRY-RUN'})...\n")

    mismatches: list[dict] = []
    no_match = 0
    async with httpx.AsyncClient(timeout=8.0) as client:
        for r in rows:
            if not (r["lat"] or r["lng"]):
                continue
            found = await _amap_lookup(client, r["name"], r["city"])
            if not found:
                no_match += 1
                continue
            a_lat, a_lng, matched = found
            dist = _haversine_km(r["lat"], r["lng"], a_lat, a_lng)
            if dist > args.threshold:
                mismatches.append({
                    "id": r["id"], "name": r["name"], "city": r["city"],
                    "old": (r["lat"], r["lng"]), "new": (a_lat, a_lng),
                    "matched": matched, "dist": dist,
                })

    if not mismatches:
        print(f"No coordinate mismatches > {args.threshold} km. "
              f"({no_match} POIs had no AMap match.)")
        return

    mismatches.sort(key=lambda m: m["dist"], reverse=True)
    print(f"Found {len(mismatches)} suspicious coordinates "
          f"({no_match} POIs had no AMap match):\n")
    for m in mismatches:
        print(f"  [{m['dist']:6.1f} km] {m['city']} · {m['name']}")
        print(f"      stored: {m['old'][0]:.5f},{m['old'][1]:.5f}  "
              f"→ AMap: {m['new'][0]:.5f},{m['new'][1]:.5f}  (matched: {m['matched']})")

    if not args.apply:
        print(f"\nDRY-RUN: {len(mismatches)} rows would be corrected. "
              f"Re-run with --apply to write them.")
        return

    async with async_session_maker() as db:
        for m in mismatches:
            await db.execute(
                text("UPDATE attractions SET lat = :lat, lng = :lng WHERE id = :id"),
                {"lat": m["new"][0], "lng": m["new"][1], "id": m["id"]},
            )
        await db.commit()
    print(f"\nApplied {len(mismatches)} coordinate corrections.")


if __name__ == "__main__":
    asyncio.run(main())

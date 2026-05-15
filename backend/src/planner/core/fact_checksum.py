"""Fact Checksum — verify that Writer never mutates itinerary facts.

Protected fields: poi_name, start_time, end_time, duration_min, location
(lat/lng), ticket_price, day_number.  These are structural — a Writer may
decorate but never alter them.
"""

import hashlib
import json

from schemas import DayPlan

_PROTECTED_FIELDS = frozenset({
    "poi_name", "start_time", "end_time", "duration_min",
    "ticket_price", "day_number",
})


def compute_checksum(itinerary: list[DayPlan]) -> str:
    """Return a stable hex digest over the protected fields of every activity."""
    payload = _build_payload(itinerary)
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()


def verify_checksum(
    original: list[DayPlan], enriched: list[DayPlan]
) -> bool:
    """True iff the protected fields in *enriched* match *original*."""
    return compute_checksum(original) == compute_checksum(enriched)


def _build_payload(itinerary: list[DayPlan]) -> list[dict]:
    """Extract only the protected fields into a canonical list-of-dicts."""
    rows: list[dict] = []
    for day in itinerary:
        for act in day.activities:
            location_payload = None
            if act.location:
                location_payload = {"lat": act.location.lat, "lng": act.location.lng}
            row = {
                "poi_name": act.poi_name,
                "start_time": act.start_time,
                "end_time": act.end_time,
                "duration_min": act.duration_min,
                "ticket_price": act.ticket_price,
                "day_number": day.day_number,
                "location": location_payload,
            }
            rows.append(row)
    rows.sort(key=lambda r: (r["day_number"], r["poi_name"], r.get("start_time", "") or ""))
    return rows

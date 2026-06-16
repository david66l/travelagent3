"""Fact Guard — verify that Writer never mutates itinerary facts.

Protected fields: poi_name, start_time, end_time, duration_min, location
(lat/lng), ticket_price. These are structural facts: Writer may decorate an
activity with prose and tags, but must not alter them.

This intentionally uses direct field comparison instead of SHA256 hashing.
The threat is unintentional LLM mutation, not adversarial tampering, and direct
comparison makes tests and fallback logs easier to reason about.
"""

from schemas import Activity

_PROTECTED_FIELDS = frozenset(
    {
        "poi_name",
        "start_time",
        "end_time",
        "duration_min",
        "ticket_price",
    }
)


def activity_fields_match(original: Activity, enriched: Activity) -> bool:
    """True iff every protected field in *enriched* matches *original*."""
    if original.poi_name != enriched.poi_name:
        return False
    if original.start_time != enriched.start_time:
        return False
    if original.end_time != enriched.end_time:
        return False
    if original.duration_min != enriched.duration_min:
        return False
    if original.ticket_price != enriched.ticket_price:
        return False

    orig_has_loc = original.location is not None
    enr_has_loc = enriched.location is not None
    if orig_has_loc != enr_has_loc:
        return False
    if orig_has_loc and enr_has_loc:
        if original.location.lat != enriched.location.lat:
            return False
        if original.location.lng != enriched.location.lng:
            return False

    return True


protected_fields_match = activity_fields_match


def protected_field_differences(original: Activity, enriched: Activity) -> list[str]:
    """Return protected field names whose values differ."""
    changed: list[str] = []
    if original.poi_name != enriched.poi_name:
        changed.append("poi_name")
    if original.start_time != enriched.start_time:
        changed.append("start_time")
    if original.end_time != enriched.end_time:
        changed.append("end_time")
    if original.duration_min != enriched.duration_min:
        changed.append("duration_min")
    if original.ticket_price != enriched.ticket_price:
        changed.append("ticket_price")

    orig_has_loc = original.location is not None
    enr_has_loc = enriched.location is not None
    if orig_has_loc != enr_has_loc:
        changed.append("location")
    elif orig_has_loc and enr_has_loc:
        if original.location.lat != enriched.location.lat:
            changed.append("location.lat")
        if original.location.lng != enriched.location.lng:
            changed.append("location.lng")

    return changed

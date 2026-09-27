"""Unit tests for shared LangGraph node implementations."""


def _make_poi_dict(spot_id: str, name: str) -> dict:
    return {
        "spot_id": spot_id,
        "spot_name": name,
        "spot_type": "attraction",
        "city": "北京",
        "lat": 39.9,
        "lng": 116.4,
        "ticket_price": 60.0,
        "tags": ["历史"],
        "duration_minutes": 180,
        "walk_intensity": 3,
        "need_reservation": False,
        "reservation_reminder": False,
    }

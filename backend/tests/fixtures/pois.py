"""Test data factories for POIs."""

from schemas import ScoredPOI, Location


def make_poi(
    name: str = "测试景点",
    category: str = "attraction",
    score: float = 0.8,
    lat: float = 39.9,
    lng: float = 116.4,
    **overrides,
) -> ScoredPOI:
    """Factory for constructing ScoredPOI test data."""
    return ScoredPOI(
        name=name,
        category=category,
        score=score,
        location=Location(lat=lat, lng=lng),
        **overrides,
    )

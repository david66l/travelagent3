"""Price Query Skill - query ticket/meal/hotel prices."""

from schemas import PriceInfo
from skills.web_search import WebSearchSkill


class PriceQuerySkill:
    """Query prices for POIs (tickets, meals, hotels)."""

    def __init__(self):
        self.search_skill = WebSearchSkill()

    async def query_price(
        self,
        poi_name: str,
        city: str,
        price_type: str,  # ticket / meal / hotel
    ) -> PriceInfo:
        """Query price for a specific POI."""
        query_map = {
            "ticket": f"{city} {poi_name} 门票价格",
            "meal": f"{city} {poi_name} 人均消费",
            "hotel": f"{city} {poi_name} 住宿价格",
        }
        query = query_map.get(price_type, f"{city} {poi_name} 价格")

        results = await self.search_skill.search(query, top_n=3)

        # For MVP, return estimated prices based on type
        price_ranges = {
            "ticket": "50-200元",
            "meal": "80-200元/人",
            "hotel": "300-800元/晚",
        }

        return PriceInfo(
            poi_name=poi_name,
            price_type=price_type,
            price_range=price_ranges.get(price_type),
            source=results[0].url if results else "",
        )

"""Travel Retrieval RAG Agent — hybrid structured + vector + BM25 retrieval for POIs."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Optional

from data.embedding import get_embedder
from data.retrieval_fallback import RetrievalFallback
from data.retrieval_repository import RetrievalRepository, retrieval_repo
from models.poi import POI
from models.travel_slots import TravelSlots

logger = logging.getLogger(__name__)


class TravelRetrievalRAGAgent:
    """Hybrid RAG retrieval agent.

    Pipeline:
      1. Build search query from slots + profile
      2. Parallel structured / vector / BM25 retrieval
      3. RRF fusion ranking
      4. Real-time enhancement (MVP mock)
      5. Mark must-visit reservation reminders
      6. Fallback if empty
    """

    RRF_K = 60
    TOP_K = 15
    HNSW_EF_SEARCH = 128

    def __init__(
        self,
        repo: Optional[RetrievalRepository] = None,
        fallback: Optional[RetrievalFallback] = None,
    ):
        self._repo = repo or retrieval_repo
        self._fallback = fallback or RetrievalFallback(repo=self._repo)

    async def retrieve(
        self,
        slots: TravelSlots,
        profile: Optional[dict[str, Any]] = None,
        *,
        top_k: int = TOP_K,
    ) -> dict[str, Any]:
        """Main entry: return Top-K POI candidates + retrieval metadata."""
        profile = profile or {}
        search_query = self._build_search_query(slots, profile)

        # Parallel three-way retrieval with exception isolation
        structured_task = asyncio.create_task(self._search_structured(slots, profile))
        vector_task = asyncio.create_task(self._search_vector(search_query, slots, profile))
        bm25_task = asyncio.create_task(self._search_bm25(search_query, slots, profile))

        structured_results, vector_results, bm25_results = await asyncio.gather(
            structured_task, vector_task, bm25_task, return_exceptions=True
        )

        structured_results = self._guard(structured_results, "structured")
        vector_results = self._guard(vector_results, "vector")
        bm25_results = self._guard(bm25_results, "bm25")

        # RRF fusion
        merged = self._rrf_fusion(structured_results, vector_results, bm25_results)
        top_pois = merged[:top_k]

        retrieval_empty = len(top_pois) == 0
        fallback_used = False
        fallback_reason = None

        if retrieval_empty:
            fallback_result = await self._fallback.fallback(slots, profile, min_results=top_k)
            top_pois = fallback_result.get("poi_candidates", [])
            retrieval_empty = fallback_result.get("retrieval_empty", True)
            fallback_used = fallback_result.get("fallback_used", False)
            fallback_reason = fallback_result.get("fallback_reason")

        # Real-time enhancement (MVP mock, fire-and-forget)
        if top_pois:
            asyncio.create_task(self._enhance_realtime(top_pois))

        # Mark must-visit reservation reminders
        must_visit = set(slots.must_visit or [])
        for poi in top_pois:
            poi.reservation_reminder = (
                poi.spot_name in must_visit and poi.need_reservation
            )

        return {
            "poi_candidates": [poi.model_dump() for poi in top_pois],
            "retrieval_query": search_query,
            "retrieval_empty": retrieval_empty,
            "retrieval_stats": {
                "structured_count": len(structured_results),
                "vector_count": len(vector_results),
                "bm25_count": len(bm25_results),
                "merged_count": len(merged),
                "returned_count": len(top_pois),
                "fallback_used": fallback_used,
                "fallback_reason": fallback_reason,
            },
        }

    @staticmethod
    def _guard(result: Any, name: str) -> list[POI]:
        """Return empty list if a retrieval task raised an exception."""
        if isinstance(result, Exception):
            logger.warning("%s retrieval failed: %s", name, result)
            return []
        return result

    @classmethod
    def _build_search_query(
        cls,
        slots: TravelSlots,
        profile: dict[str, Any],
    ) -> str:
        """Compose a query string from slots and profile for vector/BM25 search."""
        parts: list[str] = []
        if slots.destination:
            parts.append(slots.destination)
        if slots.interests:
            parts.append(" ".join(slots.interests))
        if slots.must_visit:
            parts.append(" ".join(slots.must_visit))
        if profile.get("interests"):
            parts.append(" ".join(profile["interests"]))
        if slots.play_mode:
            parts.append(slots.play_mode)

        query = " ".join(parts).strip()
        if not query and slots.destination:
            query = f"{slots.destination} 旅游景点"
        if not query:
            query = "旅游景点"
        return query

    async def _search_structured(
        self,
        slots: TravelSlots,
        profile: dict[str, Any],
    ) -> list[POI]:
        """Structured SQL pre-filtering based on physical constraints and profile."""
        city = slots.destination
        if not city:
            return []

        # Map profile constraints to filters
        max_walk = None
        if slots.max_walk_minutes is not None:
            # Rough mapping: max_walk_minutes -> walk_intensity ceiling
            if slots.max_walk_minutes <= 120:
                max_walk = 2
            elif slots.max_walk_minutes <= 180:
                max_walk = 3
            else:
                max_walk = 4

        wheelchair_only = bool(slots.has_wheelchair)

        avoid_tags = set()
        if slots.food_taboos:
            avoid_tags.update(slots.food_taboos)
        if slots.must_not_visit:
            avoid_tags.update(slots.must_not_visit)

        suitable_for: list[str] = []
        if slots.travel_companion == "family" or slots.has_children:
            suitable_for.append("family_kid")
        if slots.has_elderly:
            suitable_for.append("family_elder")
        if slots.travel_companion == "couple":
            suitable_for.append("couple")
        if slots.travel_companion == "alone":
            suitable_for.append("solo")
        if slots.travel_companion == "friends":
            suitable_for.append("friends")

        return await self._repo.search_structured(
            city,
            max_walk_intensity=max_walk,
            wheelchair_only=wheelchair_only,
            avoid_tags=list(avoid_tags) if avoid_tags else None,
            suitable_for=suitable_for if suitable_for else None,
            limit=50,
        )

    async def _search_vector(
        self,
        search_query: str,
        slots: TravelSlots,
        profile: dict[str, Any],
    ) -> list[POI]:
        """pgvector semantic search."""
        if not search_query or not slots.destination:
            return []
        return await self._repo.search_vector(
            search_query,
            city=slots.destination,
            limit=50,
            ef_search=self.HNSW_EF_SEARCH,
        )

    async def _search_bm25(
        self,
        search_query: str,
        slots: TravelSlots,
        profile: dict[str, Any],
    ) -> list[POI]:
        """PostgreSQL tsvector BM25 search."""
        if not search_query or not slots.destination:
            return []
        return await self._repo.search_bm25(
            search_query,
            city=slots.destination,
            limit=50,
        )

    @classmethod
    def _rrf_fusion(cls, *result_lists: list[POI]) -> list[POI]:
        """Reciprocal Rank Fusion across multiple ranked result lists."""
        scores: dict[str, float] = {}
        sources: dict[str, POI] = {}

        for results in result_lists:
            for rank, poi in enumerate(results, start=1):
                sid = poi.spot_id
                sources[sid] = poi
                scores[sid] = scores.get(sid, 0.0) + 1.0 / (cls.RRF_K + rank)

        # Sort by RRF score descending
        ranked_ids = sorted(scores.keys(), key=lambda sid: scores[sid], reverse=True)
        merged = []
        for sid in ranked_ids:
            poi = sources[sid]
            poi.rrf_score = scores[sid]
            merged.append(poi)
        return merged

    @staticmethod
    async def _enhance_realtime(pois: list[POI]) -> None:
        """Enhance POIs with real-time data (MVP mock)."""
        for poi in pois:
            if poi.current_weather is None:
                poi.current_weather = "晴朗"
            if poi.current_queue_time is None:
                poi.current_queue_time = poi.queue_time_avg
            if poi.is_open_today is None:
                poi.is_open_today = True

    async def embed_query(self, query_text: str) -> list[float]:
        """Utility: encode a query text for external callers."""
        embedder = await get_embedder()
        return embedder.encode_single(query_text)

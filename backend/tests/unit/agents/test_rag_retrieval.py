"""Unit tests for TravelRetrievalRAGAgent."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from agents.rag_retrieval import TravelRetrievalRAGAgent
from models.poi import POI
from models.travel_slots import TravelSlots


def _make_poi(spot_id: str, name: str, source: str = "structured") -> POI:
    return POI(
        spot_id=spot_id,
        spot_name=name,
        spot_type="attraction",
        city="北京",
        lat=39.9,
        lng=116.4,
        source=source,
    )


@pytest.fixture
def agent():
    return TravelRetrievalRAGAgent()


@pytest.mark.asyncio
async def test_build_search_query(agent):
    slots = TravelSlots(
        destination="北京",
        interests=["历史", "文化"],
        must_visit=["故宫"],
    )
    query = agent._build_search_query(slots, {})
    assert "北京" in query
    assert "历史" in query
    assert "故宫" in query


@pytest.mark.asyncio
async def test_rrf_fusion_ranks_by_reciprocal_rank(agent):
    list1 = [_make_poi("a", "A", "structured"), _make_poi("b", "B", "structured")]
    list2 = [_make_poi("b", "B", "vector"), _make_poi("c", "C", "vector")]
    merged = agent._rrf_fusion(list1, list2)

    assert len(merged) == 3
    assert merged[0].spot_id == "b"  # appears in both lists
    assert all(poi.rrf_score is not None for poi in merged)


@pytest.mark.asyncio
async def test_retrieve_returns_top_k_and_marks_reservation(agent):
    structured = [_make_poi(f"s{i}", f"结构化{i}") for i in range(5)]
    vector = [_make_poi(f"v{i}", f"向量{i}", "vector") for i in range(5)]
    bm25 = [_make_poi(f"b{i}", f"BM25{i}", "bm25") for i in range(5)]

    repo = MagicMock()
    repo.search_structured = AsyncMock(return_value=structured)
    repo.search_vector = AsyncMock(return_value=vector)
    repo.search_bm25 = AsyncMock(return_value=bm25)

    agent_with_mocks = TravelRetrievalRAGAgent(repo=repo)
    slots = TravelSlots(
        destination="北京",
        travel_days=3,
        must_visit=["结构化0"],
    )

    result = await agent_with_mocks.retrieve(slots, top_k=5)

    assert len(result["poi_candidates"]) == 5
    assert result["retrieval_query"]
    assert result["retrieval_empty"] is False
    assert result["retrieval_stats"]["merged_count"] == 15


@pytest.mark.asyncio
async def test_retrieve_triggers_fallback_when_empty(agent):
    repo = MagicMock()
    repo.search_structured = AsyncMock(return_value=[])
    repo.search_vector = AsyncMock(return_value=[])
    repo.search_bm25 = AsyncMock(return_value=[])

    fallback = MagicMock()
    fallback.fallback = AsyncMock(
        return_value={
            "poi_candidates": [_make_poi("fb-1", "Fallback", "structured")],
            "retrieval_empty": False,
            "fallback_used": True,
            "fallback_reason": "hardcoded_safety_set",
        }
    )

    agent_with_fallback = TravelRetrievalRAGAgent(repo=repo, fallback=fallback)
    slots = TravelSlots(destination="北京", travel_days=3)

    result = await agent_with_fallback.retrieve(slots, top_k=15)

    assert result["retrieval_empty"] is False
    assert result["retrieval_stats"]["fallback_used"] is True
    fallback.fallback.assert_awaited_once()

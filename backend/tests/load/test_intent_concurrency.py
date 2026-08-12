"""High-concurrency test for intent recognition after the DeepSeek Flash switch.

Intent recognition now routes to the DeepSeek cloud model instead of the local
Qwen, so the cloud path takes the full intent load. This fires many concurrent
intent calls through the real routing path (`_prepare_request` -> `select_model`
-> `_create_completion`) with only the network mocked, and asserts:

  * every concurrent call routes to DeepSeek Flash (settings.llm_model),
  * zero failures under load,
  * the deterministic parser fallback still absorbs cloud errors under load.

The global ``llm`` singleton is mocked out by ``conftest.py``, so we build a
fresh real ``LLMClient`` here to exercise the actual routing code.

Run standalone for a throughput readout:
    .venv/bin/python -m pytest tests/load/test_intent_concurrency.py -s -q
"""

import asyncio
import os
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from agents import demand_parser
from core import llm_client
from core.llm_client import LLMClient
from core.settings import settings
from models.travel_slots import SlotParseOutput

CONCURRENCY = int(os.environ.get("VUS", "500"))
NET_LATENCY_S = 0.02  # simulated DeepSeek round-trip

_INTENT_JSON = (
    '{"intent":"generate_itinerary","confidence":0.9,'
    '"sentiment":"neutral","slots":{},"missing_slots":[]}'
)


def _fake_completion(seen_models: list[str]):
    """Mock _create_completion: record the routed model, simulate latency."""

    async def _call(**kwargs):
        seen_models.append(kwargs.get("model"))
        await asyncio.sleep(NET_LATENCY_S)
        message = SimpleNamespace(content=_INTENT_JSON)
        choice = SimpleNamespace(message=message)
        return SimpleNamespace(choices=[choice], usage=None)

    return _call


@pytest.fixture
def cloud_ready(monkeypatch):
    monkeypatch.setattr(settings, "deepseek_api_key", "sk-test", raising=False)
    monkeypatch.setattr(settings, "local_llm_enabled", True, raising=False)
    monkeypatch.setattr(settings, "local_llm_model", "qwen2.5-7b-instruct", raising=False)
    monkeypatch.setattr(settings, "llm_model", "deepseek-v4-flash", raising=False)
    return settings


@pytest.mark.asyncio
async def test_intent_high_concurrency_routes_to_flash(cloud_ready):
    client = LLMClient()  # real client; the global singleton is mocked by conftest
    seen_models: list[str] = []
    messages = [{"role": "user", "content": "我想去成都玩三天"}]

    with patch.object(
        llm_client, "is_cost_circuit_active", new=AsyncMock(return_value=False)
    ), patch.object(client, "_create_completion", new=_fake_completion(seen_models)):
        start = time.monotonic()
        results = await asyncio.gather(
            *(
                client.structured_call(messages, SlotParseOutput, task_type="intent")
                for _ in range(CONCURRENCY)
            ),
            return_exceptions=True,
        )
        elapsed = time.monotonic() - start

    failures = [r for r in results if isinstance(r, Exception)]
    qps = CONCURRENCY / elapsed if elapsed else 0
    print(
        f"\n[intent-concurrency] vus={CONCURRENCY} "
        f"ok={len(results) - len(failures)} fail={len(failures)} "
        f"elapsed={elapsed:.3f}s throughput={qps:.0f} req/s"
    )

    assert not failures, f"{len(failures)} concurrent calls failed: {failures[:3]}"
    assert len(seen_models) == CONCURRENCY, "every call must reach the cloud client"
    assert set(seen_models) == {settings.llm_model}, (
        f"all intent calls must route to DeepSeek Flash, saw: {set(seen_models)}"
    )
    assert all(r.intent for r in results), "every parse must yield an intent"


@pytest.mark.asyncio
async def test_intent_fallback_holds_under_concurrency(cloud_ready):
    """Cloud failures must degrade to the deterministic parser, never raise."""
    parser = demand_parser.DemandParserAgent()
    client = LLMClient()

    async def _boom(**kwargs):
        await asyncio.sleep(NET_LATENCY_S)
        raise RuntimeError("simulated DeepSeek 5xx")

    with patch.object(
        llm_client, "is_cost_circuit_active", new=AsyncMock(return_value=False)
    ), patch.object(client, "_create_completion", new=_boom), patch.object(
        demand_parser, "llm", client
    ):
        results = await asyncio.gather(
            *(parser.parse("北京三日游", [], None) for _ in range(CONCURRENCY)),
            return_exceptions=True,
        )

    failures = [r for r in results if isinstance(r, Exception)]
    assert not failures, f"fallback must absorb cloud errors, got {len(failures)} raises"
    assert all(r.intent for r in results), "fallback must still produce an intent"

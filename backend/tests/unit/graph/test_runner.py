"""Tests for the graph runner integration."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class _AsyncIter:
    def __init__(self, items):
        self._items = items

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._items:
            raise StopAsyncIteration
        return self._items.pop(0)


def test_extract_result_normalizes_memory_writeback_to_completed():
    from graph.runner import _extract_result

    result = _extract_result(
        {
            "stage": "memory_updated",
            "messages": [{"role": "assistant", "content": "done"}],
            "booking_results": {"source": "mock"},
        }
    )
    assert result["stage"] == "completed"


@pytest.mark.asyncio
async def test_run_graph_turn_returns_final_state():
    from graph.runner import run_graph_turn

    mock_graph = MagicMock()
    mock_graph.astream = MagicMock(return_value=_AsyncIter([]))
    mock_graph.aget_state = AsyncMock(
        return_value=MagicMock(values={"stage": "completed", "messages": []})
    )

    with patch("graph.runner.get_graph", new=AsyncMock(return_value=mock_graph)):
        with patch(
            "graph.runner.SessionManager.create",
            new=AsyncMock(return_value={"user_input": "北京3天"}),
        ):
            with patch("graph.runner.SessionManager.save", new=AsyncMock()):
                result = await run_graph_turn("s1", "u1", "北京3天")

    assert result["stage"] == "completed"


@pytest.mark.asyncio
async def test_stream_graph_events_yields_final():
    from graph.runner import stream_graph_events

    mock_graph = MagicMock()
    mock_graph.astream_events = MagicMock(
        return_value=_AsyncIter(
            [
                {"event": "on_chain_start", "name": "understand", "data": {}},
                {
                    "event": "on_chain_end",
                    "name": "output",
                    "data": {
                        "output": {
                            "stage": "awaiting_booking",
                            "messages": [
                                {"role": "assistant", "content": "# 北京", "type": "itinerary"}
                            ],
                            "output_pdf_url": "http://test/pdf",
                        }
                    },
                },
            ]
        )
    )
    mock_graph.aget_state = AsyncMock(
        return_value=MagicMock(
            next=[],
            values={
                "stage": "awaiting_booking",
                "messages": [{"role": "assistant", "content": "# 北京"}],
                "output_pdf_url": "http://test/pdf",
            },
        )
    )

    with patch("graph.runner.get_graph", new=AsyncMock(return_value=mock_graph)):
        with patch(
            "graph.runner.SessionManager.create",
            new=AsyncMock(return_value={"user_input": "北京3天"}),
        ):
            with patch("graph.runner.SessionManager.save", new=AsyncMock()):
                events = []
                async for event in stream_graph_events("s1", "u1", "北京3天"):
                    events.append(event)

    assert any(e["type"] == "thinking" for e in events)
    assert any(e["type"] == "partial" for e in events)
    final = [e for e in events if e["type"] == "final"][0]
    assert final["payload"]["output_pdf_url"] == "http://test/pdf"


@pytest.mark.asyncio
async def test_stream_graph_events_graph_interrupt_yields_awaiting_confirm():
    from langgraph.errors import GraphInterrupt
    from langgraph.types import Interrupt

    from graph.runner import stream_graph_events

    itinerary = [{"day_number": 1, "activities": [], "total_cost": 0.0}]
    interrupt_exc = GraphInterrupt(
        (
            Interrupt(
                value={"type": "awaiting_confirm", "itinerary": itinerary},
                resumable=True,
                ns=["confirm_gate"],
            ),
        )
    )

    mock_graph = MagicMock()
    mock_graph.astream_events = MagicMock(side_effect=interrupt_exc)
    unpaused = MagicMock(next=[], values={})
    paused = MagicMock(
        next=["confirm_gate"],
        values={
            "itinerary": itinerary,
            "warnings": [],
            "messages": [{"role": "assistant", "content": "# 上海"}],
        },
    )
    mock_graph.aget_state = AsyncMock(side_effect=[unpaused, paused])

    with patch("graph.runner.get_graph", new=AsyncMock(return_value=mock_graph)):
        with patch(
            "graph.runner.SessionManager.create",
            new=AsyncMock(return_value={"user_input": "上海5天"}),
        ):
            with patch("graph.runner.SessionManager.save", new=AsyncMock()):
                with patch("graph.runner.patch_langsmith_root_run_for_pause") as patch_ls:
                    events = []
                    async for event in stream_graph_events("s1", "u1", "上海5天"):
                        events.append(event)

    assert not any(e["type"] == "error" for e in events)
    confirm = [e for e in events if e["type"] == "awaiting_confirm"]
    assert len(confirm) == 1
    assert confirm[0]["payload"]["itinerary"] == itinerary
    patch_ls.assert_called_once_with(interrupt_exc)


@pytest.mark.asyncio
async def test_stream_graph_events_error():
    from graph.runner import stream_graph_events

    mock_graph = MagicMock()
    mock_graph.astream_events = MagicMock(side_effect=RuntimeError("graph failed"))

    with patch("graph.runner.get_graph", new=AsyncMock(return_value=mock_graph)):
        with patch(
            "graph.runner.SessionManager.create",
            new=AsyncMock(return_value={"user_input": "北京3天"}),
        ):
            events = []
            async for event in stream_graph_events("s1", "u1", "北京3天"):
                events.append(event)

    assert any(e["type"] == "error" for e in events)


@pytest.mark.asyncio
async def test_graph_runner_run():
    from graph.runner import GraphRunner

    runner = GraphRunner()
    with patch("graph.runner.run_graph_turn", new=AsyncMock(return_value={"stage": "completed"})):
        result = await runner.run("北京3天", session_id="s1", user_id="u1")
    assert result["stage"] == "completed"


@pytest.mark.asyncio
async def test_graph_runner_stream():
    from graph.runner import GraphRunner

    runner = GraphRunner()

    async def _mock_stream(*args, **kwargs):
        yield {"type": "final", "stage": "completed", "payload": {}}

    with patch("graph.runner.stream_graph_events", new=_mock_stream):
        events = []
        async for event in runner.stream("北京3天", session_id="s1", user_id="u1"):
            events.append(event)
    assert events[0]["type"] == "final"

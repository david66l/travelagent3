"""
Agent Runner — LangGraph 6-Agent 运行器。
"""

from __future__ import annotations

import logging
from typing import Any

from langgraph.graph import StateGraph

logger = logging.getLogger(__name__)


class AgentRunner:
    def __init__(self):
        self._graph: StateGraph | None = None

    async def get_graph(self) -> StateGraph:
        if self._graph is None:
            from agent.graph import get_graph
            self._graph = await get_graph()
        return self._graph

    async def invoke(
        self,
        user_input: str,
        *,
        session_id: str,
        user_id: str = "anonymous",
        user_role: str = "guest",
        thread_id: str | None = None,
        messages: list[dict] | None = None,
        profile: dict | None = None,
    ) -> dict:
        graph = await self.get_graph()
        config = {"configurable": {"thread_id": thread_id or session_id}}

        initial = {
            "session_id": session_id,
            "user_id": user_id,
            "user_input": user_input,
            "user_role": user_role,
            "messages": messages or [],
            "profile": profile or {},
            "stage": "start",
            "turn_count": len(messages or []) // 2 + 1,
        }

        try:
            result = await graph.ainvoke(initial, config)
            logger.info("Agent done: session=%s stage=%s", session_id, result.get("stage"))
            return result
        except Exception as exc:
            logger.exception("Agent failed: %s", exc)
            return {**initial, "stage": "error", "messages": initial["messages"] + [{
                "role": "assistant", "content": "系统暂时遇到问题，请稍后重试。", "type": "error"}]}


runner = AgentRunner()

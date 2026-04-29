"""Thought Logger - comprehensive logging for AI thinking process.

Records every step, LLM call, web search, reasoning, timing, and token usage.
Uses contextvars for safe parallel node logging.
Supports real-time WebSocket status push and Markdown log export.

Logs saved to:
  - logs/thoughts/{date}/{timestamp}_{session_id}.json
  - logs/runs/{date}/{timestamp}_{session_id}.md
"""

import asyncio
import contextvars
import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Optional


# ===== Data Models =====

@dataclass
class LLMCallRecord:
    model: str
    prompt_tokens: int
    completion_tokens: int
    step_name: str = ""
    timestamp: str = ""


@dataclass
class SearchResultRecord:
    step_name: str
    query: str
    result_count: int
    result_titles: list[str]
    timestamp: str = ""


@dataclass
class ReasoningRecord:
    step_name: str
    reasoning: str
    timestamp: str = ""


@dataclass
class StepRecord:
    step_name: str
    start_time: str
    input_summary: str = ""
    end_time: str = ""
    duration_ms: int = 0
    output_summary: str = ""
    status: str = "running"
    error: str = ""
    llm_calls: list[LLMCallRecord] = field(default_factory=list)
    search_results: list[SearchResultRecord] = field(default_factory=list)
    reasoning: Optional[ReasoningRecord] = None


class SessionData:
    """Per-session logging data."""

    def __init__(self, session_id: str, user_input: str) -> None:
        self.session_id = session_id
        self.user_input = user_input
        self.start_time = datetime.now()
        self.steps: list[StepRecord] = []
        self._active_steps: dict[str, StepRecord] = {}
        self.first_llm_response_time: Optional[float] = None
        self.is_finished = False


# ===== Context Variable for Parallel Safety =====

_current_step_name: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "_current_step_name", default=None
)

_current_session_id: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "_current_session_id", default=None
)


def set_current_step_name(name: str) -> None:
    """Set the current step name for this asyncio task."""
    _current_step_name.set(name)


def get_current_step_name() -> Optional[str]:
    """Get the current step name for this asyncio task."""
    return _current_step_name.get()


def set_current_session_id(session_id: str) -> None:
    """Set the current session id for this asyncio task."""
    _current_session_id.set(session_id)


def get_current_session_id() -> Optional[str]:
    """Get the current session id for this asyncio task."""
    return _current_session_id.get()


# ===== Main Logger =====

class ThoughtLogger:
    """Thread-safe thought logger with parallel step isolation.

    Each session has its own SessionData, so concurrent graph executions
    from different sessions do not interfere with each other.

    New capabilities:
    - Real-time WebSocket status push via registered callbacks
    - TTFT (Time To First Token) tracking
    - Markdown log export alongside JSON
    """

    def __init__(self) -> None:
        # Per-session data storage
        self._sessions: dict[str, SessionData] = {}
        # WebSocket callbacks: session_id -> async send function
        self._ws_callbacks: dict[str, Callable[[dict], Any]] = {}
        # Throttle: last push timestamp per session
        self._last_push_time: dict[str, float] = {}
        self._push_interval_ms: int = 150  # min interval between pushes

    # ===== Session Lifecycle =====

    def start_session(self, session_id: str, user_input: str) -> None:
        """Start a new logging session."""
        self._sessions[session_id] = SessionData(session_id, user_input)
        self._last_push_time.pop(session_id, None)

    def _get_session(self, session_id: str) -> Optional[SessionData]:
        return self._sessions.get(session_id)

    # ===== WebSocket Callback Registry =====

    def register_ws_callback(self, session_id: str, callback: Callable[[dict], Any]) -> None:
        """Register a WebSocket callback for a session."""
        self._ws_callbacks[session_id] = callback

    def unregister_ws_callback(self, session_id: str) -> None:
        """Unregister a WebSocket callback for a session."""
        self._ws_callbacks.pop(session_id, None)
        self._last_push_time.pop(session_id, None)

    async def push_status(self, session_id: str) -> None:
        """Push current run status to the registered WebSocket callback."""
        callback = self._ws_callbacks.get(session_id)
        if not callback:
            return

        session = self._get_session(session_id)
        if not session:
            return

        # Throttle (skip if graph is finished — must deliver final status)
        now = time.time()
        last = self._last_push_time.get(session_id, 0)
        if not session.is_finished and (now - last) * 1000 < self._push_interval_ms:
            return
        self._last_push_time[session_id] = now

        elapsed = 0.0
        if session.start_time:
            elapsed = (datetime.now() - session.start_time).total_seconds()

        all_steps = session.steps + list(session._active_steps.values())
        total_input = sum(
            c.prompt_tokens for s in all_steps for c in s.llm_calls
        )
        total_output = sum(
            c.completion_tokens for s in all_steps for c in s.llm_calls
        )
        total_llm_calls = sum(len(s.llm_calls) for s in all_steps)

        completed = [s.step_name for s in session.steps]
        running = [s.step_name for s in session._active_steps.values()]

        step_details = [
            {
                "name": s.step_name,
                "status": s.status,
                "duration_ms": max(1, s.duration_ms),  # avoid 0ms
                "start_offset": round(
                    (datetime.fromisoformat(s.start_time) - session.start_time).total_seconds(), 2
                ),
                "end_offset": round(
                    (datetime.fromisoformat(s.end_time) - session.start_time).total_seconds(), 2
                ) if s.end_time else None,
            }
            for s in session.steps
        ]

        # Only report "completed" if save() has been called (graph truly done).
        # Otherwise "running" — even if _active_steps is briefly empty between nodes.
        is_done = session.is_finished
        status_data = {
            "type": "run_status",
            "status": "completed" if is_done else "running",
            "current_step": running[0] if running else None,
            "completed_steps": completed,
            "step_details": step_details,
            "elapsed_seconds": round(elapsed, 2),
            "ttft_seconds": round(session.first_llm_response_time, 2) if session.first_llm_response_time else None,
            "total_tokens": total_input + total_output,
            "prompt_tokens": total_input,
            "completion_tokens": total_output,
            "llm_calls": total_llm_calls,
            "step_count": len(session.steps) + len(session._active_steps),
            "completed_count": len(session.steps),
        }

        try:
            result = callback(status_data)
            if asyncio.iscoroutine(result):
                await result
        except Exception:
            # Don't let push failures break the main flow
            pass

    # ===== Step Recording =====

    def start_step(self, session_id: str, step_name: str, input_summary: str = "") -> Optional[StepRecord]:
        """Start recording a step. Each step is isolated by (session, name)."""
        session = self._get_session(session_id)
        if not session:
            return None

        step = StepRecord(
            step_name=step_name,
            start_time=datetime.now().isoformat(),
            input_summary=input_summary[:500],
        )
        session._active_steps[step_name] = step
        asyncio.create_task(self.push_status(session_id))
        return step

    def end_step(
        self,
        session_id: str,
        step_name: str,
        output_summary: str = "",
        status: str = "success",
        error: str = "",
    ) -> None:
        """End recording a specific step by name."""
        session = self._get_session(session_id)
        if not session:
            return

        step = session._active_steps.pop(step_name, None)
        if not step:
            return

        end_time = datetime.now()
        start = datetime.fromisoformat(step.start_time)
        step.duration_ms = int((end_time - start).total_seconds() * 1000)
        step.end_time = end_time.isoformat()
        step.output_summary = output_summary[:500]
        step.status = status
        step.error = error[:500]
        session.steps.append(step)
        asyncio.create_task(self.push_status(session_id))

    def log_llm_call(self, session_id: str, model: str, prompt_tokens: int, completion_tokens: int) -> None:
        """Record an LLM API call."""
        step_name = get_current_step_name()
        if not step_name:
            return

        session = self._get_session(session_id)
        if not session:
            return

        step = session._active_steps.get(step_name)
        if step:
            step.llm_calls.append(
                LLMCallRecord(
                    model=model,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    step_name=step_name,
                    timestamp=datetime.now().isoformat(),
                )
            )

        # Track TTFT on first LLM call
        if session.first_llm_response_time is None and session.start_time:
            session.first_llm_response_time = (datetime.now() - session.start_time).total_seconds()

        asyncio.create_task(self.push_status(session_id))

    def log_search_result(self, query: str, results: list[str]) -> None:
        """Record web search query and results."""
        step_name = get_current_step_name()
        if not step_name:
            return

        # Search results are logged to the current session if any
        # We find the session that has this step active
        for session in self._sessions.values():
            step = session._active_steps.get(step_name)
            if step:
                step.search_results.append(
                    SearchResultRecord(
                        step_name=step_name,
                        query=query[:200],
                        result_count=len(results),
                        result_titles=results[:10],
                        timestamp=datetime.now().isoformat(),
                    )
                )
                break

    def log_reasoning(self, reasoning: str) -> None:
        """Record planner/agent reasoning / CoT."""
        step_name = get_current_step_name()
        if not step_name:
            return

        for session in self._sessions.values():
            step = session._active_steps.get(step_name)
            if step:
                step.reasoning = ReasoningRecord(
                    step_name=step_name,
                    reasoning=reasoning[:3000],
                    timestamp=datetime.now().isoformat(),
                )
                break

    # ===== Persistence =====

    def save(self, session_id: str, final_response: str = "", status: str = "success") -> str:
        """Save the complete log to file (JSON + Markdown). Returns the log file path."""
        session = self._get_session(session_id)
        if not session:
            return ""

        session.is_finished = True

        # Auto-end any still-active steps
        for name in list(session._active_steps.keys()):
            self.end_step(session_id, name, output_summary="[auto-ended by save]")

        end_time = datetime.now()
        total_duration_ms = int(
            (end_time - session.start_time).total_seconds() * 1000
        )

        total_input = sum(
            c.prompt_tokens for s in session.steps for c in s.llm_calls
        )
        total_output = sum(
            c.completion_tokens for s in session.steps for c in s.llm_calls
        )
        total_search_queries = sum(
            len(s.search_results) for s in session.steps
        )

        log_data = {
            "session_id": session.session_id,
            "user_input": session.user_input,
            "start_time": session.start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "total_duration_ms": total_duration_ms,
            "total_duration_formatted": self._fmt_dur(total_duration_ms),
            "total_tokens": {
                "input": total_input,
                "output": total_output,
                "total": total_input + total_output,
            },
            "llm_call_count": sum(len(s.llm_calls) for s in session.steps),
            "search_query_count": total_search_queries,
            "steps": [
                {
                    "step_name": s.step_name,
                    "start_time": s.start_time,
                    "end_time": s.end_time,
                    "duration_ms": s.duration_ms,
                    "duration_formatted": self._fmt_dur(s.duration_ms),
                    "input_summary": s.input_summary,
                    "output_summary": s.output_summary,
                    "status": s.status,
                    "error": s.error,
                    "llm_calls": [
                        {
                            "model": c.model,
                            "prompt_tokens": c.prompt_tokens,
                            "completion_tokens": c.completion_tokens,
                            "total_tokens": c.prompt_tokens + c.completion_tokens,
                            "timestamp": c.timestamp,
                        }
                        for c in s.llm_calls
                    ],
                    "search_results": [
                        {
                            "query": sr.query,
                            "result_count": sr.result_count,
                            "result_titles": sr.result_titles,
                            "timestamp": sr.timestamp,
                        }
                        for sr in s.search_results
                    ],
                    "reasoning": {
                        "text": s.reasoning.reasoning,
                        "timestamp": s.reasoning.timestamp,
                    } if s.reasoning else None,
                }
                for s in session.steps
            ],
            "final_response": final_response[:1000],
            "status": status,
        }

        date_str = session.start_time.strftime("%Y-%m-%d")
        log_dir = os.path.join("logs", "thoughts", date_str)
        os.makedirs(log_dir, exist_ok=True)

        ts = session.start_time.strftime("%H%M%S")
        safe_session = session.session_id.replace("/", "_").replace("\\", "_")[:50]
        filename = f"{ts}_{safe_session}.json"
        log_file = os.path.join(log_dir, filename)

        with open(log_file, "w", encoding="utf-8") as f:
            json.dump(log_data, f, ensure_ascii=False, indent=2)

        # Also write Markdown log
        self._write_md_log(
            session=session,
            end_time=end_time,
            total_duration_ms=total_duration_ms,
            total_input=total_input,
            total_output=total_output,
            final_response=final_response,
            status=status,
        )

        return log_file

    def _write_md_log(
        self,
        session: SessionData,
        end_time: datetime,
        total_duration_ms: int,
        total_input: int,
        total_output: int,
        final_response: str,
        status: str,
    ) -> str:
        """Write a human-readable Markdown log file."""
        date_str = session.start_time.strftime("%Y-%m-%d")
        log_dir = os.path.join("logs", "runs", date_str)
        os.makedirs(log_dir, exist_ok=True)

        ts = session.start_time.strftime("%H%M%S")
        safe_session = session.session_id.replace("/", "_").replace("\\", "_")[:50]
        filename = f"{ts}_{safe_session}.md"
        log_file = os.path.join(log_dir, filename)

        lines: list[str] = []
        lines.append(f"# 运行日志 — {session.session_id}")
        lines.append("")
        lines.append(f"**时间**: {session.start_time.strftime('%Y-%m-%d %H:%M:%S')}  ")
        lines.append(f"**用户输入**: {session.user_input}  ")
        lines.append(f"**总耗时**: {self._fmt_dur(total_duration_ms)} | **TTFT**: {self._fmt_dur(int((session.first_llm_response_time or 0) * 1000))} | **总 Token**: {total_input + total_output} (prompt: {total_input} + completion: {total_output})  ")
        lines.append(f"**状态**: {status}")
        lines.append("")

        # Steps table
        lines.append(f"## 执行步骤 (共 {len(session.steps)} 步)")
        lines.append("")
        lines.append("| # | 步骤 | 状态 | 耗时 | LLM Calls |")
        lines.append("|---|------|------|------|-----------|")
        for i, s in enumerate(session.steps, 1):
            status_icon = "✅" if s.status == "success" else ("❌" if s.status == "error" else "⏳")
            llm_summary = ""
            if s.llm_calls:
                total = sum(c.prompt_tokens + c.completion_tokens for c in s.llm_calls)
                llm_summary = f"{len(s.llm_calls)} ({total})"
            else:
                llm_summary = "0"
            lines.append(f"| {i} | {s.step_name} | {status_icon} | {self._fmt_dur(s.duration_ms)} | {llm_summary} |")
        lines.append("")

        # LLM calls detail
        all_llm_calls = [c for s in session.steps for c in s.llm_calls]
        if all_llm_calls:
            lines.append("## LLM 调用详情")
            lines.append("")
            lines.append("| 模型 | 所属步骤 | Prompt | Completion | Total |")
            lines.append("|------|----------|--------|------------|-------|")
            for c in all_llm_calls:
                total = c.prompt_tokens + c.completion_tokens
                lines.append(f"| {c.model} | {c.step_name} | {c.prompt_tokens} | {c.completion_tokens} | {total} |")
            lines.append("")

        # Search results
        all_searches = [sr for s in session.steps for sr in s.search_results]
        if all_searches:
            lines.append("## 搜索查询")
            lines.append("")
            for sr in all_searches:
                lines.append(f"- **{sr.step_name}**: `{sr.query}` → {sr.result_count} 条结果")
            lines.append("")

        # Reasoning
        all_reasoning = [s.reasoning for s in session.steps if s.reasoning]
        if all_reasoning:
            lines.append("## 推理过程")
            lines.append("")
            for r in all_reasoning:
                lines.append(f"### {r.step_name}")
                lines.append(r.reasoning)
                lines.append("")

        # Final response
        lines.append("## 最终结果")
        lines.append("")
        lines.append(final_response[:2000] if final_response else "（无回复）")
        lines.append("")

        with open(log_file, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        return log_file

    @staticmethod
    def _fmt_dur(ms: int) -> str:
        if ms < 1000:
            return f"{ms}ms"
        s = ms // 1000
        if s < 60:
            return f"{s}.{ms % 1000 // 100}s"
        m = s // 60
        s = s % 60
        return f"{m}m{s}s"


# Global singleton
thought_logger = ThoughtLogger()


# ===== Decorator for Graph Nodes =====

from core.state import ItineraryState


def log_step(step_name: str):
    """Decorator to auto-log a graph node execution.

    Each decorated node gets its own step context via contextvars,
    so parallel nodes don't overwrite each other.
    """

    def decorator(func):
        async def wrapper(state: ItineraryState) -> dict:
            logger = thought_logger
            session_id = state.get("session_id", "unknown")

            # Auto-start session on first step if not exists
            if session_id not in logger._sessions:
                logger.start_session(
                    session_id=session_id,
                    user_input=state.get("user_input", "")[:300],
                )

            # Build input summary
            if step_name == "intent_node":
                input_summary = f"user_input: {state.get('user_input', '')[:200]}"
            elif step_name == "poi_search_node":
                profile = state.get("user_profile", {})
                input_summary = f"city={profile.get('destination')}, interests={profile.get('interests', [])}"
            elif step_name == "context_enrichment_node":
                profile = state.get("user_profile", {})
                input_summary = f"city={profile.get('destination')}, days={profile.get('travel_days')}"
            elif step_name == "planner_node":
                pois = state.get("candidate_pois", [])
                input_summary = f"pois={len(pois)}, days={state.get('user_profile', {}).get('travel_days')}"
            elif step_name == "proposal_node":
                itinerary = state.get("current_itinerary", [])
                input_summary = f"itinerary_days={len(itinerary)}"
            elif step_name == "weather_node":
                profile = state.get("user_profile", {})
                input_summary = f"city={profile.get('destination')}, dates={profile.get('travel_dates')}"
            else:
                input_summary = f"state_keys={list(state.keys())}"

            logger.start_step(session_id, step_name, input_summary=input_summary)
            set_current_step_name(step_name)
            set_current_session_id(session_id)

            try:
                result = await func(state)

                # Build output summary
                if isinstance(result, dict):
                    if "assistant_response" in result:
                        output_summary = result["assistant_response"][:200]
                    elif "intent" in result:
                        output_summary = f"intent={result.get('intent')}, confidence={result.get('intent_confidence', 0):.2f}"
                    elif "current_itinerary" in result:
                        days = result.get("current_itinerary", [])
                        output_summary = f"planned {len(days)} days"
                    elif "candidate_pois" in result:
                        pois = result.get("candidate_pois", [])
                        output_summary = f"found {len(pois)} POIs"
                    elif "travel_context" in result:
                        ctx = result.get("travel_context", {})
                        events = len(ctx.get("upcoming_events", []))
                        foods = len(ctx.get("food_specialties", []))
                        output_summary = f"events={events}, foods={foods}, tips={len(ctx.get('pitfall_tips', []))}"
                    elif "weather_data" in result:
                        wd = result.get("weather_data", [])
                        output_summary = f"weather_days={len(wd)}"
                    else:
                        output_summary = f"keys={list(result.keys())}"
                else:
                    output_summary = str(result)[:200]

                logger.end_step(session_id, step_name, output_summary=output_summary)
                return result

            except Exception as e:
                logger.end_step(session_id, step_name, status="error", error=str(e))
                raise

            finally:
                set_current_step_name(None)
                set_current_session_id("")

        return wrapper

    return decorator

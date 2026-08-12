"""Local LangSmith trace saver — writes every graph execution trace to disk.

LangSmith sends traces to cloud by default. This module adds a local
copy so you can inspect traces without the LangSmith UI.

Traces are saved as JSON to: logs/langsmith/{YYYY-MM-DD}/{HHMMSS}_{session_id}.json
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Base directory for local traces
LOCAL_TRACE_DIR = "logs/langsmith"


def save_local_trace(
    session_id: str,
    inputs: dict[str, Any],
    outputs: dict[str, Any],
    *,
    metadata: Optional[dict[str, Any]] = None,
    duration_ms: Optional[int] = None,
    status: str = "success",
    error: Optional[str] = None,
) -> str:
    """Save a complete graph execution trace to a local JSON file.

    Args:
        session_id: Session identifier (used in filename)
        inputs: Full input state to the graph
        outputs: Full output state from the graph
        metadata: Optional metadata (langsmith project, thread_id, etc.)
        duration_ms: Total execution duration in milliseconds
        status: "success" / "error" / "cancelled"
        error: Error message if status is "error"

    Returns:
        Absolute path to the saved trace file
    """
    now = datetime.now()
    date_dir = now.strftime("%Y-%m-%d")
    ts = now.strftime("%H%M%S")
    safe_sid = session_id.replace("/", "_")[:50]

    trace_dir = Path(LOCAL_TRACE_DIR) / date_dir
    trace_dir.mkdir(parents=True, exist_ok=True)

    filename = "{}_{}.json".format(ts, safe_sid)
    filepath = trace_dir / filename

    # Build a clean trace record
    trace = {
        "session_id": session_id,
        "timestamp": now.isoformat(),
        "status": status,
        "duration_ms": duration_ms,
        "metadata": metadata or {},
        "inputs": _sanitize_for_serialization(inputs),
        "outputs": _sanitize_for_serialization(outputs),
    }

    if error:
        trace["error"] = error

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(trace, f, ensure_ascii=False, indent=2, default=str)

    logger.debug("Local trace saved: %s (%d bytes)", filepath, filepath.stat().st_size)
    return str(filepath.absolute())


def save_local_trace_lightweight(
    session_id: str,
    user_input: str,
    itinerary: Optional[list[dict]] = None,
    *,
    duration_ms: Optional[int] = None,
    status: str = "success",
    warnings: Optional[list[str]] = None,
    error: Optional[str] = None,
) -> str:
    """Save a lightweight trace — just the essentials for debugging.

    Use this for quick inspection without the full graph state dump.
    """
    now = datetime.now()
    date_dir = now.strftime("%Y-%m-%d")
    ts = now.strftime("%H%M%S")
    safe_sid = session_id.replace("/", "_")[:50]

    trace_dir = Path(LOCAL_TRACE_DIR) / date_dir
    trace_dir.mkdir(parents=True, exist_ok=True)

    filename = "{}_{}_light.json".format(ts, safe_sid)
    filepath = trace_dir / filename

    trace = {
        "session_id": session_id,
        "timestamp": now.isoformat(),
        "status": status,
        "duration_ms": duration_ms,
        "user_input": user_input,
        "itinerary_days": len(itinerary) if itinerary else 0,
        "itinerary": itinerary,
        "warnings": warnings or [],
    }

    if error:
        trace["error"] = error

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(trace, f, ensure_ascii=False, indent=2, default=str)

    return str(filepath.absolute())


def list_local_traces(limit: int = 20) -> list[dict]:
    """List recent local traces."""
    trace_dir = Path(LOCAL_TRACE_DIR)
    if not trace_dir.exists():
        return []

    files = sorted(trace_dir.rglob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    results = []
    for fp in files[:limit]:
        stat = fp.stat()
        results.append(
            {
                "file": str(fp),
                "size_kb": round(stat.st_size / 1024, 1),
                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            }
        )
    return results


def _sanitize_for_serialization(obj: Any, max_depth: int = 5) -> Any:
    """Remove non-serializable objects and truncate large payloads."""
    if max_depth <= 0:
        return "<max depth reached>"

    if isinstance(obj, dict):
        result = {}
        for k, v in obj.items():
            # Skip internal/langgraph state keys that are noise
            if k.startswith("__"):
                continue
            result[k] = _sanitize_for_serialization(v, max_depth - 1)
        return result

    if isinstance(obj, list):
        # Truncate large lists
        if len(obj) > 100:
            return [_sanitize_for_serialization(x, max_depth - 1) for x in obj[:100]] + [
                "... {} more items".format(len(obj) - 100)
            ]
        return [_sanitize_for_serialization(x, max_depth - 1) for x in obj]

    if isinstance(obj, (str, int, float, bool, type(None))):
        return obj

    return str(obj)[:500]

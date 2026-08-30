"""Direct graph smoke test."""

import asyncio
import sys

sys.path.insert(0, "src")

from graph.runner import stream_graph_events


async def main():
    events = []
    async for event in stream_graph_events(
        session_id="direct-smoke-1",
        user_id="user-1",
        user_input="目的地北京，玩3天",
        messages=[],
    ):
        print("EVENT:", event)
        events.append(event)
        if event.get("type") in ("final", "error", "clarify"):
            break
    print(f"total events: {len(events)}")


if __name__ == "__main__":
    asyncio.run(main())

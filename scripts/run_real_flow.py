#!/usr/bin/env python3
"""TravelAgent2 真实模型流程运行脚本.

通过 WebSocket 发送真实旅行请求，等待后端调用 LLM/工具完成完整规划流程，
并输出最终行程提案。
"""

from __future__ import annotations

import json
import queue
import sys
import time
import uuid

import websocket

WS_URL = "ws://localhost:8000/ws/chat/{session_id}"


def run_real_flow(user_input: str, timeout: float = 300.0) -> dict:
    session_id = f"real-{uuid.uuid4().hex[:8]}"
    uri = WS_URL.format(session_id=session_id)
    msg_queue: queue.Queue[str] = queue.Queue()
    job_id: str | None = None
    completed = False
    received: list[dict] = []

    def on_message(_ws, message):
        msg_queue.put(message)

    def on_error(_ws, error):
        print(f"[REAL] WS error: {error}", file=sys.stderr)

    ws = websocket.WebSocketApp(
        uri,
        on_message=on_message,
        on_error=on_error,
    )

    import threading

    wst = threading.Thread(target=ws.run_forever, daemon=True)
    wst.start()

    # Wait for open
    for _ in range(50):
        if ws.sock and ws.sock.connected:
            break
        time.sleep(0.1)
    else:
        print("[REAL] FAIL: websocket did not open", file=sys.stderr)
        ws.close()
        return {"session_id": session_id, "job_id": None, "messages": []}

    print(f"[REAL] session_id={session_id}")
    print(f"[REAL] user_input={user_input!r}")
    ws.send(json.dumps({"type": "chat", "content": user_input, "user_id": "real-user"}))

    deadline = time.time() + timeout
    while time.time() < deadline and not completed:
        try:
            raw = msg_queue.get(timeout=2)
        except queue.Empty:
            continue

        msg = json.loads(raw)
        received.append(msg)
        msg_type = msg.get("type")

        if msg_type == "job_created":
            job_id = msg.get("job_id")
            print(f"[REAL] job_created: {job_id}")
        elif msg_type == "stage":
            stage = msg.get("stage")
            event_type = msg.get("event_type")
            print(f"[REAL] stage: {stage} ({event_type})")
            if stage in ("completed", "failed", "cancelled"):
                completed = True
                break
        elif msg_type == "needs_clarification":
            print(f"[REAL] needs_clarification: {msg.get('questions', msg.get('reason'))}")
            completed = True
            break
        elif msg_type == "error":
            print(f"[REAL] error: {msg}", file=sys.stderr)
            completed = True
            break

    ws.close()
    wst.join(timeout=5)

    return {
        "session_id": session_id,
        "job_id": job_id,
        "messages": received,
        "completed": completed,
    }


def extract_proposal(result: dict) -> tuple[str | None, dict | None]:
    for msg in result["messages"]:
        if msg.get("type") == "stage" and msg.get("stage") == "completed":
            payload = msg.get("payload", {})
            return payload.get("proposal_text"), payload
    return None, None


def main() -> int:
    user_input = sys.argv[1] if len(sys.argv) > 1 else "我想去北京玩3天"
    result = run_real_flow(user_input)

    if not result["job_id"]:
        print("[REAL] FAIL: did not receive job_created", file=sys.stderr)
        return 1

    print(f"\n[REAL] job_id={result['job_id']}")
    print(f"[REAL] completed={result['completed']}")

    proposal, payload = extract_proposal(result)
    if proposal:
        print("\n" + "=" * 60)
        print("最终行程提案")
        print("=" * 60)
        print(proposal)
        print("=" * 60)
    elif payload:
        print("\n[REAL] 未找到 proposal_text，最终 payload:")
        print(json.dumps(payload, ensure_ascii=False, indent=2)[:2000])
    else:
        print("\n[REAL] 未收到 completed 事件，完整消息:")
        for m in result["messages"]:
            print(json.dumps(m, ensure_ascii=False, indent=2)[:500])

    return 0


if __name__ == "__main__":
    sys.exit(main())

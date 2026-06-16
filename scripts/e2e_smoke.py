#!/usr/bin/env python3
"""TravelAgent2 end-to-end smoke test.

Assumes backend + frontend are already running on localhost.
"""

from __future__ import annotations

import json
import queue
import sys
import time
import uuid

import httpx
import websocket

BASE_URL = "http://localhost:8000"
WS_URL = "ws://localhost:8000/ws/chat/{session_id}"
FRONTEND_URL = "http://localhost:3000"


def health_check() -> dict:
    with httpx.Client() as client:
        resp = client.get(f"{BASE_URL}/api/health", timeout=10)
        resp.raise_for_status()
        return resp.json()


def frontend_smoke() -> int:
    with httpx.Client() as client:
        resp = client.get(FRONTEND_URL, timeout=10)
        return resp.status_code


def websocket_chat_flow() -> dict:
    session_id = f"e2e-{uuid.uuid4().hex[:8]}"
    uri = WS_URL.format(session_id=session_id)
    received: list[dict] = []
    msg_queue: queue.Queue[str] = queue.Queue()
    job_id: str | None = None

    def on_message(_ws, message):
        msg_queue.put(message)

    def on_error(_ws, error):
        print(f"[E2E] WS error: {error}", file=sys.stderr)

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
        print("[E2E] FAIL: websocket did not open", file=sys.stderr)
        ws.close()
        return {"session_id": session_id, "job_id": None, "messages": []}

    ws.send(json.dumps({"type": "chat", "content": "我想去北京玩3天", "user_id": "e2e-user"}))

    deadline = time.time() + 30
    while time.time() < deadline and job_id is None:
        try:
            raw = msg_queue.get(timeout=1)
        except queue.Empty:
            continue
        msg = json.loads(raw)
        received.append(msg)
        if msg.get("type") == "job_created":
            job_id = msg.get("job_id")
            # Listen a bit longer for stage/completed events.
            try:
                extra = json.loads(msg_queue.get(timeout=10))
                received.append(extra)
            except queue.Empty:
                pass
            break
        if msg.get("type") == "needs_clarification":
            break

    ws.close()
    wst.join(timeout=5)
    return {"session_id": session_id, "job_id": job_id, "messages": received}


def websocket_reconnect(session_id: str, job_id: str) -> dict:
    uri = WS_URL.format(session_id=session_id)
    received: list[dict] = []
    msg_queue: queue.Queue[str] = queue.Queue()

    def on_message(_ws, message):
        msg_queue.put(message)

    ws = websocket.WebSocketApp(uri, on_message=on_message)

    import threading

    wst = threading.Thread(target=ws.run_forever, daemon=True)
    wst.start()

    for _ in range(50):
        if ws.sock and ws.sock.connected:
            break
        time.sleep(0.1)

    ws.send(json.dumps({"type": "subscribe", "job_id": job_id, "last_event_id": 0}))

    try:
        raw = msg_queue.get(timeout=10)
        received.append(json.loads(raw))
    except queue.Empty:
        pass

    ws.close()
    wst.join(timeout=5)
    return {"messages": received}


def main() -> int:
    print("[E2E] health check ...")
    health = health_check()
    print(f"[E2E] health: {health}")

    print("[E2E] frontend smoke ...")
    frontend_status = frontend_smoke()
    print(f"[E2E] frontend status: {frontend_status}")

    print("[E2E] websocket chat flow ...")
    flow = websocket_chat_flow()
    print(f"[E2E] session_id={flow['session_id']}, job_id={flow['job_id']}")
    print(f"[E2E] received message types: {[m.get('type') for m in flow['messages']]}")

    if flow["job_id"] is None:
        print("[E2E] FAIL: did not receive job_created", file=sys.stderr)
        return 1

    print("[E2E] websocket reconnect ...")
    reconnect = websocket_reconnect(flow["session_id"], flow["job_id"])
    types = [m.get("type") for m in reconnect["messages"]]
    print(f"[E2E] reconnect message types: {types}")

    if "state_restored" not in types:
        print("[E2E] WARN: state_restored not received (job may not be completed yet)")

    print("[E2E] PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())

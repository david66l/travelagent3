"""End-to-end smoke test for the graph-backed chat runtime.

Runs against a live backend and verifies that a planning request produces an
assistant message and PDF/Excel artifact URLs.
"""

import json
import sys
import time
import uuid

import requests

BASE = "http://localhost:8000"


def post_message(token: str, fingerprint: str, conversation_id: str, content: str) -> None:
    r = requests.post(
        f"{BASE}/api/v1/chat/message",
        headers={
            "Authorization": f"Bearer {token}",
            "X-Device-Fingerprint": fingerprint,
            "Content-Type": "application/json",
        },
        json={
            "conversation_id": conversation_id,
            "content": content,
            "stream": True,
        },
        timeout=10,
    )
    print(f"[OK] message '{content[:40]}...' status: {r.status_code}")


def stream_events(token: str, fingerprint: str, conversation_id: str, max_time: float = 180.0) -> list[dict]:
    print("[INFO] streaming events...")
    response = requests.get(
        f"{BASE}/api/v1/chat/stream",
        headers={
            "Authorization": f"Bearer {token}",
            "X-Device-Fingerprint": fingerprint,
        },
        params={"conversation_id": conversation_id},
        stream=True,
        timeout=max_time + 10,
    )
    response.raise_for_status()

    events: list[dict] = []
    start = time.time()
    for line in response.iter_lines():
        if not line:
            continue
        line = line.decode("utf-8")
        if not line.startswith("data: "):
            continue
        payload = line[6:]
        if payload == "[DONE]":
            break
        try:
            event = json.loads(payload)
        except json.JSONDecodeError:
            continue
        events.append(event)
        print(f"[EVENT] type={event.get('type')} stage={event.get('stage')}")
        if event.get("type") in ("done", "error"):
            break
        if event.get("type") == "needs_clarification":
            break
        if time.time() - start > max_time:
            print("[ERROR] stream timeout")
            break
    return events


def extract_result(events: list[dict]) -> tuple[str, dict]:
    content = ""
    urls: dict = {}
    for e in events:
        if e.get("type") == "message" and e.get("role") == "assistant":
            content = e.get("content", "")
            urls = {
                "pdf": e.get("output_pdf_url"),
                "excel": e.get("output_excel_url"),
                "map": e.get("output_map_url"),
            }
        elif e.get("type") == "done":
            urls = {
                "pdf": e.get("output_pdf_url", urls.get("pdf")),
                "excel": e.get("output_excel_url", urls.get("excel")),
                "map": e.get("output_map_url", urls.get("map")),
            }
    return content, urls


def main() -> int:
    fingerprint = str(uuid.uuid4())
    # 1. Guest auth
    r = requests.post(
        f"{BASE}/api/v1/auth/guest",
        json={"device_fingerprint": fingerprint},
        timeout=10,
    )
    r.raise_for_status()
    token = r.json()["data"]["access_token"]
    print(f"[OK] guest token: {token[:20]}...")

    # 2. Create conversation
    r = requests.post(
        f"{BASE}/api/v1/conversations",
        headers={
            "Authorization": f"Bearer {token}",
            "X-Device-Fingerprint": fingerprint,
            "Content-Type": "application/json",
        },
        json={"title": "smoke"},
        timeout=10,
    )
    r.raise_for_status()
    conversation_id = r.json()["data"]["id"]
    print(f"[OK] conversation: {conversation_id}")

    turns = [
        "北京3天，预算5000元",
        "目的地北京，玩3天",
        "从上海出发，2026年7月1日出发，一个人，喜欢历史文化",
    ]

    all_events: list[dict] = []
    for i, content in enumerate(turns):
        post_message(token, fingerprint, conversation_id, content)
        events = stream_events(token, fingerprint, conversation_id, max_time=180.0)
        all_events.extend(events)
        if any(e.get("type") == "message" and e.get("role") == "assistant" for e in events):
            break
        if not any(e.get("type") == "needs_clarification" for e in events):
            print("[WARN] no clarification and no assistant message; stopping")
            break
        questions = next(
            (e.get("questions", []) for e in events if e.get("type") == "needs_clarification"),
            [],
        )
        print(f"[INFO] turn {i+1} clarification requested: {questions}")

    content, urls = extract_result(all_events)

    print(f"[INFO] assistant content length: {len(content)}")
    print(f"[INFO] assistant content: {content[:500]}")
    print(f"[INFO] output_urls: {urls}")

    if not content:
        print("[FAIL] no assistant message")
        return 1

    if not (urls.get("pdf") or urls.get("excel")):
        print("[FAIL] no PDF/Excel artifact URLs")
        return 1

    print("[PASS] graph runtime returned assistant message and artifact URLs")
    for name, url in urls.items():
        if not url:
            continue
        # Use GET with streaming so we only check the response status without
        # downloading the whole file (download endpoints do not support HEAD).
        with requests.get(url, timeout=10, stream=True) as resp:
            print(f"[OK] {name} url get status: {resp.status_code} ({url})")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""TravelAgent2 端到端冒烟测试（SSE 主路径）。

前置：backend (8000) + frontend (3000) 已启动。
用法：python scripts/e2e_smoke.py
"""

from __future__ import annotations

import json
import os
import sys
import time
import uuid
from datetime import date, timedelta

import httpx

BASE_URL = os.environ.get("E2E_BASE_URL", "http://127.0.0.1:8000")
FRONTEND_URL = os.environ.get("E2E_FRONTEND_URL", "http://127.0.0.1:3000")
# Cursor/shell 常带 HTTP 代理，直连本机需绕过
HTTPX_KWARGS = {"trust_env": False}


def health_check() -> dict:
    with httpx.Client(**HTTPX_KWARGS) as client:
        resp = client.get(f"{BASE_URL}/api/health", timeout=10)
        resp.raise_for_status()
        return resp.json()


def metrics_check() -> str:
    with httpx.Client(**HTTPX_KWARGS) as client:
        resp = client.get(f"{BASE_URL}/api/v1/metrics", timeout=10)
        resp.raise_for_status()
        return resp.text


def frontend_smoke() -> int:
    with httpx.Client(**HTTPX_KWARGS) as client:
        resp = client.get(FRONTEND_URL, timeout=10)
        return resp.status_code


def _future_trip_query() -> str:
    """Build a complete request that remains valid when the test is run later."""
    start_date = date.today() + timedelta(days=30)
    end_date = start_date + timedelta(days=2)
    return (
        f"我和朋友两个人从上海出发，{start_date:%Y年%m月%d日}到"
        f"{end_date:%Y年%m月%d日}去北京玩3天，不带老人和小孩，"
        "总预算5000元，喜欢历史文化和美食"
    )


def _collect_events(
    client: httpx.Client,
    *,
    conversation_id: str,
    headers: dict,
    timeout: int = 120,
    stop_types: set[str] | None = None,
) -> list[dict]:
    received: list[dict] = []
    terminal_types = stop_types or {"done", "error"}
    with client.stream(
        "GET",
        f"{BASE_URL}/api/v1/chat/stream",
        params={"conversation_id": conversation_id, "timeout": timeout},
        headers=headers,
        timeout=timeout + 10,
    ) as stream:
        deadline = time.time() + timeout
        for line in stream.iter_lines():
            if time.time() > deadline:
                break
            if not line or not line.startswith("data:"):
                continue
            payload = json.loads(line[5:].strip())
            received.append(payload)
            if payload.get("type") in terminal_types:
                break
    return received


def _itinerary_from_events(events: list[dict]) -> list[dict] | None:
    itinerary = None
    for event in events:
        payload = event.get("payload") if isinstance(event.get("payload"), dict) else event
        candidate = payload.get("itinerary") or payload.get("itinerary_final")
        if isinstance(candidate, list) and candidate:
            itinerary = candidate
    return itinerary


def sse_chat_flow(*, content: str | None = None) -> dict:
    fp = str(uuid.uuid4())
    with httpx.Client(**HTTPX_KWARGS) as client:
        guest = client.post(
            f"{BASE_URL}/api/v1/auth/guest",
            json={"device_fingerprint": fp},
            timeout=10,
        )
        guest.raise_for_status()
        token = guest.json()["data"]["access_token"]
        headers = {
            "Authorization": f"Bearer {token}",
            "X-Device-Fingerprint": fp,
        }

        conv = client.post(
            f"{BASE_URL}/api/v1/conversations",
            json={"title": "e2e"},
            headers=headers,
            timeout=10,
        )
        conv.raise_for_status()
        conversation_id = conv.json()["data"]["id"]

        msg = client.post(
            f"{BASE_URL}/api/v1/chat/message",
            json={
                "conversation_id": conversation_id,
                "content": content or _future_trip_query(),
                "stream": True,
            },
            headers=headers,
            timeout=10,
        )
        msg.raise_for_status()

        started_at = time.perf_counter()
        received = _collect_events(
            client,
            conversation_id=conversation_id,
            headers=headers,
            stop_types={"awaiting_confirm", "needs_clarification", "done", "error"},
        )
        job_id = next(
            (event.get("job_id") for event in received if event.get("type") == "job_created"),
            None,
        )

        return {
            "conversation_id": conversation_id,
            "job_id": job_id,
            "messages": received,
            "headers": headers,
            "itinerary": _itinerary_from_events(received),
            "elapsed_seconds": time.perf_counter() - started_at,
        }


def confirm_draft(flow: dict) -> dict:
    with httpx.Client(**HTTPX_KWARGS) as client:
        response = client.post(
            f"{BASE_URL}/api/v1/chat/message",
            json={
                "conversation_id": flow["conversation_id"],
                "content": "",
                "stream": True,
                "action": "confirm",
            },
            headers=flow["headers"],
            timeout=10,
        )
        response.raise_for_status()
        started_at = time.perf_counter()
        events = _collect_events(
            client,
            conversation_id=flow["conversation_id"],
            headers=flow["headers"],
            timeout=180,
        )
    return {
        "messages": events,
        "itinerary": _itinerary_from_events(events),
        "elapsed_seconds": time.perf_counter() - started_at,
    }


def clarification_flow() -> dict:
    """Verify that an incomplete request asks for missing slots, not a plan."""
    flow = sse_chat_flow(content="我想去北京看看历史文化景点")
    message_types = [message.get("type") for message in flow["messages"]]
    return {
        "passed": "needs_clarification" in message_types and flow["job_id"] is None,
        "message_types": message_types,
    }


def wait_for_job(headers: dict, job_id: str, timeout: int = 180) -> dict:
    deadline = time.time() + timeout
    poll_interval = float(os.environ.get("E2E_JOB_POLL_SEC", "5"))
    with httpx.Client(**HTTPX_KWARGS) as client:
        while time.time() < deadline:
            resp = client.get(
                f"{BASE_URL}/api/v1/planning-jobs/{job_id}",
                headers=headers,
                timeout=15,
            )
            if resp.status_code == 429:
                time.sleep(15)
                continue
            resp.raise_for_status()
            data = resp.json()["data"]
            status = data.get("status")
            if status in ("failed", "cancelled", "force_cancelled"):
                return data
            if status == "completed" and job_has_itinerary_payload(data):
                return data
            time.sleep(poll_interval)
    raise TimeoutError(f"job {job_id} did not finish within {timeout}s")


def list_itineraries(headers: dict, limit: int = 10) -> list[dict]:
    with httpx.Client(**HTTPX_KWARGS) as client:
        resp = client.get(
            f"{BASE_URL}/api/v1/itineraries",
            params={"limit": limit, "offset": 0},
            headers=headers,
            timeout=15,
        )
        resp.raise_for_status()
        return resp.json().get("data") or []


def job_has_itinerary_payload(job: dict) -> bool:
    result = job.get("result") or {}
    for key in ("itinerary_final", "itinerary_draft", "itinerary_enriched"):
        days = result.get(key)
        if isinstance(days, list) and days:
            return True
    if result.get("proposal_text") or result.get("proposal_text_preview"):
        return True
    return False


def main() -> int:
    print("[E2E] health check ...")
    health = health_check()
    print(f"[E2E] health: {health}")

    print("[E2E] metrics ...")
    metrics = metrics_check()
    print(f"[E2E] metrics sample: {metrics.splitlines()[:3]}")

    print("[E2E] frontend smoke ...")
    frontend_status = frontend_smoke()
    print(f"[E2E] frontend status: {frontend_status}")

    print("[E2E] SSE chat flow ...")
    flow = sse_chat_flow()
    print(f"[E2E] conversation_id={flow['conversation_id']}, job_id={flow['job_id']}")
    print(f"[E2E] received message types: {[m.get('type') for m in flow['messages']]}")

    message_types = [message.get("type") for message in flow["messages"]]
    print(f"[E2E] draft latency={flow['elapsed_seconds']:.2f}s")
    if flow["job_id"]:
        print("[E2E] queued planning mode detected; wait for job completion ...")
        job_timeout = int(os.environ.get("E2E_JOB_TIMEOUT", "180"))
        try:
            job = wait_for_job(flow["headers"], flow["job_id"], timeout=job_timeout)
        except TimeoutError as exc:
            print(f"[E2E] FAIL: {exc}", file=sys.stderr)
            return 1
        print(f"[E2E] job status={job.get('status')}, latency_ms={job.get('latency_ms')}")
        if job.get("status") != "completed" or not job_has_itinerary_payload(job):
            print("[E2E] FAIL: queued planning did not produce an itinerary", file=sys.stderr)
            return 1
    else:
        if "awaiting_confirm" not in message_types or not flow["itinerary"]:
            print("[E2E] FAIL: interactive graph did not produce a confirmable draft", file=sys.stderr)
            return 1
        print(f"[E2E] draft itinerary days={len(flow['itinerary'])}")
        print("[E2E] confirm draft ...")
        confirmed = confirm_draft(flow)
        confirmed_types = [message.get("type") for message in confirmed["messages"]]
        print(f"[E2E] confirm message types: {confirmed_types}")
        print(f"[E2E] confirm latency={confirmed['elapsed_seconds']:.2f}s")
        if "done" not in confirmed_types or not confirmed["itinerary"]:
            print("[E2E] FAIL: confirmation did not produce a final itinerary", file=sys.stderr)
            return 1

    itineraries = list_itineraries(flow["headers"])
    print(f"[E2E] itineraries table rows={len(itineraries)} (optional; planning stores payload on job)")

    print("[E2E] incomplete-requirement clarification ...")
    clarification = clarification_flow()
    print(f"[E2E] clarification message types: {clarification['message_types']}")
    if not clarification["passed"]:
        print("[E2E] FAIL: incomplete request did not enter clarification", file=sys.stderr)
        return 1

    print("[E2E] PASS (planning path + clarification path)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

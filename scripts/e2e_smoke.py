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


def sse_chat_flow() -> dict:
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
                "content": "我想3月15日到3月18日去北京玩3天，预算5000元，喜欢历史文化和美食",
                "stream": True,
            },
            headers=headers,
            timeout=10,
        )
        msg.raise_for_status()

        received: list[dict] = []
        job_id: str | None = None
        with client.stream(
            "GET",
            f"{BASE_URL}/api/v1/chat/stream",
            params={"conversation_id": conversation_id, "timeout": 120},
            headers=headers,
            timeout=130,
        ) as stream:
            deadline = time.time() + 45
            for line in stream.iter_lines():
                if time.time() > deadline:
                    break
                if not line or not line.startswith("data:"):
                    continue
                payload = json.loads(line[5:].strip())
                received.append(payload)
                if payload.get("type") == "job_created":
                    job_id = payload.get("job_id")
                if payload.get("type") in ("done", "error"):
                    break

        return {
            "conversation_id": conversation_id,
            "job_id": job_id,
            "messages": received,
            "headers": headers,
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

    if flow["job_id"] is None:
        print("[E2E] FAIL: did not receive job_created", file=sys.stderr)
        return 1

    print("[E2E] wait for planning job completion ...")
    job_timeout = int(os.environ.get("E2E_JOB_TIMEOUT", "180"))
    try:
        job = wait_for_job(flow["headers"], flow["job_id"], timeout=job_timeout)
    except TimeoutError as exc:
        print(f"[E2E] FAIL: {exc}", file=sys.stderr)
        return 1
    print(f"[E2E] job status={job.get('status')}, latency_ms={job.get('latency_ms')}")

    if job.get("status") != "completed":
        print(f"[E2E] FAIL: job ended with status={job.get('status')}", file=sys.stderr)
        if job.get("error_message") or job.get("last_error"):
            print(f"[E2E] error: {job.get('error_message') or job.get('last_error')}", file=sys.stderr)
        return 1

    if not job_has_itinerary_payload(job):
        print("[E2E] FAIL: completed job missing itinerary payload in result", file=sys.stderr)
        return 1

    itineraries = list_itineraries(flow["headers"])
    print(f"[E2E] itineraries table rows={len(itineraries)} (optional; planning stores payload on job)")

    print("[E2E] PASS (full path: chat → job → itinerary payload)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

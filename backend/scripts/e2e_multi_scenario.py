"""Multi-scenario backend end-to-end smoke test."""

from __future__ import annotations

import json
import sys
import time
import uuid
from urllib.parse import urljoin

import requests

BASE_URL = "http://localhost:8000"

SCENARIOS = [
    {
        "name": "成都-火锅历史",
        "query": "成都 4 天，预算 3000 元，喜欢火锅和历史文化",
        "expected_days": 4,
        "min_activities": 6,
    },
    {
        "name": "上海-亲子游",
        "query": "上海 3 天，预算 5000 元，带小孩，喜欢亲子和博物馆",
        "expected_days": 3,
        "min_activities": 4,
    },
    {
        "name": "北京-历史文化",
        "query": "北京 4 天，预算 4000 元，喜欢历史文化和古建筑",
        "expected_days": 4,
        "min_activities": 6,
    },
    {
        "name": "广州-美食",
        "query": "广州 2 天，预算 2000 元，喜欢吃早茶和粤菜",
        "expected_days": 2,
        "min_activities": 3,
    },
    {
        "name": "杭州-西湖自然",
        "query": "杭州 3 天，预算 3500 元，喜欢西湖和自然风光",
        "expected_days": 3,
        "min_activities": 5,
    },
    {
        "name": "西安-历史古迹",
        "query": "西安 4 天，预算 5000 元，喜欢历史古迹和博物馆",
        "expected_days": 4,
        "min_activities": 5,
    },
    {
        "name": "西安-7天深度",
        "query": "西安 7 天，预算 8000 元，喜欢历史古迹和博物馆",
        "expected_days": 7,
        "min_activities": 10,
    },
]


def post(path: str, *, json_body=None, headers=None, timeout=30):
    url = urljoin(BASE_URL, path)
    resp = requests.post(url, json=json_body, headers=headers, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def run_scenario(scenario: dict) -> dict:
    fingerprint = str(uuid.uuid4())
    name = scenario["name"]
    query = scenario["query"]

    # 1. guest auth
    auth_resp = post(
        "/api/v1/auth/guest",
        json_body={"device_fingerprint": fingerprint},
    )
    token = auth_resp["data"]["access_token"]
    headers = {
        "Authorization": f"Bearer {token}",
        "X-Device-Fingerprint": fingerprint,
        "Content-Type": "application/json",
    }

    # 2. create conversation
    conv_resp = post("/api/v1/conversations", json_body={"title": f"e2e-{name}"}, headers=headers)
    conversation_id = conv_resp["data"]["id"]

    # 3. send chat message
    post(
        "/api/v1/chat/message",
        json_body={
            "conversation_id": conversation_id,
            "content": query,
            "stream": True,
        },
        headers=headers,
    )

    # 4. connect SSE stream
    stream_url = f"{BASE_URL}/api/v1/chat/stream?conversation_id={conversation_id}"
    itinerary = None
    output_pdf_url = None
    output_excel_url = None

    start = time.time()
    with requests.get(stream_url, headers=headers, stream=True, timeout=180) as r:
        r.raise_for_status()
        for line in r.iter_lines(decode_unicode=True):
            if not line:
                continue
            if line.startswith("data:"):
                data = line[5:].strip()
                if not data:
                    continue
                try:
                    event = json.loads(data)
                except json.JSONDecodeError:
                    continue

                etype = event.get("type")
                if etype in ("message", "final"):
                    payload = event.get("payload", event)
                    itinerary = (
                        payload.get("itinerary") or payload.get("itinerary_final") or itinerary
                    )
                    output_pdf_url = payload.get("output_pdf_url") or output_pdf_url
                    output_excel_url = payload.get("output_excel_url") or output_excel_url
                elif etype == "completed":
                    payload = event.get("payload", {})
                    itinerary = (
                        payload.get("itinerary") or payload.get("itinerary_final") or itinerary
                    )
                    output_pdf_url = payload.get("output_pdf_url") or output_pdf_url
                    output_excel_url = payload.get("output_excel_url") or output_excel_url
                elif etype == "done":
                    break

            if time.time() - start > 180:
                break

    elapsed = time.time() - start

    # 5. verify
    if not itinerary:
        return {"name": name, "status": "FAIL", "error": "no itinerary", "elapsed": elapsed}

    days = len(itinerary)
    total_activities = sum(len(day.get("activities", [])) for day in itinerary)
    blank_days = [day.get("day_number") for day in itinerary if not day.get("activities")]

    if days != scenario["expected_days"]:
        return {
            "name": name,
            "status": "FAIL",
            "error": f"expected {scenario['expected_days']} days, got {days}",
            "elapsed": elapsed,
        }

    if total_activities < scenario["min_activities"]:
        return {
            "name": name,
            "status": "FAIL",
            "error": f"expected >= {scenario['min_activities']} activities, got {total_activities}",
            "elapsed": elapsed,
        }

    if blank_days:
        return {
            "name": name,
            "status": "FAIL",
            "error": f"blank days: {blank_days}",
            "elapsed": elapsed,
        }

    # 6. verify exports
    if not output_pdf_url or not output_excel_url:
        return {"name": name, "status": "FAIL", "error": "missing export URLs", "elapsed": elapsed}

    pdf_resp = requests.get(urljoin(BASE_URL, output_pdf_url), headers=headers, timeout=30)
    excel_resp = requests.get(urljoin(BASE_URL, output_excel_url), headers=headers, timeout=30)

    if pdf_resp.status_code != 200 or not pdf_resp.content.startswith(b"%PDF"):
        return {"name": name, "status": "FAIL", "error": "invalid PDF", "elapsed": elapsed}
    if excel_resp.status_code != 200 or len(excel_resp.content) < 100:
        return {"name": name, "status": "FAIL", "error": "invalid Excel", "elapsed": elapsed}

    return {
        "name": name,
        "status": "PASS",
        "days": days,
        "activities": total_activities,
        "blank_days": blank_days,
        "elapsed": elapsed,
        "itinerary": itinerary,
    }


def main():
    results = []
    for scenario in SCENARIOS:
        print(f"\n>>> Running: {scenario['name']} ...")
        try:
            result = run_scenario(scenario)
        except Exception as exc:
            result = {"name": scenario["name"], "status": "FAIL", "error": str(exc), "elapsed": 0}
        results.append(result)

        print(
            f"    {result['status']}: {result.get('days', '-')} days, {result.get('activities', '-')} activities, {result['elapsed']:.1f}s"
        )
        if result.get("itinerary"):
            for day in result["itinerary"]:
                acts = day.get("activities", [])
                print(f"      Day {day.get('day_number')}: {len(acts)} activities")
                for a in acts:
                    print(
                        f"        {a.get('start_time', '')} - {a.get('end_time', '')} {a.get('poi_name', '')}"
                    )
        if result["status"] != "PASS":
            print(f"      error: {result.get('error')}")

    print("\n" + "=" * 60)
    passed = sum(1 for r in results if r["status"] == "PASS")
    failed = len(results) - passed
    print(f"Summary: {passed} passed, {failed} failed / {len(results)} scenarios")
    if failed:
        for r in results:
            if r["status"] != "PASS":
                print(f"  - {r['name']}: {r.get('error')}")
        sys.exit(1)
    print("All scenarios passed.")


if __name__ == "__main__":
    main()

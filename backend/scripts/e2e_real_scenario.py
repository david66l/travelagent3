"""Backend end-to-end smoke test: guest -> conversation -> chat -> itinerary + exports."""

import json
import sys
import time
import uuid
from urllib.parse import urljoin

import requests

BASE_URL = "http://localhost:8000"
QUERY = "成都 4 天，预算 3000 元，喜欢火锅和历史文化"


def post(path: str, *, json_body=None, headers=None):
    url = urljoin(BASE_URL, path)
    resp = requests.post(url, json=json_body, headers=headers, timeout=30)
    resp.raise_for_status()
    return resp.json()


def main():
    fingerprint = str(uuid.uuid4())

    # 1. guest auth
    auth_resp = post(
        "/api/v1/auth/guest",
        json_body={"device_fingerprint": fingerprint},
    )
    token = auth_resp["data"]["access_token"]
    print(f"Guest token: {token[:20]}...")

    headers = {
        "Authorization": f"Bearer {token}",
        "X-Device-Fingerprint": fingerprint,
        "Content-Type": "application/json",
    }

    # 2. create conversation
    conv_resp = post("/api/v1/conversations", json_body={"title": "e2e test"}, headers=headers)
    conversation_id = conv_resp["data"]["id"]
    print(f"Conversation: {conversation_id}")

    # 3. send chat message
    post(
        "/api/v1/chat/message",
        json_body={
            "conversation_id": conversation_id,
            "content": QUERY,
            "stream": True,
        },
        headers=headers,
    )
    print("Message sent, waiting for SSE...")

    # 4. connect SSE stream
    stream_url = f"{BASE_URL}/api/v1/chat/stream?conversation_id={conversation_id}"
    itinerary = None
    output_pdf_url = None
    output_excel_url = None
    final_text = ""

    start = time.time()
    with requests.get(stream_url, headers=headers, stream=True, timeout=120) as r:
        r.raise_for_status()
        buffer = []
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
                if etype == "token":
                    final_text += event.get("chunk", "")
                elif etype == "stage":
                    print(f"  stage: {event.get('stage')} ({event.get('event_id')})")
                elif etype in ("message", "final"):
                    payload = event.get("payload", event)
                    itinerary = payload.get("itinerary") or payload.get("itinerary_final") or itinerary
                    output_pdf_url = payload.get("output_pdf_url") or output_pdf_url
                    output_excel_url = payload.get("output_excel_url") or output_excel_url
                    final_text += payload.get("content", "")
                elif etype == "completed":
                    payload = event.get("payload", {})
                    itinerary = payload.get("itinerary") or payload.get("itinerary_final") or itinerary
                    output_pdf_url = payload.get("output_pdf_url") or output_pdf_url
                    output_excel_url = payload.get("output_excel_url") or output_excel_url
                    final_text += payload.get("proposal_text", "")
                elif etype == "done":
                    break

            if time.time() - start > 120:
                print("Timeout waiting for completion")
                break

    elapsed = time.time() - start
    print(f"Stream finished in {elapsed:.1f}s")

    # 5. verify itinerary
    if not itinerary:
        print("FAIL: no itinerary received")
        sys.exit(1)

    print(f"Itinerary days: {len(itinerary)}")
    if len(itinerary) < 1:
        print("FAIL: itinerary empty")
        sys.exit(1)

    total_activities = sum(len(day.get("activities", [])) for day in itinerary)
    print(f"Total activities: {total_activities}")
    for day in itinerary:
        acts = day.get("activities", [])
        print(f"  Day {day.get('day_number')}: {len(acts)} activities")
        for a in acts:
            print(f"    {a.get('start_time','')} - {a.get('end_time','')} {a.get('poi_name','')}")
    if total_activities == 0:
        print("FAIL: no activities in itinerary")
        sys.exit(1)

    # 6. verify export URLs
    print(f"PDF URL: {output_pdf_url}")
    print(f"Excel URL: {output_excel_url}")

    if not output_pdf_url or not output_excel_url:
        print("FAIL: missing export URLs")
        sys.exit(1)

    pdf_resp = requests.get(urljoin(BASE_URL, output_pdf_url), headers=headers, timeout=30)
    excel_resp = requests.get(urljoin(BASE_URL, output_excel_url), headers=headers, timeout=30)

    print(f"PDF status: {pdf_resp.status_code}, size: {len(pdf_resp.content)}")
    print(f"Excel status: {excel_resp.status_code}, size: {len(excel_resp.content)}")

    if pdf_resp.status_code != 200 or not pdf_resp.content.startswith(b"%PDF"):
        print("FAIL: PDF not valid")
        sys.exit(1)

    if excel_resp.status_code != 200 or len(excel_resp.content) < 100:
        print("FAIL: Excel not valid")
        sys.exit(1)

    print("PASS: end-to-end real scenario succeeded")


if __name__ == "__main__":
    main()

"""End-to-end test through the Go gateway (:8080).

Exercises the full path frontend → gateway → backend → Celery → solver → LLM,
covering every wired feature: edge auth, planning + SSE streaming, in-trip
replan (trip_event), booking REST API, and account/profile/logout.
"""
import json
import sys
import threading
import time
import uuid
import urllib.request
import urllib.error

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

GW = "http://127.0.0.1:8080/api/v1"
FP = f"e2e-{uuid.uuid4()}"
results: list[tuple[str, bool, str]] = []


def rec(name, ok, detail=""):
    results.append((name, ok, detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{(' — ' + detail) if detail else ''}")


def call(method, path, body=None, tok=None, timeout=15):
    h = {"Content-Type": "application/json"}
    if tok:
        h["Authorization"] = "Bearer " + tok
        h["X-Device-Fingerprint"] = FP
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(GW + path, data=data, headers=h, method=method)
    try:
        r = urllib.request.urlopen(req, timeout=timeout)
        return r.status, json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()[:200]


def stream(cid, tok, fire_msg=None, max_seconds=70, fire_before_open=False):
    """Open SSE through the gateway; optionally fire a message; collect events."""
    if fire_msg and fire_before_open:
        call("POST", "/chat/message", fire_msg, tok=tok)
        fire_msg = None
    url = f"{GW}/chat/stream?conversation_id={cid}"
    req = urllib.request.Request(
        url, headers={"Authorization": "Bearer " + tok, "X-Device-Fingerprint": FP,
                      "Accept": "text/event-stream"})
    resp = urllib.request.urlopen(req, timeout=max_seconds + 10)
    ctype = resp.headers.get("Content-Type", "")
    if fire_msg:
        threading.Thread(target=lambda: call("POST", "/chat/message", fire_msg, tok=tok),
                         daemon=True).start()
    events, itinerary, t0 = [], None, time.time()
    while time.time() - t0 < max_seconds:
        line = resp.readline()
        if not line:
            break
        s = line.decode("utf-8", "ignore").strip()
        if not s.startswith("data:"):
            continue
        try:
            d = json.loads(s[5:].strip())
        except Exception:
            continue
        events.append(d)
        pl = d.get("payload") or {}
        it = (
            pl.get("itinerary") or pl.get("itinerary_draft") or pl.get("itinerary_final")
            or d.get("itinerary") or d.get("itinerary_draft") or d.get("itinerary_final")
        )
        if it:
            itinerary = it
        event_type = d.get("type")
        if event_type == "awaiting_confirm" and itinerary:
            break
        if event_type == "message" and d.get("stage") == "completed" and itinerary:
            break
        if event_type == "done":
            break
    resp.close()
    return ctype, events, itinerary


print("=== TravelAgent2 — E2E through Go gateway (:8080) ===\n")

# 1. Gateway health
st, _ = call("GET", "/health")
rec("gateway /health (via :8080)", 200 <= st < 300, f"HTTP {st}")

# 2. Guest auth through gateway (public route)
st, body = call("POST", "/auth/guest", {"device_fingerprint": FP})
tok = body["data"]["access_token"] if 200 <= st < 300 else None
rec("guest auth (public proxy)", bool(tok), f"token len {len(tok) if tok else 0}")

# 3. Edge auth: protected route without token must 401 at the gateway
st, _ = call("POST", "/conversations", {})
rec("edge-auth rejects no-token (401)", st == 401, f"HTTP {st}")

# 4. Create conversation (protected, JWT validated at edge)
st, body = call("POST", "/conversations", {"title": "e2e"}, tok=tok)
cid = body["data"]["id"] if 200 <= st < 300 else None
rec("create conversation (protected proxy)", bool(cid), str(cid)[:8] if cid else "")

# 5. Full planning turn + SSE streaming through gateway
print("\n  → planning (LLM + VRP solver, streamed via gateway, ~30-50s)...")
ctype, events, itinerary = stream(
    cid, tok,
    fire_msg={"conversation_id": cid,
              "content": "我2026年10月1日从北京出发，2人去上海玩5天，预算每人8000元，喜欢美食和历史，请规划行程",
              "stream": True},
    max_seconds=90)
rec("SSE Content-Type text/event-stream", "text/event-stream" in ctype, ctype)
rec("SSE events streamed through gateway", len(events) >= 3, f"{len(events)} events")
n_attr = 0
days = 0
if itinerary:
    days = len(itinerary)
    n_attr = sum(len([a for a in d.get("activities", []) if a.get("category") == "attraction"])
                 for d in itinerary)
rec("planning produced an itinerary", bool(itinerary), f"{days} days, {n_attr} attractions")
rec(
    "draft pauses for explicit confirmation",
    any(e.get("type") == "awaiting_confirm" for e in events),
)
poi_names = [a.get("poi_name") for d in (itinerary or []) for a in d.get("activities", [])
             if a.get("category") == "attraction"]

# 6. Structured draft modification, followed by explicit confirmation.
if itinerary and poi_names:
    removed = poi_names[0]
    print(f"\n  → structured modify: removing '{removed}' ...")
    _, modify_events, modified = stream(
        cid, tok,
        fire_msg={"conversation_id": cid, "content": "", "stream": True,
                  "action": "modify",
                  "change": {"action": "remove", "poi_id": removed}},
        max_seconds=50)
    still = modified and any(a.get("poi_name") == removed
                             for d in modified for a in d.get("activities", []))
    rec("structured modify returns a new draft", bool(modified), f"{len(modify_events)} events")
    rec("structured remove really changes itinerary", bool(modified) and not still)
    itinerary = modified or itinerary
    poi_names = [a.get("poi_name") for d in itinerary for a in d.get("activities", [])
                 if a.get("category") == "attraction"]

print("\n  → confirming draft and running tool/fact-check/booking chain ...")
_, confirm_events, confirmed = stream(
    cid, tok,
    fire_msg={"conversation_id": cid, "content": "", "stream": True,
              "action": "confirm"},
    max_seconds=90)
completed_messages = [
    e for e in confirm_events
    if e.get("type") == "message" and e.get("role") == "assistant"
    and e.get("stage") == "completed"
]
final_payload = completed_messages[-1] if completed_messages else {}
rec("confirm reaches completed final state", final_payload.get("stage") == "completed",
    final_payload.get("stage", "missing"))
rec("confirm includes tool results", bool(final_payload.get("tool_results")))
rec("confirm includes transparent budget breakdown", bool(final_payload.get("budget_breakdown")))
itinerary = confirmed or itinerary
poi_names = [a.get("poi_name") for d in (itinerary or []) for a in d.get("activities", [])
             if a.get("category") == "attraction"]

# 7. In-trip replan (trip_event: closure of a scheduled POI)
if itinerary and poi_names:
    target = poi_names[0]
    print(f"\n  → in-trip replan: closing '{target}' ...")
    ctype2, events2, it2 = stream(
        cid, tok,
        fire_msg={"conversation_id": cid, "content": "", "stream": True,
                  "action": "trip_event",
                  "external_event": {"type": "closure", "poi": target, "detail": "临时闭馆"}},
        max_seconds=40,
        fire_before_open=False)
    still = it2 and any(a.get("poi_name") == target
                        for d in it2 for a in d.get("activities", []))
    rec("trip_event replan returned itinerary", bool(it2), f"{len(events2)} events")
    rec(f"closed POI '{target}' removed from plan", bool(it2) and not still,
        "removed" if (it2 and not still) else "still present/unknown")
    if it2:
        _, reconfirm_events, _ = stream(
            cid, tok,
            fire_msg={"conversation_id": cid, "content": "", "stream": True,
                      "action": "confirm"},
            max_seconds=90)
        rec(
            "replanned draft can be confirmed again",
            any(e.get("type") == "message" and e.get("stage") == "completed"
                for e in reconfirm_events),
        )
else:
    rec("trip_event replan", False, "skipped — no itinerary/POIs")

# 8. Booking REST API through gateway
st, body = call("POST", "/bookings/flights/search", {"origin": "北京", "dest": "上海", "date": "2026-10-10"}, tok=tok)
fl = body.get("data", {}).get("flights", []) if isinstance(body, dict) else []
rec("booking: flights/search", 200 <= st < 300 and len(fl) > 0, f"{len(fl)} flights")
st, body = call("POST", "/bookings/hotels/search", {"city": "上海", "checkin": "2026-10-10", "checkout": "2026-10-15", "guests": 1}, tok=tok)
ht = body.get("data", {}).get("hotels", []) if isinstance(body, dict) else []
rec("booking: hotels/search", 200 <= st < 300 and len(ht) > 0, f"{len(ht)} hotels")
if poi_names:
    st, body = call("POST", "/bookings/attractions/tickets", {"poi_name": poi_names[0], "date": "2026-10-10"}, tok=tok)
    rec("booking: attractions/tickets", 200 <= st < 300, f"HTTP {st}")

# 9. Account / profile / memory / logout
st, me = call("GET", "/users/me", tok=tok)
rec("account: GET /users/me", 200 <= st < 300 and me.get("data", {}).get("role") == "guest",
    me.get("data", {}).get("role") if isinstance(me, dict) else "")
st, _ = call("GET", "/users/me/profile", tok=tok)
rec("account: GET /users/me/profile", 200 <= st < 300, f"HTTP {st}")
st, up = call("PUT", "/users/me/profile", {"preferences": {"pace": "紧凑", "food": "本地特色"}}, tok=tok)
for _ in range(3):
    if st != 429:
        break
    time.sleep(10)
    st, up = call("PUT", "/users/me/profile", {"preferences": {"pace": "紧凑", "food": "本地特色"}}, tok=tok)
saved_prefs = (up.get("data") or {}).get("preferences", {}) if isinstance(up, dict) else {}
rec("account: PUT /users/me/profile persists", 200 <= st < 300 and saved_prefs.get("pace") == "紧凑", f"HTTP {st}")
st, out = call("POST", "/auth/logout", {}, tok=tok)
rec("account: POST /auth/logout", 200 <= st < 300, out.get("message") if isinstance(out, dict) else str(out)[:40])

# Summary
print("\n=== SUMMARY ===")
passed = sum(1 for _, ok, _ in results if ok)
print(f"  {passed}/{len(results)} checks passed")
for name, ok, _ in results:
    if not ok:
        print(f"   ✗ {name}")
print("  RESULT:", "ALL PASS ✓" if passed == len(results) else f"{len(results)-passed} FAILED")
sys.exit(0 if passed == len(results) else 1)

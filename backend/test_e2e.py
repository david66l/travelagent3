"""End-to-end test via WebSocket — job-based planning flow."""

import asyncio
import json
import time
import sys
import uuid

sys.path.insert(0, "src")

import websockets

API_URL = "ws://localhost:8000/ws/chat/"


async def run_scenario(name: str, message: str, timeout: int = 120):
    session_id = f"test_{uuid.uuid4().hex[:8]}"
    url = f"{API_URL}{session_id}"
    start = time.time()
    stages = []
    error = None
    final_response = None
    done = False

    print(f"\n{'='*60}")
    print(f"场景: {name}")
    print(f"输入: {message}")
    print(f"超时: {timeout}s")
    print("=" * 60)

    try:
        async with websockets.connect(url, open_timeout=5, close_timeout=5) as ws:
            await ws.send(json.dumps({"content": message, "user_id": "test_user"}))

            while not done:
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=timeout)
                    data = json.loads(raw)
                    elapsed = time.time() - start
                    msg_type = data.get("type")

                    if msg_type == "job_created":
                        print(f"  [{elapsed:5.1f}s] JOB     | id={data.get('job_id')}")
                    elif msg_type == "needs_clarification":
                        questions = data.get("questions") or []
                        preview = questions[0] if questions else "需要补充信息"
                        print(f"  [{elapsed:5.1f}s] CLARIFY | {preview[:60]}")
                        done = True
                    elif msg_type == "stage" or data.get("stage"):
                        stage = data.get("stage")
                        stages.append({"elapsed": round(elapsed, 1), "stage": stage})
                        print(f"  [{elapsed:5.1f}s] STAGE   | {stage}")
                        if stage in ("completed", "failed", "cancelled"):
                            payload = data.get("payload") or {}
                            if payload.get("proposal_text"):
                                final_response = str(payload["proposal_text"])[:200]
                            done = True
                    elif msg_type == "error":
                        error = data.get("error", "unknown")
                        print(f"  [{elapsed:5.1f}s] ERROR   | {error}")
                        done = True
                    else:
                        print(f"  [{elapsed:5.1f}s] {msg_type or 'unknown'}")
                except asyncio.TimeoutError:
                    error = f"WebSocket recv timeout after {timeout}s"
                    print(f"  [>{timeout}s] TIMEOUT | {error}")
                    done = True
    except Exception as e:
        error = str(e)
        print(f"  [ERR] Connection failed: {error}")

    total = time.time() - start
    print(f"\n  总耗时: {total:.1f}s")
    if error:
        print(f"  结果: ❌ ERROR — {error}")
    elif final_response:
        print(f"  结果: ✅ COMPLETED — {final_response}...")
    else:
        print(f"  结果: {'✅' if not error else '❌'} 完成")
    return {"name": name, "total": total, "error": error, "stages": stages}


async def main():
    print("后端地址:", API_URL.replace("ws://", "http://").replace("/ws/chat/", "/api/health"))
    print("开始端到端测试...")

    results = []

    results.append(
        await run_scenario(
            "上海2天（完整规划）",
            "上海玩2天，预算3000，喜欢美食",
            timeout=180,
        )
    )

    results.append(
        await run_scenario(
            "信息不完整（追问）",
            "我想去旅游",
            timeout=30,
        )
    )

    print(f"\n{'='*60}")
    print("测试汇总")
    print("=" * 60)
    for r in results:
        status = "✅" if not r["error"] else "❌"
        print(
            f"  {status} {r['name']:30s} | {r['total']:5.1f}s | {'OK' if not r['error'] else r['error'][:40]}"
        )


if __name__ == "__main__":
    asyncio.run(main())

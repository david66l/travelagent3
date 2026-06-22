"""End-to-end test via WebSocket — 4 scenarios."""
import asyncio, json, time, sys, uuid
sys.path.insert(0, "src")

import websockets

API_URL = "ws://localhost:8000/ws/chat/"

async def run_scenario(name: str, message: str, timeout: int = 120):
    session_id = f"test_{uuid.uuid4().hex[:8]}"
    url = f"{API_URL}{session_id}"
    start = time.time()
    steps = []
    error = None
    final_response = None

    print(f"\n{'='*60}")
    print(f"场景: {name}")
    print(f"输入: {message}")
    print(f"超时: {timeout}s")
    print("=" * 60)

    try:
        async with websockets.connect(url, open_timeout=5, close_timeout=5) as ws:
            await ws.send(json.dumps({"content": message, "user_id": "test_user"}))

            while True:
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=timeout)
                    data = json.loads(raw)
                    elapsed = time.time() - start

                    if data.get("type") == "run_status":
                        status = data.get("status")
                        current = data.get("current_step")
                        completed = data.get("completed_steps", [])
                        steps.append({
                            "elapsed": round(elapsed, 1),
                            "status": status,
                            "current": current,
                            "completed": completed,
                            "tokens": data.get("total_tokens", 0),
                        })
                        print(f"  [{elapsed:5.1f}s] {status:8s} | step={current or '—':20s} | completed={len(completed)} | tokens={data.get('total_tokens', 0)}")

                        if status == "completed":
                            break
                    elif data.get("type") == "message":
                        final_response = data.get("assistant_message", "")[:200]
                        print(f"  [{elapsed:5.1f}s] MESSAGE | {final_response}...")
                        break
                    elif data.get("type") == "error":
                        error = data.get("error", "unknown")
                        print(f"  [{elapsed:5.1f}s] ERROR   | {error}")
                        break
                except asyncio.TimeoutError:
                    error = f"WebSocket recv timeout after {timeout}s"
                    print(f"  [>{timeout}s] TIMEOUT | {error}")
                    break
    except Exception as e:
        error = str(e)
        print(f"  [ERR] Connection failed: {error}")

    total = time.time() - start
    print(f"\n  总耗时: {total:.1f}s")
    if error:
        print(f"  结果: ❌ ERROR — {error}")
    elif final_response:
        print(f"  结果: ✅ MESSAGE — {final_response}...")
    else:
        print(f"  结果: {'✅' if not error else '❌'} 完成")
    return {"name": name, "total": total, "error": error, "steps": steps}


async def main():
    print("后端地址:", API_URL.replace("ws://", "http://").replace("/ws/chat/", "/api/health"))
    print("开始端到端测试...")

    results = []

    # 场景1: 济南 — 内置 fallback，应该很快
    results.append(await run_scenario(
        "济南5天（内置fallback）",
        "我下周去济南旅游五天",
        timeout=60,
    ))

    # 场景2: 北京 — 另一个内置城市
    results.append(await run_scenario(
        "北京3天（内置fallback）",
        "北京3天，预算5000，喜欢历史文化",
        timeout=60,
    ))

    # 场景3: 信息不完整 — 应该追问
    results.append(await run_scenario(
        "信息不完整（追问）",
        "我想去旅游",
        timeout=30,
    ))

    # 场景4: 虚构城市 — 验证超时兜底
    results.append(await run_scenario(
        "虚构城市（超时兜底）",
        "我要去火星城玩3天",
        timeout=30,
    ))

    # 汇总
    print(f"\n{'='*60}")
    print("测试汇总")
    print("=" * 60)
    for r in results:
        status = "✅" if not r["error"] else "❌"
        print(f"  {status} {r['name']:30s} | {r['total']:5.1f}s | {'OK' if not r['error'] else r['error'][:40]}")


if __name__ == "__main__":
    asyncio.run(main())

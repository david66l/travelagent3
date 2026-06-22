"""Real end-to-end test — 10 scenarios, no cache, full graph execution."""
import asyncio, sys, time
sys.path.insert(0, "src")

from graph.graph import build_graph
from core.state import ItineraryState
from core.checkpointer import create_checkpointer


async def run_scenario(name: str, message: str, timeout: int = 120):
    print(f"\n{'='*70}")
    print(f"场景: {name}")
    print(f"输入: {message}")
    print(f"超时: {timeout}s")
    print("=" * 70)

    checkpointer = await create_checkpointer()
    graph = build_graph().compile(checkpointer=checkpointer)
    config = {"configurable": {"thread_id": f"test_{int(time.time()*1000)}"}}
    state = ItineraryState(
        session_id=f"test_{int(time.time())}",
        user_input=message,
        messages=[{"role": "user", "content": message}],
    )

    start = time.time()
    try:
        result = await asyncio.wait_for(graph.ainvoke(state, config=config), timeout=timeout)
        elapsed = time.time() - start

        intent = result.get("intent", "unknown")
        clarification = result.get("needs_clarification", False)
        confirmation = result.get("waiting_for_confirmation", False)
        pois = len(result.get("candidate_pois", []))
        weather = len(result.get("weather_data", []))
        itinerary_days = len(result.get("current_itinerary", []))
        resp = str(result.get("assistant_response", ""))[:200]

        print(f"  耗时: {elapsed:.1f}s")
        print(f"  intent: {intent}")
        print(f"  needs_clarification: {clarification}")
        print(f"  waiting_for_confirmation: {confirmation}")
        print(f"  candidate_pois: {pois}")
        print(f"  weather_data: {weather}")
        print(f"  current_itinerary: {itinerary_days} days")
        print(f"  assistant_response: {resp}...")

        if itinerary_days > 0:
            print("  ✅ 成功（生成了行程）")
            return {"name": name, "status": "success", "elapsed": elapsed, "days": itinerary_days, "pois": pois}
        elif clarification:
            print("  ✅ 成功（需要追问）")
            return {"name": name, "status": "clarification", "elapsed": elapsed, "days": 0, "pois": 0}
        elif confirmation:
            print("  ✅ 成功（等待确认）")
            return {"name": name, "status": "confirmation", "elapsed": elapsed, "days": 0, "pois": 0}
        else:
            print(f"  ⚠️ 完成但没有行程")
            return {"name": name, "status": "no_itinerary", "elapsed": elapsed, "days": 0, "pois": pois}

    except asyncio.TimeoutError:
        elapsed = time.time() - start
        print(f"  耗时: {elapsed:.1f}s")
        print("  ❌ 超时")
        return {"name": name, "status": "timeout", "elapsed": elapsed, "days": 0, "pois": 0}
    except Exception as e:
        elapsed = time.time() - start
        print(f"  耗时: {elapsed:.1f}s")
        print(f"  ❌ 错误: {e}")
        return {"name": name, "status": "error", "elapsed": elapsed, "days": 0, "pois": 0}


async def main():
    print("真实端到端测试 — 10 个场景，无缓存，完整 Graph 执行")
    print("注意: 首次执行会走 Tavily + LLM，后续如果有 Redis 缓存会更快")

    scenarios = [
        ("成都4天美食游", "我想去成都玩4天，喜欢吃辣，预算4000", 120),
        ("杭州3天西湖游", "杭州3天，喜欢自然风光和茶文化", 120),
        ("西安5天历史游", "西安5天，对历史和文化感兴趣，预算6000", 120),
        ("厦门3天海滨海岛", "厦门3天，想看海和吃海鲜", 120),
        ("青岛4天啤酒海鲜", "青岛4天，想喝啤酒吃海鲜", 120),
        ("深圳3天现代都市", "深圳3天，喜欢现代建筑和科技", 120),
        ("南京4天文化游", "南京4天，对民国历史和文学感兴趣", 120),
        ("重庆5天火锅山城", "重庆5天，爱吃火锅，想看夜景", 120),
        ("广州3天早茶游", "广州3天，想吃早茶和粤菜", 120),
        ("上海5天深度游", "上海5天，喜欢老上海风情和法租界", 120),
        ("信息不完整", "我想去旅游", 60),
        ("火星城未知", "我要去火星城玩3天", 60),
    ]

    results = []
    for name, msg, timeout in scenarios:
        r = await run_scenario(name, msg, timeout)
        results.append(r)
        # 每个场景之间等 2 秒，避免 API 限流
        await asyncio.sleep(2)

    # 汇总
    print(f"\n{'='*70}")
    print("测试汇总")
    print("=" * 70)
    success = 0
    timeout_count = 0
    clarification = 0
    error_count = 0

    for r in results:
        status_icon = {
            "success": "✅",
            "clarification": "📝",
            "confirmation": "⏳",
            "timeout": "⏱️",
            "error": "❌",
            "no_itinerary": "⚠️",
        }.get(r["status"], "?")
        print(f"  {status_icon} {r['name']:20s} | {r['elapsed']:5.1f}s | {r['status']:15s} | {r['days']} days, {r['pois']} POIs")

        if r["status"] == "success":
            success += 1
        elif r["status"] == "timeout":
            timeout_count += 1
        elif r["status"] == "clarification":
            clarification += 1
        elif r["status"] == "error":
            error_count += 1

    print(f"\n总计: {len(results)} 个场景")
    print(f"  ✅ 成功生成行程: {success}")
    print(f"  📝 需要追问: {clarification}")
    print(f"  ⏱️ 超时: {timeout_count}")
    print(f"  ❌ 错误: {error_count}")


if __name__ == "__main__":
    asyncio.run(main())

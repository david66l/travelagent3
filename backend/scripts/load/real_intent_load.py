"""Real-environment concurrency probe for intent recognition.

Hits the actual DeepSeek Flash endpoint (no mocks) through the real LLMClient /
model-router path, ramping concurrency and reporting per-level success rate,
latency percentiles, throughput, and rate-limit/error counts.

Usage (env must be loaded so settings picks up DEEPSEEK_API_KEY):
    set -a && . ../.env && set +a
    .venv/bin/python scripts/load/real_intent_load.py
    LEVELS=20,50,100 .venv/bin/python scripts/load/real_intent_load.py
"""

import asyncio
import os
import time

from core.llm_client import LLMClient
from core.model_router import select_model
from core.settings import settings
from models.travel_slots import SlotParseOutput

LEVELS = [int(x) for x in os.environ.get("LEVELS", "10,30,60").split(",")]

_PROMPTS = [
    "我想下个月和女朋友去成都玩4天，喜欢美食和历史",
    "帮我规划一个北京三日游，带老人",
    "国庆想去三亚度假五天，预算一万",
    "和朋友周末去杭州两天，喜欢逛西湖",
    "下周出差上海，顺便玩一天",
    "想去云南大理丽江七天，节奏慢一点",
]


def _pct(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = min(len(s) - 1, int(round((p / 100) * (len(s) - 1))))
    return s[k]


async def _one(client: LLMClient, prompt: str) -> tuple[bool, float, str]:
    msgs = [
        {
            "role": "system",
            "content": "你是意图识别器，输出JSON: intent/confidence/sentiment/slots/missing_slots",
        },
        {"role": "user", "content": prompt},
    ]
    t = time.monotonic()
    try:
        await client.structured_call(msgs, SlotParseOutput, task_type="intent")
        return True, time.monotonic() - t, ""
    except Exception as exc:  # noqa: BLE001 — load probe records every failure mode
        kind = type(exc).__name__
        if "429" in str(exc) or "rate" in str(exc).lower():
            kind = "RateLimit"
        return False, time.monotonic() - t, kind


async def run_level(client: LLMClient, n: int) -> None:
    start = time.monotonic()
    results = await asyncio.gather(*(_one(client, _PROMPTS[i % len(_PROMPTS)]) for i in range(n)))
    wall = time.monotonic() - start

    oks = [r for r in results if r[0]]
    fails = [r for r in results if not r[0]]
    lat = [r[1] for r in oks]
    err_kinds: dict[str, int] = {}
    for _, _, k in fails:
        err_kinds[k] = err_kinds.get(k, 0) + 1

    print(
        f"  concurrency={n:<4} ok={len(oks):<4} fail={len(fails):<3} "
        f"wall={wall:5.2f}s throughput={n / wall:5.1f} req/s | "
        f"latency p50={_pct(lat, 50):4.2f}s p95={_pct(lat, 95):4.2f}s "
        f"p99={_pct(lat, 99):4.2f}s max={max(lat) if lat else 0:4.2f}s"
        + (f" | errors={err_kinds}" if err_kinds else "")
    )


async def main() -> None:
    print(
        f"target={settings.llm_model} base={settings.openai_base_url} "
        f"route(intent)={select_model(task_type='intent')}\n"
    )
    client = LLMClient()
    for n in LEVELS:
        await run_level(client, n)


if __name__ == "__main__":
    asyncio.run(main())

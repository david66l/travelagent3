"""
大小模型混合路由 — 按任务复杂度 + 成本 + 用户等级智能选模。

规则:
  - 意图识别/情感分析 → 本地 llama.cpp (qwen2.5-7b)
  - 复杂规划/文案润色 → OpenAI GPT-4o-mini
  - 成本熔断激活 → 全部降级到本地小模型
"""

from __future__ import annotations

import logging

from core.settings import settings
from core.user_tier import tier_limits

logger = logging.getLogger(__name__)

MODEL_REGISTRY = {
    "large": "qwen2.5-72b-instruct",
    "small": "qwen2.5-7b-instruct",
    "repair": "qwen2.5-14b-instruct",
    "writer": "qwen2.5-72b-instruct",
    "default": "qwen2.5-14b-instruct",
}

TASK_MODEL_MAP: dict[str, str] = {
    "intent": "small",
    "chat": "small",
    "clarify": "small",
    "simple_qa": "small",
    "sentiment": "small",
    "slot_filling": "small",
    "itinerary": "large",
    "planning": "large",
    "plan": "large",
    "polish": "large",
    "structured": "large",
    "writer": "writer",
    "repair": "repair",
    "fix": "repair",
    "validate": "repair",
}


def select_model(
    *,
    role: str = "guest",
    task_type: str = "chat",
    cost_circuit_active: bool = False,
    prefer_small: bool = False,
) -> str:
    """智能选模。意图识别 → 本地, 复杂规划 → OpenAI。"""
    limits = tier_limits(role)

    # 意图识别走本地 llama.cpp
    if task_type in ("intent", "clarify", "simple_qa", "sentiment", "slot_filling", "chat"):
        if settings.local_llm_enabled:
            return settings.local_llm_model

    # 成本熔断
    if cost_circuit_active or prefer_small:
        if settings.local_llm_enabled:
            return settings.local_llm_model
        return settings.small_model or MODEL_REGISTRY["small"]

    # 任务映射
    recommended_size = TASK_MODEL_MAP.get(task_type, "default")

    if limits.allow_large_model and recommended_size == "default":
        recommended_size = "large"
    if not limits.allow_large_model:
        if recommended_size in ("large", "writer", "repair"):
            recommended_size = "small"

    model_name = MODEL_REGISTRY.get(recommended_size, MODEL_REGISTRY["default"])

    if recommended_size == "small" and settings.small_model:
        model_name = settings.small_model
    if settings.default_model:
        model_name = settings.default_model

    return model_name or settings.llm_model

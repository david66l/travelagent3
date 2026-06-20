"""
大小模型混合路由 — 按任务复杂度 + 成本 + 用户等级智能选模。

规则:
  - 意图识别/情感分析/聊天 → small_model
  - 复杂规划/文案润色 → default_model（大模型）
  - 修复/校验 → repair_model
  - 成本熔断激活 → 全部降级到 small_model
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
    """智能选模。按任务类型 + 成本 + 用户等级选择模型名称。"""
    limits = tier_limits(role)

    # 1. 任务类型映射到模型类别
    recommended_size = TASK_MODEL_MAP.get(task_type, "default")

    # 2. 成本熔断或显式偏好小模型 → 强制降级为 small
    if cost_circuit_active or prefer_small:
        recommended_size = "small"

    # 3. 用户等级不允许大模型时降级
    if not limits.allow_large_model and recommended_size in ("large", "writer", "repair"):
        recommended_size = "small"

    # 4. 根据类别返回具体模型名（settings 配置优先于注册表默认值）
    if recommended_size == "small":
        # 当未配置 small_model 时，可回落到本地小模型名
        return settings.small_model or settings.local_llm_model or MODEL_REGISTRY["small"]
    if recommended_size == "repair":
        return settings.repair_model or MODEL_REGISTRY["repair"]
    if recommended_size == "writer":
        return MODEL_REGISTRY["writer"]
    if recommended_size == "large":
        return settings.default_model or MODEL_REGISTRY["large"]

    # default
    return settings.default_model or MODEL_REGISTRY["default"]

"""
模型路由 — 所有任务（含意图识别）默认走 DeepSeek Flash 云端模型。

规则:
  - 所有任务 (意图/规划/文案/校验/聊天/槽位…) → DeepSeek Flash (settings.llm_model)
  - 成本熔断激活 或 显式偏好小模型 或 未配置 DeepSeek Key → 回退本地 Qwen
本地 7B 吞吐有限，仅作为成本熔断/无云端 Key 时的降级兜底。
"""

from __future__ import annotations

import logging

from core.settings import settings

logger = logging.getLogger(__name__)

# Fallback model names if neither settings nor env provide one.
MODEL_REGISTRY = {
    "small": "qwen2.5-7b-instruct",
    "default": "qwen2.5-14b-instruct",
}

# No task is pinned to the local Qwen anymore; intent now also uses DeepSeek
# Flash. The local model is reached only via the cost-circuit / no-key fallback.
LOCAL_ONLY_TASKS: set[str] = set()


def _local_model() -> str:
    return settings.local_llm_model or MODEL_REGISTRY["small"]


def select_model(
    *,
    role: str = "guest",
    task_type: str = "chat",
    cost_circuit_active: bool = False,
    prefer_small: bool = False,
) -> str:
    """选模：所有任务默认走 DeepSeek Flash，降级时回退本地 Qwen。"""
    # 1. 显式 pin 到本地的任务（当前为空）→ 本地小模型
    if task_type in LOCAL_ONLY_TASKS:
        return _local_model()

    # 2. 成本熔断 / 显式偏好小模型 / 未配置 DeepSeek Key → 回退本地
    no_cloud_key = not (settings.deepseek_api_key or settings.openai_api_key)
    if (cost_circuit_active or prefer_small or no_cloud_key) and settings.local_llm_enabled:
        return _local_model()

    # 3. 其余所有任务 → DeepSeek Flash 云端模型
    return settings.llm_model or settings.default_model or MODEL_REGISTRY["default"]

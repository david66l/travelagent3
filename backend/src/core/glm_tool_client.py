"""GLM transport for the existing NativeToolAgentPolicy (no decision logic).

One client belongs to one sequential episode. Raw request/response evidence is
for private evaluation audits; reasoning_content is not a student SFT target.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import time
from typing import Any

import httpx

from core.inference_metrics import InferenceMetrics

SINGLE_CALL_REMINDER = (
    "接口约束：本轮必须且只能调用一个工具。即使多个工具互相独立，也不要在同一响应中调用多个工具；"
    "选一个作为下一步，收到执行结果后再决定后续动作。不要输出计划或普通正文，使用原生 tool_calls 返回一个调用。"
)


class GLMProviderError(RuntimeError):
    """Infrastructure failure, never eligible for policy self-repair."""


class GLMToolClient:
    def __init__(self, api_key: str, *,
                 base_url: str = "https://open.bigmodel.cn/api/paas/v4",
                 timeout: float = 180, transport: httpx.AsyncBaseTransport | None = None,
                 reasoning_effort: str = "max"):
        if not api_key:
            raise ValueError("ZAI_API_KEY is required")
        if reasoning_effort not in {"max", "high", "low"}:
            raise ValueError("Unsupported GLM reasoning_effort")
        self.reasoning_effort = reasoning_effort
        if base_url.rstrip("/") not in {
            "https://open.bigmodel.cn/api/paas/v4", "https://api.z.ai/api/paas/v4"
        }:
            raise ValueError("Use the selected provider's official general API endpoint")
        self.base_url = base_url.rstrip("/")
        self._http = httpx.AsyncClient(
            headers={"Authorization": f"Bearer {api_key}"}, timeout=timeout,
            transport=transport, follow_redirects=False,
        )
        self.last_token_usage = 0
        self.last_request_metrics = None
        self.last_generation_audit: dict[str, Any] = {}

    async def aclose(self):
        await self._http.aclose()

    async def tool_call(self, messages, tools, *, temperature=1.0,
                        max_tokens=8192, task_type="agent_policy",
                        model_override="glm-5.3-flash", seed=None):
        if seed is not None:
            raise ValueError("GLM provider seed is not supported by this adapter")
        request = {
            "model": model_override,
            "messages": [*messages, {"role": "user", "content": SINGLE_CALL_REMINDER}],
            "tools": tools,
            "tool_choice": "auto", "stream": False,
            "temperature": temperature, "top_p": 0.95, "max_tokens": max_tokens,
            "thinking": {"type": "enabled", "clear_thinking": True},
            "reasoning_effort": self.reasoning_effort,
        }
        self.last_token_usage = 0
        self.last_request_metrics = None
        self.last_generation_audit = {
            "request": request, "endpoint": self.base_url,
            "request_sha256": hashlib.sha256(json.dumps(request, ensure_ascii=False,
                sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
            "reasoning_tokens": None, "usage_known": False,
        }
        started = time.perf_counter()
        try:
            response = await self._http.post(self.base_url + "/chat/completions", json=request)
        except asyncio.CancelledError:
            self.last_generation_audit["request_cancelled"] = True
            raise
        except httpx.HTTPError as exc:
            # Do not include request headers or arbitrary upstream error text.
            self.last_generation_audit["provider_error"] = type(exc).__name__
            raise GLMProviderError(type(exc).__name__) from None
        finally:
            self.last_generation_audit["request_latency_ms"] = (time.perf_counter() - started) * 1000
        self.last_generation_audit["http_status"] = response.status_code
        if response.status_code == 429:
            from email.utils import parsedate_to_datetime
            value = response.headers.get("Retry-After", "")
            try:
                seconds = float(value)
            except ValueError:
                try:
                    seconds = parsedate_to_datetime(value).timestamp() - time.time()
                except (ValueError, TypeError, OverflowError):
                    seconds = 30.0
            import math
            self.last_generation_audit["retry_after_seconds"] = max(0.0, seconds) if math.isfinite(seconds) else 30.0
        if response.status_code != 200:
            try:
                code = str(response.json().get("error", {}).get("code", "unknown"))[:64]
            except (ValueError, AttributeError):
                code = "unknown"
            detail = f"HTTP {response.status_code}; provider code {code}"
            self.last_generation_audit["provider_error"] = detail
            raise GLMProviderError(detail)
        try:
            raw = response.json()
        except ValueError:
            self.last_generation_audit["provider_error"] = "non-JSON response"
            raise GLMProviderError("non-JSON response") from None
        self.last_generation_audit["response"] = raw
        usage = raw.get("usage") or {}
        if not all(isinstance(usage.get(k), int) and usage[k] >= 0
                   for k in ("prompt_tokens", "completion_tokens")):
            self.last_generation_audit["provider_error"] = "missing token usage"
            raise GLMProviderError("missing token usage; cannot enforce completion budget")
        choices = raw.get("choices") or []
        finish = choices[0].get("finish_reason") if choices else None
        self.last_token_usage = usage["completion_tokens"]
        self.last_request_metrics = InferenceMetrics(
            model=raw.get("model") or model_override, backend="glm-general-api",
            task_type=task_type, thinking_mode="enabled",
            prompt_tokens=usage["prompt_tokens"], completion_tokens=usage["completion_tokens"],
            cached_prompt_tokens=(usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0),
            request_latency_ms=self.last_generation_audit["request_latency_ms"], finish_reason=finish,
        )
        self.last_generation_audit.update(
            usage_known=True,
            reasoning_tokens=(usage.get("completion_tokens_details") or {}).get("reasoning_tokens"),
            inference_metrics=self.last_request_metrics.model_dump(mode="json"),
        )
        if finish == "length":
            raise ValueError("GLM output truncated at max_tokens")
        if len(choices) != 1:
            raise ValueError("Expected exactly one GLM choice")
        calls = choices[0].get("message", {}).get("tool_calls") or []
        if len(calls) != 1:
            raise ValueError("Expected exactly one native tool call")
        function = calls[0].get("function") or {}
        name = function.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError("Missing function name")
        arguments = function.get("arguments")
        if isinstance(arguments, str):
            arguments = json.loads(arguments)
        if not isinstance(arguments, dict):
            raise ValueError("Function arguments must be a JSON object")
        return {"action": name, "arguments": arguments}

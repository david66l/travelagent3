import json
import time
import asyncio
from typing import Any, Optional, TypeVar

from openai import AsyncOpenAI, APIStatusError
from pydantic import BaseModel

from core.cost_circuit_breaker import is_cost_circuit_active, record_daily_tokens
from core.metrics import (
    record_llm_duration,
    record_llm_tokens,
    set_cost_circuit_active,
    set_daily_cost_gauges,
)
from core.model_router import select_model
from core.prompt_compress import compress_message_history
from core.settings import settings
from core.user_tier import resolve_tier

T = TypeVar("T", bound=BaseModel)


class LLMClient:
    def __init__(self):
        base_url = settings.vllm_base_url if settings.vllm_enabled else settings.openai_base_url
        api_key = settings.vllm_api_key if settings.vllm_enabled else settings.openai_api_key
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=settings.llm_timeout,
        )
        self.model = settings.llm_model
        self._using_vllm = settings.vllm_enabled

        # 本地 llama.cpp 客户端（意图识别专用）
        self._local_client: AsyncOpenAI | None = None
        if settings.local_llm_enabled:
            self._local_client = AsyncOpenAI(
                api_key="not-needed",
                base_url=settings.local_llm_url,
                timeout=settings.llm_timeout,
            )

    def _client_for_model(self, model_name: str) -> tuple[AsyncOpenAI, str]:
        """根据模型名返回对应的客户端和实际模型名。"""
        if settings.local_llm_enabled and model_name == settings.local_llm_model:
            return self._local_client or self.client, model_name
        return self.client, model_name

    async def _create_completion(self, **kwargs: Any) -> Any:
        """Call OpenAI-compatible API with optional vLLM 503 retry."""
        model_name = kwargs.get("model", self.model)
        client, _ = self._client_for_model(model_name)
        max_attempts = settings.vllm_max_retries + 1 if self._using_vllm else 1
        last_exc: Optional[Exception] = None
        for attempt in range(max_attempts):
            try:
                return await client.chat.completions.create(**kwargs)
            except APIStatusError as exc:
                last_exc = exc
                if self._using_vllm and exc.status_code == 503 and attempt < max_attempts - 1:
                    await asyncio.sleep(0.5 * (attempt + 1))
                    continue
                raise
        if last_exc:
            raise last_exc
        raise RuntimeError("LLM completion failed without exception")

    async def _prepare_request(
        self,
        messages: list[dict[str, str]],
        task_type: Optional[str] = None,
    ) -> tuple[str, list[dict[str, str]], str]:
        from core.thought_logger import get_current_step_name, get_current_user_role

        role = get_current_user_role() or "user"
        tier = resolve_tier(role)
        task = task_type or get_current_step_name() or "chat"
        cost_active = await is_cost_circuit_active()
        set_cost_circuit_active(cost_active)
        model = select_model(
            role=role,
            task_type=task,
            cost_circuit_active=cost_active,
        )
        compressed = compress_message_history(messages)
        return model, compressed, tier

    async def chat(
        self,
        messages: list[dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        task_type: Optional[str] = None,
    ) -> str:
        model, prepared_messages, tier = await self._prepare_request(messages, task_type)
        start = time.monotonic()
        response = await self._create_completion(
            model=model,
            messages=prepared_messages,  # type: ignore
            temperature=temperature or settings.llm_temperature,
            max_tokens=max_tokens or settings.llm_max_tokens,
        )
        duration = time.monotonic() - start
        self._log_usage(response, model=model, tier=tier, duration_s=duration)
        return response.choices[0].message.content or ""

    async def structured_call(
        self,
        messages: list[dict[str, str]],
        response_model: type[T],
        temperature: Optional[float] = None,
        task_type: Optional[str] = None,
    ) -> T:
        model, prepared_messages, tier = await self._prepare_request(messages, task_type)
        start = time.monotonic()
        response = await self._create_completion(
            model=model,
            messages=prepared_messages,  # type: ignore
            response_format={"type": "json_object"},
            temperature=temperature or 0.3,
            max_tokens=settings.llm_max_tokens,
        )
        duration = time.monotonic() - start
        self._log_usage(response, model=model, tier=tier, duration_s=duration)
        content = response.choices[0].message.content or "{}"
        return response_model.model_validate_json(content)

    async def json_chat(
        self,
        messages: list[dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        task_type: Optional[str] = None,
    ) -> dict:
        model, prepared_messages, tier = await self._prepare_request(messages, task_type)
        start = time.monotonic()
        response = await self._create_completion(
            model=model,
            messages=prepared_messages,  # type: ignore
            response_format={"type": "json_object"},
            temperature=temperature or settings.llm_temperature,
            max_tokens=max_tokens or settings.llm_max_tokens,
        )
        duration = time.monotonic() - start
        self._log_usage(response, model=model, tier=tier, duration_s=duration)
        content = response.choices[0].message.content or "{}"
        return json.loads(content)

    async def stream_chat(
        self,
        messages: list[dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        task_type: Optional[str] = None,
    ) -> Any:
        """Stream LLM tokens. Yields content chunks."""
        model, prepared_messages, tier = await self._prepare_request(messages, task_type)
        start = time.monotonic()
        response = await self._create_completion(
            model=model,
            messages=prepared_messages,  # type: ignore
            temperature=temperature or settings.llm_temperature,
            max_tokens=max_tokens or settings.llm_max_tokens,
            stream=True,
            stream_options={"include_usage": True},
        )
        full_content = ""
        prompt_tokens = 0
        completion_tokens = 0
        try:
            async for chunk in response:
                delta = chunk.choices[0].delta.content if chunk.choices else None
                if delta:
                    full_content += delta
                    yield delta
                if chunk.usage:
                    prompt_tokens = chunk.usage.prompt_tokens or prompt_tokens
                    completion_tokens = chunk.usage.completion_tokens or completion_tokens
        finally:
            duration = time.monotonic() - start
            if prompt_tokens or completion_tokens:
                total_tokens = prompt_tokens + completion_tokens
                from core.token_quota import check_and_record_tokens
                from core.thought_logger import (
                    get_current_session_id,
                    get_current_step_name,
                    get_current_user_id,
                    get_current_user_role,
                    thought_logger,
                )

                user_id = get_current_user_id()
                user_role = get_current_user_role() or "user"
                step_name = get_current_step_name()
                session_id = get_current_session_id()
                if user_id and total_tokens > 0:
                    try:
                        loop = asyncio.get_running_loop()
                        loop.create_task(
                            check_and_record_tokens(user_id, user_role, total_tokens)
                        )
                        loop.create_task(record_daily_tokens(total_tokens))
                    except RuntimeError:
                        pass
                if step_name and session_id:
                    thought_logger.log_llm_call(
                        session_id=session_id,
                        model=model,
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                    )
                record_llm_tokens(
                    model,
                    step_name or "chat",
                    prompt_tokens,
                    completion_tokens,
                    user_tier=tier,
                )
                record_llm_duration(model, step_name or "chat", duration)

    def _log_usage(
        self,
        response: Any,
        *,
        model: str,
        tier: str,
        duration_s: float,
    ) -> None:
        try:
            if response.usage:
                from core.thought_logger import (
                    get_current_session_id,
                    get_current_step_name,
                    get_current_user_id,
                    get_current_user_role,
                    thought_logger,
                )

                step_name = get_current_step_name()
                session_id = get_current_session_id()
                user_id = get_current_user_id()
                user_role = get_current_user_role() or "user"
                prompt_tokens = response.usage.prompt_tokens or 0
                completion_tokens = response.usage.completion_tokens or 0
                total_tokens = prompt_tokens + completion_tokens

                if user_id and total_tokens > 0:
                    import asyncio

                    from core.token_quota import check_and_record_tokens

                    try:
                        loop = asyncio.get_running_loop()
                        loop.create_task(
                            check_and_record_tokens(user_id, user_role, total_tokens)
                        )
                        loop.create_task(record_daily_tokens(total_tokens))
                    except RuntimeError:
                        pass

                if step_name and session_id:
                    thought_logger.log_llm_call(
                        session_id=session_id,
                        model=model,
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                    )

                task_type = step_name or "chat"
                record_llm_tokens(
                    model,
                    task_type,
                    prompt_tokens,
                    completion_tokens,
                    user_tier=tier,
                )
                record_llm_duration(model, task_type, duration_s)
        except Exception:
            pass


llm = LLMClient()

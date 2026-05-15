import json
from typing import Any, Optional, TypeVar

from openai import AsyncOpenAI
from pydantic import BaseModel

from core.settings import settings

T = TypeVar("T", bound=BaseModel)


class LLMClient:
    def __init__(self):
        self.client = AsyncOpenAI(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
            timeout=60,  # 60s timeout for all API calls
        )
        self.model = settings.llm_model

    async def chat(
        self,
        messages: list[dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,  # type: ignore
            temperature=temperature or settings.llm_temperature,
            max_tokens=max_tokens or settings.llm_max_tokens,
        )

        # Log token usage via contextvar (safe for parallel nodes)
        self._log_usage(response)

        return response.choices[0].message.content or ""

    async def structured_call(
        self,
        messages: list[dict[str, str]],
        response_model: type[T],
        temperature: Optional[float] = None,
    ) -> T:
        """Use JSON mode for structured output."""
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,  # type: ignore
            response_format={"type": "json_object"},
            temperature=temperature or 0.3,
            max_tokens=settings.llm_max_tokens,
        )

        # Log token usage via contextvar (safe for parallel nodes)
        self._log_usage(response)

        content = response.choices[0].message.content or "{}"
        return response_model.model_validate_json(content)

    async def json_chat(
        self,
        messages: list[dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> dict:
        """Chat with JSON mode enforced. Returns parsed dict."""
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,  # type: ignore
            response_format={"type": "json_object"},
            temperature=temperature or settings.llm_temperature,
            max_tokens=max_tokens or settings.llm_max_tokens,
        )
        self._log_usage(response)
        content = response.choices[0].message.content or "{}"
        return json.loads(content)

    def _log_usage(self, response: Any) -> None:
        """Log LLM token usage to the current step via contextvar."""
        try:
            if response.usage:
                from core.thought_logger import (
                    thought_logger,
                    get_current_step_name,
                    get_current_session_id,
                )

                step_name = get_current_step_name()
                session_id = get_current_session_id()
                if step_name and session_id:
                    thought_logger.log_llm_call(
                        session_id=session_id,
                        model=self.model,
                        prompt_tokens=response.usage.prompt_tokens or 0,
                        completion_tokens=response.usage.completion_tokens or 0,
                    )
        except Exception:
            # Don't let logging failures break the main flow
            pass


llm = LLMClient()

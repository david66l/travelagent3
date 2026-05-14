"""Helpers for configuring LLM mocks in tests."""

from unittest.mock import AsyncMock

from schemas import IntentResult


def make_intent_result(
    intent: str,
    entities: dict | None = None,
    missing_required: list[str] | None = None,
    missing_recommended: list[str] | None = None,
    preference_changes: list[dict] | None = None,
    clarification_questions: list[str] | None = None,
    confidence: float = 0.95,
    reasoning: str = "",
) -> IntentResult:
    """Factory for constructing IntentResult test data."""
    return IntentResult(
        intent=intent,
        confidence=confidence,
        user_entities=entities or {},
        missing_required=missing_required or [],
        missing_recommended=missing_recommended or [],
        preference_changes=preference_changes or [],
        clarification_questions=clarification_questions or [],
        reasoning=reasoning,
    )


def configure_mock_llm_for_intent(mock_llm, intent_result: IntentResult) -> None:
    """Configure the global LLM mock to return a specific intent result."""
    mock_llm.structured_call = AsyncMock(return_value=intent_result)


def configure_mock_llm_for_chat(mock_llm, response: str) -> None:
    """Configure the global LLM mock to return a specific chat response."""
    mock_llm.chat = AsyncMock(return_value=response)


def configure_mock_llm_for_json(mock_llm, data: dict) -> None:
    """Configure the global LLM mock to return a specific JSON response."""
    mock_llm.json_chat = AsyncMock(return_value=data)

"""Text input normalizer."""

from perception.types import PerceptionOutput


def build_text_perception(
    content: str,
    *,
    conversation_id: str | None = None,
    session_id: str | None = None,
) -> PerceptionOutput:
    """Normalize a plain-text user message into PerceptionOutput."""
    return PerceptionOutput(
        user_input=content.strip(),
        attachments_meta=[],
        external_event=None,
    )

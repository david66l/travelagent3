"""Types for the perception layer."""

from typing import Literal, NotRequired, TypedDict


class AttachmentMeta(TypedDict):
    """Metadata for a single user attachment after parsing."""

    type: Literal["image", "pdf", "audio", "file", "url"]
    mime_type: str
    source: str  # base64 data URI or public URL
    filename: str | None
    extracted_text: str | None
    metadata: dict


class PerceptionOutput(TypedDict):
    """Unified output from the perception layer.

    This is the canonical shape handed off to the LangGraph orchestrator.
    """

    user_input: str  # normalized text merged from content + attachments
    messages: NotRequired[list[dict]]  # optional normalized message history
    attachments_meta: list[AttachmentMeta]
    external_event: dict | None

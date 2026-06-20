"""Perception layer: unify text / image / PDF / webhook inputs into PerceptionOutput."""

from perception.attachment_parser import AttachmentParser
from perception.text_input import build_text_perception
from perception.types import (
    AttachmentMeta,
    PerceptionOutput,
)

__all__ = [
    "AttachmentMeta",
    "AttachmentParser",
    "build_text_perception",
    "PerceptionOutput",
]

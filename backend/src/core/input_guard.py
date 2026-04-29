"""Input Guard - sanitize user inputs to prevent prompt injection."""

import re


# Common prompt injection patterns
_INJECTION_PATTERNS = [
    r"(?i)ignore\s+(all\s+)?previous\s+instructions",
    r"(?i)ignore\s+(the\s+)?above\s+instructions",
    r"(?i)you\s+are\s+now\s+",
    r"(?i)new\s+role\s*:",
    r"(?i)system\s*override",
    r"(?i)===\s*SYSTEM\s*===",
    r"(?i)<\s*system\s*>",
    r"(?i)```\s*system",
    r"(?i)\[\s*system\s*\]",
    r"(?i)DAN\s*mode",
    r"(?i)jailbreak",
    r"(?i)prompt\s*injection",
]

# Control characters to strip (except normal newlines)
_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")


def sanitize_user_input(text: str) -> str:
    """Sanitize user input to prevent prompt injection attacks.

    1. Strips control characters
    2. Detects and neutralizes common injection patterns
    3. Limits length
    4. Normalizes whitespace
    """
    if not isinstance(text, str):
        text = str(text)

    # Limit length (generous limit for travel queries)
    text = text[:2000]

    # Strip control characters
    text = _CONTROL_CHARS_RE.sub("", text)

    # Neutralize injection patterns by breaking them up
    for pattern in _INJECTION_PATTERNS:
        text = re.sub(pattern, lambda m: _neutralize(m.group(0)), text)

    # Normalize excessive whitespace
    text = " ".join(text.split())

    return text


def _neutralize(matched: str) -> str:
    """Break up a matched injection pattern by inserting zero-width spaces."""
    return "\u200b".join(matched)


def wrap_user_input(text: str) -> str:
    """Wrap sanitized user input in clear boundaries for LLM prompts."""
    sanitized = sanitize_user_input(text)
    return f"[用户输入开始]\n{sanitized}\n[用户输入结束]"

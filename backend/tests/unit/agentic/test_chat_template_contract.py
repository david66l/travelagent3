import hashlib
from pathlib import Path

from agentic.chat_template_contract import (
    AGENT_CHAT_TEMPLATE_KWARGS,
    AGENT_CHAT_TEMPLATE_PATH,
    AGENT_CHAT_TEMPLATE_SHA256,
    AGENT_RENDER_PROTOCOL_VERSION,
    install_agent_chat_template,
    load_agent_chat_template,
    validate_agent_chat_template,
)


class _Tokenizer:
    chat_template = "native-template"


def test_agent_chat_template_asset_is_hash_pinned_and_installable():
    template = load_agent_chat_template()

    assert AGENT_RENDER_PROTOCOL_VERSION == "qwen3-agent-prefix-preserving.v1"
    assert AGENT_CHAT_TEMPLATE_KWARGS == {"enable_thinking": False}
    assert hashlib.sha256(template.encode("utf-8")).hexdigest() == (
        AGENT_CHAT_TEMPLATE_SHA256
    )
    assert AGENT_CHAT_TEMPLATE_PATH.is_file()

    tokenizer = _Tokenizer()
    assert install_agent_chat_template(tokenizer) == template
    assert tokenizer.chat_template == template
    validate_agent_chat_template(tokenizer.chat_template)


def test_vllm_policy_launcher_pins_the_shared_template():
    repo_root = Path(__file__).resolve().parents[4]
    launcher = (repo_root / "scripts/serve_native_react_multi_lora.sh").read_text(
        encoding="utf-8"
    )

    assert "--chat-template" in launcher
    assert "qwen3_agent_prefix_preserving_v1.jinja" in launcher
    assert AGENT_CHAT_TEMPLATE_SHA256 in launcher

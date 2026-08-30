"""Minimal OpenAI-compatible tool-call server for remote GPU smoke tests.

This is intentionally a development/evaluation adapter, not a production
serving stack.  It lets the local TravelAgent runtime exercise its real
OpenAI-tool protocol against a Hugging Face checkpoint on an SSH GPU host.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from contextlib import asynccontextmanager
from typing import Any
from uuid import uuid4

import torch
from fastapi import FastAPI, HTTPException
from transformers import AutoModelForCausalLM, AutoTokenizer


_TOOL_CALL = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    return parser.parse_args()


ARGS = _parse_args()
TOKENIZER: Any = None
MODEL: Any = None


@asynccontextmanager
async def _lifespan(_: FastAPI):
    global TOKENIZER, MODEL
    TOKENIZER = AutoTokenizer.from_pretrained(ARGS.model, trust_remote_code=False)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    MODEL = AutoModelForCausalLM.from_pretrained(
        ARGS.model,
        device_map="auto",
        dtype=dtype,
        trust_remote_code=False,
    )
    MODEL.eval()
    yield


app = FastAPI(lifespan=_lifespan)


@app.get("/v1/models")
async def models() -> dict[str, Any]:
    return {
        "object": "list",
        "data": [{"id": ARGS.model, "object": "model", "owned_by": "local"}],
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: dict[str, Any]) -> dict[str, Any]:
    if MODEL is None or TOKENIZER is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    messages = request.get("messages") or []
    tools = request.get("tools") or []
    encoded = TOKENIZER.apply_chat_template(
        messages,
        tools=tools or None,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        enable_thinking=False,
    )
    device = next(MODEL.parameters()).device
    encoded = {key: value.to(device) for key, value in encoded.items()}
    prompt_tokens = int(encoded["input_ids"].shape[-1])
    with torch.inference_mode():
        generated = MODEL.generate(
            **encoded,
            max_new_tokens=min(int(request.get("max_tokens") or ARGS.max_new_tokens), 512),
            do_sample=False,
            pad_token_id=TOKENIZER.pad_token_id or TOKENIZER.eos_token_id,
            eos_token_id=TOKENIZER.eos_token_id,
        )
    completion_ids = generated[0, prompt_tokens:]
    raw = TOKENIZER.decode(completion_ids, skip_special_tokens=False)
    match = _TOOL_CALL.search(raw)
    tool_calls: list[dict[str, Any]] = []
    content: str | None = raw
    finish_reason = "stop"
    if match:
        try:
            payload = json.loads(match.group(1))
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=500, detail=f"invalid model tool JSON: {exc}") from exc
        tool_calls = [
            {
                "id": f"call_{uuid4().hex}",
                "type": "function",
                "function": {
                    "name": str(payload.get("name") or ""),
                    "arguments": json.dumps(
                        payload.get("arguments") or {}, ensure_ascii=False, separators=(",", ":")
                    ),
                },
            }
        ]
        content = None
        finish_reason = "tool_calls"
    return {
        "id": f"chatcmpl-{uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": str(request.get("model") or ARGS.model),
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content, "tool_calls": tool_calls},
                "finish_reason": finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": int(completion_ids.numel()),
            "total_tokens": prompt_tokens + int(completion_ids.numel()),
        },
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=ARGS.host, port=ARGS.port, log_level="info")

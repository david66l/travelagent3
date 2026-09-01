"""Audit the isolated GRPO runtime before any GPU training job is admitted."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import inspect
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


REQUIRED_DISTRIBUTIONS = {
    "torch": "2.6.0",
    "transformers": "5.12.1",
    "trl": "1.9.2",
    "peft": "0.20.0",
    "bitsandbytes": "0.50.0",
    "accelerate": "1.14.0",
    "datasets": "5.0.1",
}
FORBIDDEN_DISTRIBUTIONS = (
    "travel-agent",
    "vllm",
    "ortools",
    "SQLAlchemy",
    "pgvector",
)
REQUIRED_GRPO_CONFIG_FIELDS = (
    "num_generations_eval",
    "max_tool_calling_iterations",
    "loss_type",
    "use_vllm",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _distribution_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def audit(checkpoint: Path) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    versions = {
        name: _distribution_version(name) for name in REQUIRED_DISTRIBUTIONS
    }
    for name, expected in REQUIRED_DISTRIBUTIONS.items():
        if versions[name] != expected:
            errors.append(f"VERSION_MISMATCH:{name}:{versions[name]}!={expected}")
    forbidden = {
        name: version
        for name in FORBIDDEN_DISTRIBUTIONS
        if (version := _distribution_version(name)) is not None
    }
    if forbidden:
        errors.append("FORBIDDEN_DISTRIBUTIONS_PRESENT:" + ",".join(sorted(forbidden)))

    pip_check = subprocess.run(
        [sys.executable, "-m", "pip", "check"],
        capture_output=True,
        text=True,
        check=False,
    )
    if pip_check.returncode:
        errors.append("PIP_CHECK_FAILED")

    import bitsandbytes as bnb
    import torch
    from peft import PeftConfig
    from transformers import AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer

    config_fields = set(getattr(GRPOConfig, "__dataclass_fields__", {}))
    config_fields.update(inspect.signature(GRPOConfig).parameters)
    missing_config_fields = sorted(set(REQUIRED_GRPO_CONFIG_FIELDS) - config_fields)
    trainer_fields = set(inspect.signature(GRPOTrainer.__init__).parameters)
    if missing_config_fields:
        errors.append("GRPO_CONFIG_API_MISSING:" + ",".join(missing_config_fields))
    if "environment_factory" not in trainer_fields:
        errors.append("GRPO_TRAINER_API_MISSING:environment_factory")

    cuda_available = torch.cuda.is_available()
    bf16_supported = bool(cuda_available and torch.cuda.is_bf16_supported())
    if not cuda_available:
        errors.append("CUDA_UNAVAILABLE")
    if not bf16_supported:
        errors.append("BF16_UNAVAILABLE")
    bnb_nf4_ok = False
    if cuda_available:
        try:
            tensor = torch.linspace(-1, 1, 256, device="cuda", dtype=torch.bfloat16)
            quantized, state = bnb.functional.quantize_4bit(tensor, quant_type="nf4")
            restored = bnb.functional.dequantize_4bit(quantized, state)
            bnb_nf4_ok = bool(torch.isfinite(restored).all().item())
            del tensor, quantized, state, restored
            torch.cuda.empty_cache()
        except Exception as exc:  # pragma: no cover - exercised on the cloud GPU
            errors.append(f"BNB_NF4_FAILED:{type(exc).__name__}:{exc}")
    if not bnb_nf4_ok and cuda_available and not any(
        item.startswith("BNB_NF4_FAILED") for item in errors
    ):
        errors.append("BNB_NF4_FAILED:non_finite_output")

    adapter_config_path = checkpoint / "adapter_config.json"
    adapter_path = checkpoint / "adapter_model.safetensors"
    for path in (adapter_config_path, adapter_path):
        if not path.is_file():
            errors.append(f"CHECKPOINT_FILE_MISSING:{path.name}")
    base_model_path: Path | None = None
    tokenizer_chat_template = False
    if adapter_config_path.is_file():
        peft_config = PeftConfig.from_pretrained(checkpoint)
        base_model_path = Path(peft_config.base_model_name_or_path)
        if not base_model_path.exists():
            errors.append("BASE_MODEL_PATH_MISSING")
        tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=False)
        tokenizer_chat_template = bool(tokenizer.chat_template)
        if not tokenizer_chat_template:
            errors.append("TOKENIZER_CHAT_TEMPLATE_MISSING")

    gpu = None
    if cuda_available:
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        gpu = {
            "name": torch.cuda.get_device_name(0),
            "compute_capability": list(torch.cuda.get_device_capability(0)),
            "free_bytes": free_bytes,
            "total_bytes": total_bytes,
        }
    if sys.version_info < (3, 11):
        errors.append("PYTHON_TOO_OLD")
    return {
        "schema_version": "grpo-runtime-environment-audit.v1",
        "ready": not errors,
        "python": sys.version,
        "executable": sys.executable,
        "required_versions": REQUIRED_DISTRIBUTIONS,
        "installed_versions": versions,
        "forbidden_distributions": forbidden,
        "pip_check": {
            "returncode": pip_check.returncode,
            "stdout": pip_check.stdout.strip(),
            "stderr": pip_check.stderr.strip(),
        },
        "trl_api": {
            "required_config_fields": list(REQUIRED_GRPO_CONFIG_FIELDS),
            "missing_config_fields": missing_config_fields,
            "environment_factory": "environment_factory" in trainer_fields,
        },
        "cuda": {
            "available": cuda_available,
            "bf16_supported": bf16_supported,
            "bnb_nf4_ok": bnb_nf4_ok,
            "gpu": gpu,
        },
        "checkpoint": {
            "path": str(checkpoint),
            "adapter_config_sha256": (
                _sha256(adapter_config_path) if adapter_config_path.is_file() else None
            ),
            "adapter_sha256": _sha256(adapter_path) if adapter_path.is_file() else None,
            "base_model_path": str(base_model_path) if base_model_path else None,
            "tokenizer_chat_template": tokenizer_chat_template,
        },
        "warnings": warnings,
        "errors": errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.checkpoint.resolve())
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if report["ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

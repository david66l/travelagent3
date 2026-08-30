"""Create a validated linear PEFT adapter blend for bounded evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


_COMPATIBILITY_FIELDS = (
    "base_model_name_or_path",
    "peft_type",
    "task_type",
    "r",
    "lora_alpha",
    "modules_to_save",
)


def load_adapter_config(path: Path) -> dict[str, Any]:
    config_path = path / "adapter_config.json"
    if not config_path.is_file():
        raise ValueError(f"adapter config missing: {config_path}")
    return json.loads(config_path.read_text(encoding="utf-8"))


def validate_compatible_lora_configs(
    primary: dict[str, Any], candidate: dict[str, Any]
) -> None:
    mismatches = {
        field: (primary.get(field), candidate.get(field))
        for field in _COMPATIBILITY_FIELDS
        if primary.get(field) != candidate.get(field)
    }
    primary_targets = set(primary.get("target_modules") or [])
    candidate_targets = set(candidate.get("target_modules") or [])
    if primary_targets != candidate_targets:
        mismatches["target_modules"] = (
            sorted(primary_targets),
            sorted(candidate_targets),
        )
    if primary.get("peft_type") != "LORA":
        mismatches["peft_type"] = (primary.get("peft_type"), "LORA required")
    if mismatches:
        raise ValueError(f"incompatible LoRA adapters: {mismatches}")


def merge(
    primary_path: Path,
    candidate_path: Path,
    output_path: Path,
    *,
    candidate_weight: float,
) -> dict[str, Any]:
    if not 0 < candidate_weight < 1:
        raise ValueError("candidate_weight must be between 0 and 1")
    primary_config = load_adapter_config(primary_path)
    candidate_config = load_adapter_config(candidate_path)
    validate_compatible_lora_configs(primary_config, candidate_config)

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    if output_path.exists():
        raise ValueError(f"refusing to overwrite existing output: {output_path}")
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32
    base = AutoModelForCausalLM.from_pretrained(
        primary_config["base_model_name_or_path"],
        dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=False,
    )
    model = PeftModel.from_pretrained(base, primary_path, adapter_name="primary")
    model.load_adapter(candidate_path, adapter_name="candidate")
    model.add_weighted_adapter(
        adapters=["primary", "candidate"],
        weights=[1.0 - candidate_weight, candidate_weight],
        adapter_name="default",
        combination_type="linear",
    )
    model.set_adapter("default")
    output_path.mkdir(parents=True)
    model.save_pretrained(
        output_path,
        safe_serialization=True,
        selected_adapters=["default"],
    )
    tokenizer_source = candidate_path if (candidate_path / "tokenizer_config.json").is_file() else primary_path
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=False)
    tokenizer.save_pretrained(output_path)
    report = {
        "schema_version": "lora-linear-blend.v1",
        "method": "PEFT add_weighted_adapter linear",
        "primary_adapter": str(primary_path),
        "candidate_adapter": str(candidate_path),
        "primary_weight": 1.0 - candidate_weight,
        "candidate_weight": candidate_weight,
        "base_model": primary_config["base_model_name_or_path"],
        "compatibility_fields": list(_COMPATIBILITY_FIELDS),
        "target_modules": sorted(set(primary_config.get("target_modules") or [])),
        "status": "evaluation_only",
    }
    (output_path / "merge_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidate-weight", type=float, required=True)
    args = parser.parse_args()
    report = merge(
        args.primary,
        args.candidate,
        args.output,
        candidate_weight=args.candidate_weight,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

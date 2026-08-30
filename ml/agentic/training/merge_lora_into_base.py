"""Merge one audited LoRA adapter into its base model for static serving."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any


_TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
    "chat_template.jinja",
)


def copy_base_tokenizer_files(tokenizer_source: Path, output_path: Path) -> list[str]:
    copied: list[str] = []
    for name in _TOKENIZER_FILES:
        source = tokenizer_source / name
        if source.is_file():
            shutil.copy2(source, output_path / name)
            copied.append(name)
    if "tokenizer_config.json" not in copied:
        raise ValueError("base tokenizer_config.json is required for static serving")
    return copied


def load_merge_contract(adapter_path: Path, output_path: Path) -> dict[str, Any]:
    config_path = adapter_path / "adapter_config.json"
    report_path = adapter_path / "training_report.json"
    if not config_path.is_file() or not report_path.is_file():
        raise ValueError("adapter config and training report are required")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    training = json.loads(report_path.read_text(encoding="utf-8"))
    if config.get("peft_type") != "LORA":
        raise ValueError("only LoRA adapters can be merged")
    if training.get("status") != "trained":
        raise ValueError("adapter training report is not successful")
    if output_path.exists():
        raise ValueError(f"refusing to overwrite existing output: {output_path}")
    base_model = str(config.get("base_model_name_or_path") or "")
    if not base_model:
        raise ValueError("adapter does not declare a base model")
    return {
        "base_model": base_model,
        "dataset_version": training.get("dataset_version"),
        "run_scope": training.get("run_scope"),
        "adapter_config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "training_report_sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
    }


def merge(
    adapter_path: Path,
    output_path: Path,
    merge_dtype: str = "float16",
    adapter_scale: float = 1.0,
) -> dict[str, Any]:
    contract = load_merge_contract(adapter_path, output_path)
    if not 0 < adapter_scale <= 4:
        raise ValueError("adapter scale must be in (0, 4]")
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    dtypes = {"float16": torch.float16, "bfloat16": torch.bfloat16}
    if merge_dtype not in dtypes:
        raise ValueError(f"unsupported merge dtype: {merge_dtype}")
    dtype = dtypes[merge_dtype]
    base = AutoModelForCausalLM.from_pretrained(
        contract["base_model"],
        dtype=dtype,
        device_map={"": 0},
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base, adapter_path)
    merge_adapters = None
    if adapter_scale != 1.0:
        scaled_name = "static_scaled"
        model.add_weighted_adapter(
            adapters=["default"],
            weights=[adapter_scale],
            adapter_name=scaled_name,
            combination_type="linear",
        )
        model.set_adapter(scaled_name)
        merge_adapters = [scaled_name]
    merged = model.merge_and_unload(safe_merge=True, adapter_names=merge_adapters)
    output_path.mkdir(parents=True)
    merged.save_pretrained(output_path, safe_serialization=True, max_shard_size="4GB")
    # LoRA does not update the vocabulary.  Always keep the base tokenizer so a
    # tokenizer_config.json emitted by a newer training-time Transformers build
    # cannot make the merged model incompatible with the serving environment.
    tokenizer_source = Path(contract["base_model"])
    tokenizer_files = copy_base_tokenizer_files(tokenizer_source, output_path)
    report = {
        "schema_version": "merged-lora-base.v1",
        "status": "merged_for_static_serving",
        "adapter": str(adapter_path),
        "output": str(output_path),
        "tokenizer_source": str(tokenizer_source),
        "tokenizer_files": tokenizer_files,
        "merge_dtype": merge_dtype,
        "adapter_scale": adapter_scale,
        **contract,
    }
    (output_path / "merge_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--merge-dtype",
        choices=("float16", "bfloat16"),
        default="float16",
        help="float16 preserves small LoRA deltas better than bfloat16 for static serving",
    )
    parser.add_argument(
        "--adapter-scale",
        type=float,
        default=1.0,
        help="Calibrated multiplier for the LoRA delta; select on validation only",
    )
    args = parser.parse_args()
    print(
        json.dumps(
            merge(
                args.adapter,
                args.output,
                merge_dtype=args.merge_dtype,
                adapter_scale=args.adapter_scale,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

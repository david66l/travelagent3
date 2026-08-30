import pytest

from ml.agentic.training.merge_lora_adapters import validate_compatible_lora_configs


def _config():
    return {
        "base_model_name_or_path": "Qwen/Qwen2.5-3B-Instruct",
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "r": 16,
        "lora_alpha": 32,
        "target_modules": ["q_proj", "v_proj"],
        "modules_to_save": None,
    }


def test_merge_requires_same_lora_parameterization():
    primary = _config()
    candidate = _config()

    validate_compatible_lora_configs(primary, candidate)

    candidate["r"] = 8
    with pytest.raises(ValueError, match="incompatible LoRA adapters"):
        validate_compatible_lora_configs(primary, candidate)


def test_merge_compares_target_modules_without_order_sensitivity():
    primary = _config()
    candidate = _config()
    candidate["target_modules"] = list(reversed(candidate["target_modules"]))

    validate_compatible_lora_configs(primary, candidate)

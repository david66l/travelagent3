# Legacy ML training placeholder (not Agent Policy SFT)

The scripts in this directory only exercise MLflow wiring and simple JSONL
format checks. They do **not** run PEFT/QLoRA, do not produce model weights, and
must not be reported as completed SFT.

Audited Agent Policy data is built through
`scripts/build_sft_dataset.py`; see `ml/agentic/datasets/README.md`. A real
TRL/PEFT training entrypoint will consume that versioned dataset in the next
phase.

## Prerequisites

- `pip install mlflow` or `uv pip install -e ".[mlflow]"`
- `MLFLOW_TRACKING_URI=http://localhost:5000`

## Train LoRA

```bash
export MLFLOW_TRACKING_URI=http://localhost:5000
python ml/training/train_lora.py --experiment travel-lora --adapter travel-plan-v1
python ml/training/eval_lora.py --dataset ml/training/data/holdout.jsonl --adapter travel-plan-v1
```

Artifacts land under `ml/adapters/<adapter-id>/`; register production models via `core.mlflow_registry.register_model_version`.

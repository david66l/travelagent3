# ML training pipeline skeleton (M4/M5 integration with MLflow)

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

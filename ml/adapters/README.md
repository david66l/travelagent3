# LoRA adapter artifacts (M4)

Register three adapters with vLLM (`--enable-lora`):

| Adapter ID | Use case | Training script |
|------------|----------|-----------------|
| `travel-chat-v1` | Intent / chat / clarify | `ml/training/train_lora.py --adapter travel-chat-v1` |
| `travel-plan-v1` | Itinerary planning / polish | `ml/training/train_lora.py --adapter travel-plan-v1` |
| `travel-repair-v1` | Repair explanations | `ml/training/train_lora.py --adapter travel-repair-v1` |

Each subdirectory should contain the exported LoRA weights (e.g. `adapter_config.json` + `adapter_model.safetensors`).
Until real weights exist, vLLM runs the base model; backend `VLLM_ENABLED=false` keeps using cloud OpenAI-compatible APIs.

"""Score frozen teacher preferences by completion log-probability margin."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def summarize_preference_scores(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("preference score rows are empty")
    families: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        families[str(row["family"])].append(row)

    def summary(items: list[dict[str, Any]]) -> dict[str, Any]:
        margins = [float(item["mean_logprob_margin"]) for item in items]
        sequence_margins = [
            float(
                item.get(
                    "sum_logprob_margin",
                    float(item["chosen_sum_logprob"])
                    - float(item["rejected_sum_logprob"]),
                )
            )
            for item in items
        ]
        return {
            "pairs": len(items),
            "chosen_preferred": sum(margin > 0 for margin in margins),
            "preference_accuracy": round(
                sum(margin > 0 for margin in margins) / len(items), 6
            ),
            "mean_logprob_margin": round(sum(margins) / len(margins), 6),
            "median_logprob_margin": round(sorted(margins)[len(margins) // 2], 6),
            "sequence_chosen_preferred": sum(margin > 0 for margin in sequence_margins),
            "sequence_preference_accuracy": round(
                sum(margin > 0 for margin in sequence_margins) / len(items), 6
            ),
            "mean_sequence_logprob_margin": round(
                sum(sequence_margins) / len(sequence_margins), 6
            ),
            "median_sequence_logprob_margin": round(
                sorted(sequence_margins)[len(sequence_margins) // 2], 6
            ),
            "chosen_mean_logprob": round(
                sum(float(item["chosen_mean_logprob"]) for item in items) / len(items),
                6,
            ),
            "rejected_mean_logprob": round(
                sum(float(item["rejected_mean_logprob"]) for item in items)
                / len(items),
                6,
            ),
        }

    return {
        "overall": summary(rows),
        "families": {
            family: summary(items) for family, items in sorted(families.items())
        },
    }


def _load_pairs(path: Path) -> list[dict[str, Any]]:
    pairs = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not pairs:
        raise ValueError("preference file is empty")
    required = {"pair_id", "family", "messages", "tools", "chosen", "rejected"}
    for pair in pairs:
        missing = required - set(pair)
        if missing:
            raise ValueError(
                f"preference pair is missing {sorted(missing)}: {pair.get('pair_id')}"
            )
        if len(pair["messages"]) < 2 or pair["messages"][0].get("role") != "system":
            raise ValueError(f"preference prompt is invalid: {pair['pair_id']}")
    return pairs


def _token_ids(tokenizer: Any, value: Any) -> list[int]:
    if hasattr(value, "input_ids"):
        value = value.input_ids
    elif isinstance(value, dict):
        value = value["input_ids"]
    if hasattr(value, "tolist"):
        value = value.tolist()
    if value and isinstance(value[0], list):
        value = value[0]
    return list(value)


def _encoded_response(
    tokenizer: Any,
    pair: dict[str, Any],
    response: dict[str, Any],
) -> tuple[list[int], int]:
    messages = pair["messages"]
    kwargs = {"tools": pair["tools"], "enable_thinking": False}
    prompt_ids = _token_ids(
        tokenizer,
        tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            **kwargs,
        ),
    )
    full_ids = _token_ids(
        tokenizer,
        tokenizer.apply_chat_template(
            [*messages, response],
            tokenize=True,
            add_generation_prompt=False,
            **kwargs,
        ),
    )
    if full_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError(f"completion prefix mismatch: {pair['pair_id']}")
    if len(full_ids) <= len(prompt_ids):
        raise ValueError(f"empty completion: {pair['pair_id']}")
    return full_ids, len(prompt_ids)


def _score_encoded(
    model: Any, tokenizer: Any, rows: list[tuple[list[int], int]]
) -> list[dict[str, float | int]]:
    import torch

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    width = max(len(ids) for ids, _ in rows)
    input_ids = torch.full((len(rows), width), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((len(rows), width), dtype=torch.long)
    for index, (ids, _) in enumerate(rows):
        input_ids[index, : len(ids)] = torch.tensor(ids, dtype=torch.long)
        attention_mask[index, : len(ids)] = 1
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    with torch.inference_mode():
        logits = model(
            input_ids=input_ids, attention_mask=attention_mask, use_cache=False
        ).logits
        token_logprobs = (
            torch.log_softmax(logits[:, :-1].float(), dim=-1)
            .gather(-1, input_ids[:, 1:].unsqueeze(-1))
            .squeeze(-1)
        )
    scores = []
    for index, (ids, prompt_length) in enumerate(rows):
        start = prompt_length - 1
        end = len(ids) - 1
        values = token_logprobs[index, start:end]
        count = int(values.numel())
        total = float(values.sum().item())
        scores.append(
            {"tokens": count, "sum_logprob": total, "mean_logprob": total / count}
        )
    return scores


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    pairs = _load_pairs(args.preference_file)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=quantization,
        device_map={"": 0},
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    if args.adapter is not None:
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    encoded: list[tuple[str, str, list[int], int]] = []
    for pair in pairs:
        for kind in ("chosen", "rejected"):
            response = pair[kind]
            ids, prefix = _encoded_response(tokenizer, pair, response)
            encoded.append((pair["pair_id"], kind, ids, prefix))

    by_pair: dict[str, dict[str, dict[str, float | int]]] = defaultdict(dict)
    for start in range(0, len(encoded), args.batch_size):
        batch = encoded[start : start + args.batch_size]
        scores = _score_encoded(
            model, tokenizer, [(ids, prefix) for _, _, ids, prefix in batch]
        )
        for (pair_id, kind, _, _), score in zip(batch, scores, strict=True):
            by_pair[pair_id][kind] = score
        print(
            f"[{min(start + len(batch), len(encoded))}/{len(encoded)}] completions",
            flush=True,
        )

    rows = []
    family_by_id = {pair["pair_id"]: pair["family"] for pair in pairs}
    for pair in pairs:
        chosen = by_pair[pair["pair_id"]]["chosen"]
        rejected = by_pair[pair["pair_id"]]["rejected"]
        rows.append(
            {
                "pair_id": pair["pair_id"],
                "family": family_by_id[pair["pair_id"]],
                "chosen_tokens": chosen["tokens"],
                "rejected_tokens": rejected["tokens"],
                "chosen_sum_logprob": round(float(chosen["sum_logprob"]), 6),
                "rejected_sum_logprob": round(float(rejected["sum_logprob"]), 6),
                "chosen_mean_logprob": round(float(chosen["mean_logprob"]), 6),
                "rejected_mean_logprob": round(float(rejected["mean_logprob"]), 6),
                "sum_logprob_margin": round(
                    float(chosen["sum_logprob"]) - float(rejected["sum_logprob"]), 6
                ),
                "mean_logprob_margin": round(
                    float(chosen["mean_logprob"]) - float(rejected["mean_logprob"]), 6
                ),
            }
        )
    report = {
        "schema_version": "preference-logprob-evaluation.v1",
        "model": str(args.model),
        "adapter": str(args.adapter) if args.adapter is not None else None,
        "preference_file": str(args.preference_file),
        "quantization": "nf4-double-quant",
        "normalization": "mean assistant-completion token log-probability",
        **summarize_preference_scores(rows),
        "family_counts": dict(Counter(pair["family"] for pair in pairs)),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "scores.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--preference-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("batch-size must be positive")
    report = evaluate(args)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

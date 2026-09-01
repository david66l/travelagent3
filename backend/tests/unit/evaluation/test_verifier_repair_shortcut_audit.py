from pathlib import Path

from agentic.grpo_training import load_grpo_corpus
from scripts.audit_verifier_repair_shortcuts import audit
from scripts.build_verifier_repair_grpo_corpus import _TEMPLATES, _prepare_variant


def test_implicit_counterfactual_validation_rejects_shallow_shortcuts():
    source = load_grpo_corpus(Path("ml/agentic/datasets/native-react-grpo-v1/train.jsonl"))[0]
    rows = [
        _prepare_variant(
            source,
            split="validation",
            template=template,
            ordinal=index,
        )
        for index, template in enumerate(_TEMPLATES["validation"])
    ]
    report = audit(rows)

    assert report["passed"] is True
    assert report["target_counts"] == {
        "abort": 3,
        "propose_tradeoff": 3,
        "retry_solve": 3,
    }
    assert report["accuracies"]["violation_code_majority"] <= 1 / 3 + 1e-6
    assert report["accuracies"]["contract_majority"] <= 1 / 3 + 1e-6
    assert report["strong_baselines"]["deterministic_contract_router"] == 1.0

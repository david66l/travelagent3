"""Explicit source-equal SFT objective, independent of target count per source."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

SOURCE_EQUAL_LOSS_CONTRACT = "mean-source-mean-example-mean-completion-token.v1"


def source_equal_weights(
    rows: list[dict[str, Any]], lineage: list[dict[str, Any]], *, split: str,
) -> tuple[list[float], dict[str, Any]]:
    if not rows:
        raise ValueError("source weighting needs nonempty examples")
    index = {}
    source_splits = defaultdict(set)
    for item in lineage:
        key = item["record_id"]
        if key in index:
            raise ValueError("duplicate lineage record id")
        index[key] = item
        source_splits[item["source_state_id"]].add(item["split"])
    if any(len(splits) != 1 for splits in source_splits.values()):
        raise ValueError("source appears in multiple splits")
    ids = [item["example_id"] for item in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate training example id")
    sources = []
    for key in ids:
        if key not in index or index[key]["split"] != split:
            raise ValueError("example has missing or wrong-split lineage")
        source = index[key]["source_state_id"]
        if not source:
            raise ValueError("empty source id")
        sources.append(source)
    counts = Counter(sources)
    if max(counts.values()) > 2:
        raise ValueError("H006 permits at most two targets per source")
    scale = len(rows) / len(counts)
    weights = [scale / counts[source] for source in sources]
    # Uniform row traversal plus this OUTSIDE-the-token-mean multiplier yields
    # (1/S) sum_s (1/n_s) sum_i mean_completion_CE_i, with no replacement.
    totals = defaultdict(float)
    for source, weight in zip(sources, weights, strict=True):
        totals[source] += weight
    report = {
        "contract": SOURCE_EQUAL_LOSS_CONTRACT,
        "examples": len(rows), "sources": len(counts),
        "max_examples_per_source": max(counts.values()),
        "weight_min": min(weights), "weight_max": max(weights),
        "total_example_weight": sum(weights),
        "source_total_weight_min": min(totals.values()),
        "source_total_weight_max": max(totals.values()),
        "verified_equal_source_mass": max(totals.values()) - min(totals.values()) < 1e-9,
    }
    return weights, report


def completion_weighted_loss(logits, labels, token_weights, source_weights=None):
    """Causal CE; never divide source weights away in a one-example batch."""
    import torch

    shifted = logits[..., :-1, :].contiguous()
    targets = labels[..., 1:].to(shifted.device).contiguous()
    weights = token_weights[..., 1:].to(shifted.device)
    weights = weights * (targets != -100)
    losses = torch.nn.functional.cross_entropy(
        shifted.view(-1, shifted.size(-1)), targets.view(-1),
        reduction="none", ignore_index=-100,
    ).view_as(targets)
    if source_weights is None:
        return (losses * weights).sum() / weights.sum().clamp_min(1.0)
    if source_weights.numel() != targets.shape[0] or torch.any(source_weights <= 0):
        raise ValueError("invalid per-example source weights")
    denominators = weights.sum(dim=-1)
    if torch.any(denominators <= 0):
        raise ValueError("source-equal loss cannot include an empty completion")
    example_losses = (losses * weights).sum(dim=-1) / denominators
    return (example_losses * source_weights.to(example_losses.device)).mean()

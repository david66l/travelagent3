# Agent Policy dataset pipeline

Core SFT data is generated from finalized `AgentEpisode` records, not from the
legacy `ml/training/data/train.jsonl` prompt/completion demo.

Build audited splits:

```powershell
.\backend\.venv\Scripts\python.exe scripts\build_sft_dataset.py `
  --input ml\agentic\datasets\candidates.jsonl `
  --output-dir ml\agentic\datasets\build\sft-v1
```

Each input line is an `EpisodeCandidate` containing source metadata and a
finalized, hash-verified episode. The builder writes:

- `train.jsonl`, `validation.jsonl`, and `test.jsonl`: policy-decision examples;
- `reviews.jsonl`: accepted/rejected decision and machine-readable reasons;
- `manifest.json`: content-derived dataset version, counts, provenance and split audit.

The builder rejects unfinalized/tampered episodes, PII, unauthorized production
data, invalid or ungrounded arguments, missing tool observations, repeated
successful calls and outcomes that are not a validated plan, necessary
clarification, or safe termination.

Do not commit raw production candidates or generated dataset builds. Register
the manifest and approved immutable dataset in the experiment artifact store.

After the production-size data gate passes, run real QLoRA/SFT with:

```powershell
uv pip install -e ".\backend[agentic-training]"
.\backend\.venv\Scripts\python.exe ml\agentic\training\train_sft.py `
  --dataset-dir ml\agentic\datasets\build\sft-v1 `
  --output-dir ml\agentic\artifacts\agent-policy-sft-v1
```

Use `--preflight-only` on a CPU development machine. `--allow-small-dataset`
exists only for a local smoke test and must not be used for reported results.

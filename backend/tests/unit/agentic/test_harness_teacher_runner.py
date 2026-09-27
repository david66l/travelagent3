"""Resume guards must prevent duplicate spending and mixed evidence."""
from argparse import Namespace
import json

import pytest

from scripts import evaluate_harness_teacher as runner


def setup_run(tmp_path, monkeypatch):
    monkeypatch.setenv("ZAI_API_KEY", "test-only")
    def no_request(*args, **kwargs):
        raise AssertionError("Resume guard must run before opening any API client")
    monkeypatch.setattr(runner, "GLMToolClient", no_request)
    args = Namespace(output=tmp_path, model="glm-5.3-flash", base_url="https://open.bigmodel.cn/api/paas/v4",
                     max_tokens=8192, episode_timeout=600, request_timeout=180, concurrency=4, case=["basic-01"])
    runner.base.dump(tmp_path/"cases.json", runner.base.build_cases())
    runner.base.dump(tmp_path/"manifest.json", runner.manifest_for(args, tmp_path/"cases.json"))
    return args


@pytest.mark.asyncio
async def test_resume_refuses_incompatible_model_without_network(tmp_path, monkeypatch):
    args = setup_run(tmp_path, monkeypatch)
    args.model = "different-model"
    with pytest.raises(AssertionError, match="Configuration/source changed"):
        await runner.main(args)


@pytest.mark.asyncio
async def test_resume_refuses_partial_episode_without_network(tmp_path, monkeypatch):
    args = setup_run(tmp_path, monkeypatch)
    target = tmp_path/"basic-01"
    target.mkdir()
    (target/"model-calls.jsonl").write_text('{}\n')
    with pytest.raises(RuntimeError, match="Interrupted evidence"):
        await runner.main(args)


@pytest.mark.asyncio
async def test_resume_does_not_hide_previously_recorded_provider_failure(tmp_path, monkeypatch):
    args = setup_run(tmp_path, monkeypatch)
    target = tmp_path/"basic-01"
    target.mkdir()
    summary = {"case": "basic-01", "provider_failures": 1}
    runner.base.dump(target/"summary.json", summary)
    with pytest.raises(RuntimeError, match="Provider failure"):
        await runner.main(args)
    assert json.loads((target/"summary.json").read_text()) == summary

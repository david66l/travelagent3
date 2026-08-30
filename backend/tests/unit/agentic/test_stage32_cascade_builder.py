import importlib.util
import json
from pathlib import Path

import pytest

from agentic.corpus_generation import CurriculumTeacherPolicy, build_curriculum_case
from agentic.distillation import build_teacher_candidate
from agentic.environment import TravelAgentEnvironment
from agentic.policy import ControllerFirstPolicy


SCRIPT = Path(__file__).resolve().parents[4] / "scripts" / "build_stage32_cascade_distillation.py"
SPEC = importlib.util.spec_from_file_location("build_stage32_cascade_distillation", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


@pytest.mark.asyncio
async def test_builder_exports_auditable_cascade_dataset(tmp_path: Path):
    task, snapshot = build_curriculum_case(6)
    rows = []
    for sample in range(2):
        rollout = await TravelAgentEnvironment(task, snapshot).rollout(
            ControllerFirstPolicy(CurriculumTeacherPolicy())
        )
        candidate = build_teacher_candidate(rollout, family="search", sample_index=sample)
        candidate.score.successful = True
        candidate.score.gate_status = "passed"
        candidate.score.hard_pass = True
        rows.append(candidate)

    teacher_files = []
    for index, teacher in enumerate(("qwen3-4b", "qwen3-8b")):
        path = tmp_path / f"{teacher}.jsonl"
        path.write_text(rows[index].model_dump_json() + "\n", encoding="utf-8")
        teacher_files.append(path)
    forbidden = tmp_path / "forbidden.jsonl"
    forbidden.write_text(
        json.dumps({"natural_request": "帮我编写一份完全无关的数据库迁移方案"}, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )
    config = {
        "teacher_runs": [
            {
                "teacher_id": "qwen3-4b",
                "model": "qwen3-4b",
                "checkpoint": "stage28-dpo",
                "tier": "student_teacher",
                "run_id": "pilot-4b",
                "candidates_file": str(teacher_files[0]),
            },
            {
                "teacher_id": "qwen3-8b",
                "model": "qwen3-8b",
                "checkpoint": "base",
                "tier": "complex_teacher",
                "run_id": "pilot-8b",
                "candidates_file": str(teacher_files[1]),
            },
        ],
        "forbidden_corpora": [str(forbidden)],
        "required_families": ["search"],
        "pilot_thresholds": {
            "minimum_selected_tasks": 1,
            "minimum_selected_per_family": 1,
            "minimum_student_teacher_chosen_share": 1.0,
        },
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    output_dir = tmp_path / "output"

    manifest = MODULE.build(config_path, output_dir)

    assert manifest["status"] == "passed"
    assert manifest["selected_tasks"] == 1
    assert manifest["chosen_teacher_counts"] == {"qwen3-4b": 1}
    assert manifest["forbidden_evaluation_contamination"]["passed"] is True
    assert manifest["forbidden_evaluation_contamination"]["quarantined_tasks"] == 0
    assert manifest["sft"]["examples"] >= 1
    assert (output_dir / "sft" / "manifest.json").is_file()
    assert (output_dir / "preferences" / "manifest.json").is_file()

    exact_request = rows[0].rollout.episode.initial_state["goal"]["original_request"]
    forbidden.write_text(
        json.dumps(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": json.dumps(
                            {"original_request": exact_request}, ensure_ascii=False
                        ),
                    }
                ]
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    rejected = MODULE.build(config_path, tmp_path / "contaminated-output")

    assert rejected["status"] == "rejected"
    assert rejected["selected_tasks"] == 0
    assert rejected["forbidden_evaluation_contamination"]["quarantined_tasks"] == 1
    assert rejected["forbidden_evaluation_contamination"]["retained_exact_matches"] == 0

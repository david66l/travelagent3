"""Audit and freeze the Stage 25 evidence before starting external evaluation.

The generated report never copies secret values or repository file contents. It
records only redacted rule hits, aggregate Git state and SHA-256 inventories.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.build_stage24_final_report import build as build_stage24
from scripts.build_stage25_showcase import build as build_stage25


FROZEN_BASELINE = {
    "strict_success": "516/516",
    "rollout_success": "172/172",
    "mean_reward": 0.965842,
    "teacher_call_share": 0.02325581,
    "token_reduction_vs_all_teacher_percent": 28.52,
    "latency_reduction_vs_all_teacher_percent": 47.87,
}

DEFAULT_ARTIFACTS = (
    ("report", "ml/agentic/reports/stage24-final-evaluation-v1/report.json"),
    ("report", "ml/agentic/reports/stage24-final-evaluation-v1/REPORT.md"),
    ("report", "ml/agentic/reports/stage25-showcase-v1/showcase.json"),
    ("report", "ml/agentic/reports/stage25-showcase-v1/SHOWCASE.md"),
    ("release_archive", "stage24-final-evaluation-v1.tgz"),
    ("release_archive", "stage25-final-showcase-v1.tgz"),
    ("model", "ml/agentic/checkpoints/qwen3-4b-stage21-sft-balanced-formal-v1/adapter_model.safetensors"),
    ("model", "ml/agentic/checkpoints/qwen3-4b-stage21-sft-balanced-formal-v1/adapter_config.json"),
    ("model", "ml/agentic/checkpoints/qwen3-4b-stage21-sft-balanced-formal-v1/training_report.json"),
    ("model", "ml/agentic/checkpoints/qwen3-4b-stage22-dpo-balanced-formal-v1/adapter_model.safetensors"),
    ("model", "ml/agentic/checkpoints/qwen3-4b-stage22-dpo-balanced-formal-v1/adapter_config.json"),
    ("model", "ml/agentic/checkpoints/qwen3-4b-stage22-dpo-balanced-formal-v1/training_report.json"),
    ("dataset", "ml/agentic/datasets/qwen3-stage19-holdout-v1/manifest.json"),
    ("dataset", "ml/agentic/datasets/qwen3-stage20-teacher-sft-reverified-v3/sft/manifest.json"),
    ("dataset", "ml/agentic/datasets/qwen3-stage21-student-sft-balanced-v1/manifest.json"),
    ("dataset", "ml/agentic/datasets/qwen3-stage22-preferences-balanced-v1/manifest.json"),
)

TEXT_EXTENSIONS = {
    ".cfg", ".conf", ".css", ".csv", ".env", ".example", ".html", ".ini",
    ".js", ".json", ".jsonl", ".md", ".mjs", ".py", ".sh", ".toml",
    ".ts", ".tsx", ".txt", ".yaml", ".yml",
}

SECRET_PATTERNS = {
    "private_key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----"),
    "openai_api_key": re.compile(r"\bsk-(?:proj-|svcacct-)?[A-Za-z0-9_-]{20,}\b"),
    "github_token": re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}\b"),
    "huggingface_token": re.compile(r"\bhf_[A-Za-z0-9]{30,}\b"),
    "aws_access_key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
}

# Git uses POSIX extended regular expressions rather than Python syntax. These
# patterns deliberately omit word-boundary escapes for cross-platform parity.
GIT_SECRET_PATTERNS = {
    "private_key": r"BEGIN (RSA |EC |OPENSSH |DSA )?PRIVATE KEY",
    "openai_api_key": r"sk-(proj-|svcacct-)?[A-Za-z0-9_-]{20,}",
    "github_token": r"gh[pousr]_[A-Za-z0-9]{30,}",
    "huggingface_token": r"hf_[A-Za-z0-9]{30,}",
    "aws_access_key": r"AKIA[0-9A-Z]{16}",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def run_git(repo_root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return result.stdout


def summarize_git_porcelain(status_text: str) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    entries = [line for line in status_text.splitlines() if line]
    for line in entries:
        code = line[:2]
        if code == "??":
            counts["untracked"] += 1
            continue
        if code[0] != " ":
            counts["index_changes"] += 1
        if code[1] != " ":
            counts["worktree_changes"] += 1
    return {
        "clean": not entries,
        "entries": len(entries),
        "index_changes": counts["index_changes"],
        "worktree_changes": counts["worktree_changes"],
        "untracked": counts["untracked"],
        "privacy_note": "Only aggregate counts are stored; paths and diff contents are omitted.",
    }


def git_state(repo_root: Path) -> dict[str, Any]:
    summary = summarize_git_porcelain(run_git(repo_root, "status", "--porcelain=v1"))
    summary.update(
        {
            "head": run_git(repo_root, "rev-parse", "HEAD").strip(),
            "branch": run_git(repo_root, "rev-parse", "--abbrev-ref", "HEAD").strip(),
        }
    )
    return summary


def iter_release_files(repo_root: Path) -> Iterable[Path]:
    output = run_git(repo_root, "ls-files", "-z", "--cached", "--others", "--exclude-standard")
    for relative in output.split("\0"):
        if not relative:
            continue
        path = repo_root / relative
        if (
            path.is_file()
            and path.suffix.lower() in TEXT_EXTENSIONS
            and path.stat().st_size <= 2 * 1024 * 1024
        ):
            yield path


def parse_history_scan(output: str) -> tuple[list[str], list[str]]:
    commits = set()
    paths = set()
    for line in output.splitlines():
        if line.startswith("commit:"):
            commits.add(line.removeprefix("commit:"))
        elif line.strip():
            paths.add(line.strip())
    return sorted(commits), sorted(paths)


def scan_secret_history(repo_root: Path) -> list[dict[str, Any]]:
    findings = []
    for rule, pattern in GIT_SECRET_PATTERNS.items():
        output = run_git(
            repo_root,
            "log",
            "--all",
            "--extended-regexp",
            "-G",
            pattern,
            "--format=commit:%H",
            "--name-only",
            "--",
            ".",
        )
        commits, paths = parse_history_scan(output)
        if commits:
            findings.append(
                {
                    "rule": rule,
                    "commit_count": len(commits),
                    "paths": paths,
                    "match": "[REDACTED]",
                }
            )
    return findings


def scan_secrets(repo_root: Path, paths: Iterable[Path] | None = None) -> dict[str, Any]:
    findings: list[dict[str, Any]] = []
    scanned_files = 0
    scanned_bytes = 0
    explicit_paths = paths is not None
    for path in paths if explicit_paths else iter_release_files(repo_root):
        try:
            content = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        scanned_files += 1
        scanned_bytes += path.stat().st_size
        for line_number, line in enumerate(content.splitlines(), start=1):
            for rule, pattern in SECRET_PATTERNS.items():
                if pattern.search(line):
                    findings.append(
                        {
                            "path": path.relative_to(repo_root).as_posix(),
                            "line": line_number,
                            "rule": rule,
                            "match": "[REDACTED]",
                        }
                    )
    history_findings = [] if explicit_paths else scan_secret_history(repo_root)
    return {
        "passed": not findings and not history_findings,
        "scanned_files": scanned_files,
        "scanned_bytes": scanned_bytes,
        "findings": findings,
        "history_findings": history_findings,
        "privacy_note": "Matched values are never written to the report.",
    }


def validate_baseline(stage24: dict[str, Any], stage25: dict[str, Any]) -> dict[str, Any]:
    actual24 = stage24.get("headline", {})
    metrics25 = stage25.get("headline_metrics", {})
    actual25 = {
        "strict_success": metrics25.get("strict_tool_decisions"),
        "rollout_success": metrics25.get("multi_turn_tasks"),
        "mean_reward": metrics25.get("mean_reward"),
        "teacher_call_share": metrics25.get("teacher_task_share"),
        "token_reduction_vs_all_teacher_percent": metrics25.get(
            "token_reduction_vs_all_teacher_percent"
        ),
        "latency_reduction_vs_all_teacher_percent": metrics25.get(
            "model_latency_reduction_vs_all_teacher_percent"
        ),
    }
    checks = {
        "stage24_schema": stage24.get("schema_version")
        == "travel-agent-stage24-final-evaluation.v1",
        "stage24_status": stage24.get("status") == "passed",
        "stage24_metrics": all(actual24.get(key) == value for key, value in FROZEN_BASELINE.items()),
        "stage25_schema": stage25.get("schema_version") == "travel-agent-stage25-showcase.v1",
        "stage25_status": stage25.get("status") == "ready",
        "stage25_metrics": actual25 == FROZEN_BASELINE,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "expected": FROZEN_BASELINE,
        "stage24_actual": {key: actual24.get(key) for key in FROZEN_BASELINE},
        "stage25_actual": actual25,
    }


def inventory_artifacts(repo_root: Path) -> dict[str, Any]:
    artifacts = []
    for category, relative in DEFAULT_ARTIFACTS:
        path = repo_root / relative
        row: dict[str, Any] = {"category": category, "path": relative, "exists": path.is_file()}
        if path.is_file():
            row.update({"bytes": path.stat().st_size, "sha256": sha256_file(path)})
        artifacts.append(row)
    return {
        "complete": all(row["exists"] for row in artifacts),
        "count": len(artifacts),
        "artifacts": artifacts,
    }


def verify_regeneration(repo_root: Path, frozen24: dict[str, Any], frozen25: dict[str, Any]) -> dict[str, Any]:
    previous_cwd = Path.cwd()
    try:
        # The frozen builders intentionally persist repository-relative source
        # paths. Rebuild from the repository root so path formatting itself is
        # covered by the exact comparison.
        os.chdir(repo_root)
        regenerated24 = build_stage24(
            Path("ml/agentic/reports"), Path("ml/agentic/checkpoints")
        )
        regenerated25 = build_stage25(Path("ml/agentic/reports"))
    finally:
        os.chdir(previous_cwd)
    checks = {
        "stage24_json_match": canonical_hash(regenerated24) == canonical_hash(frozen24),
        "stage25_json_match": canonical_hash(regenerated25) == canonical_hash(frozen25),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "stage24_canonical_sha256": canonical_hash(regenerated24),
        "stage25_canonical_sha256": canonical_hash(regenerated25),
    }


def build_report(repo_root: Path) -> dict[str, Any]:
    stage24_path = repo_root / "ml/agentic/reports/stage24-final-evaluation-v1/report.json"
    stage25_path = repo_root / "ml/agentic/reports/stage25-showcase-v1/showcase.json"
    frozen24 = read_json(stage24_path)
    frozen25 = read_json(stage25_path)

    baseline = validate_baseline(frozen24, frozen25)
    regeneration = verify_regeneration(repo_root, frozen24, frozen25)
    inventory = inventory_artifacts(repo_root)
    secrets = scan_secrets(repo_root)
    repository = git_state(repo_root)
    gates = {
        "frozen_baseline_valid": baseline["passed"],
        "summary_regeneration_exact": regeneration["passed"],
        "artifact_inventory_complete": inventory["complete"],
        "secret_scan_clean": secrets["passed"],
        "git_worktree_clean": repository["clean"],
    }
    status = "passed" if all(gates.values()) else "blocked"
    next_actions = []
    if not secrets["passed"]:
        next_actions.append("Review redacted secret-scan hits and rotate any real credential before release.")
    if not repository["clean"]:
        next_actions.append("Review and commit/stash the existing worktree; do not auto-commit unrelated changes.")
    if not baseline["passed"] or not regeneration["passed"]:
        next_actions.append("Resolve frozen-report drift before creating the Phase 26 release candidate.")
    if not inventory["complete"]:
        next_actions.append("Restore missing frozen artifacts before tagging the release candidate.")
    if status == "passed":
        next_actions.append("Create the Stage 25 release-candidate tag and begin the external benchmark.")
    return {
        "schema_version": "travel-agent-stage26-evidence-freeze.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "gates": gates,
        "baseline": baseline,
        "regeneration": regeneration,
        "repository": repository,
        "secret_scan": secrets,
        "inventory": inventory,
        "next_actions": next_actions,
        "scope": (
            "This is a release-readiness audit. It does not upgrade the internal Stage 25 "
            "benchmark into an external benchmark."
        ),
    }


def render_markdown(report: dict[str, Any]) -> str:
    baseline = report["baseline"]["expected"]
    lines = [
        "# TravelAgent 阶段 26：仓库与证据冻结审计",
        "",
        f"> 状态：`{report['status']}`  ",
        f"> Git：`{report['repository']['branch']}` / `{report['repository']['head'][:12]}`  ",
        "> 本报告只冻结阶段 25 工程证据，不把内部评测包装成外部 Benchmark。",
        "",
        "## 冻结基线",
        "",
        "| 指标 | 冻结值 |",
        "|---|---:|",
        f"| 严格工具决策 | {baseline['strict_success']} |",
        f"| 完整多轮任务 | {baseline['rollout_success']} |",
        f"| 平均 Reward | {baseline['mean_reward']:.6f} |",
        f"| 8B 教师任务占比 | {baseline['teacher_call_share'] * 100:.2f}% |",
        f"| Token 降幅 | {baseline['token_reduction_vs_all_teacher_percent']:.2f}% |",
        f"| 模型延迟降幅 | {baseline['latency_reduction_vs_all_teacher_percent']:.2f}% |",
        "",
        "## 发布门",
        "",
        "| 检查 | 结果 |",
        "|---|---|",
    ]
    for gate, passed in report["gates"].items():
        lines.append(f"| {gate} | {'PASS' if passed else 'BLOCKED'} |")
    git = report["repository"]
    scan = report["secret_scan"]
    lines.extend(
        [
            "",
            "## 仓库审计摘要",
            "",
            f"- Git 状态条目：{git['entries']}（index {git['index_changes']} / worktree {git['worktree_changes']} / untracked {git['untracked']}）。",
            f"- Secret scan：{scan['scanned_files']} 个文本文件、当前树 {len(scan['findings'])} 个脱敏命中、历史 {len(scan['history_findings'])} 类脱敏命中。",
            f"- 哈希清单：{report['inventory']['count']} 个模型、数据、报告与归档产物。",
            f"- Stage 24/25 摘要重建：{'字节语义一致' if report['regeneration']['passed'] else '存在漂移'}。",
            "",
            "## SHA-256 清单",
            "",
            "| 类型 | 路径 | 大小 | SHA-256 |",
            "|---|---|---:|---|",
        ]
    )
    for item in report["inventory"]["artifacts"]:
        if item["exists"]:
            lines.append(
                f"| {item['category']} | `{item['path']}` | {item['bytes']} | `{item['sha256']}` |"
            )
        else:
            lines.append(f"| {item['category']} | `{item['path']}` | — | MISSING |")
    lines.extend(["", "## 下一步", ""])
    lines.extend(f"- {item}" for item in report["next_actions"])
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("ml/agentic/reports/stage26-evidence-freeze-v1"),
    )
    args = parser.parse_args()
    repo_root = args.repo_root.resolve()
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = repo_root / output_dir
    report = build_report(repo_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "REPORT.md").write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps({"status": report["status"], "gates": report["gates"]}, indent=2))
    return 0 if report["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())

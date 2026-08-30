"""Reject high-confidence credentials in files staged for Git commit.

The scanner reports only a rule, path and line number. Matched values are never
printed, which keeps CI and local hook logs safe.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path
from typing import Any


SECRET_PATTERNS = {
    "private_key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----"),
    "openai_compatible_api_key": re.compile(
        r"\bsk-(?:proj-|svcacct-)?[A-Za-z0-9_-]{20,}\b"
    ),
    "github_token": re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}\b"),
    "huggingface_token": re.compile(r"\bhf_[A-Za-z0-9]{30,}\b"),
    "aws_access_key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
}

TEXT_EXTENSIONS = {
    ".cfg", ".conf", ".css", ".csv", ".env", ".example", ".html", ".ini",
    ".js", ".json", ".jsonl", ".md", ".mjs", ".py", ".sh", ".toml",
    ".ts", ".tsx", ".txt", ".yaml", ".yml",
}
MAX_SCAN_BYTES = 2 * 1024 * 1024


def scan_text(path: str, content: str) -> list[dict[str, Any]]:
    findings = []
    for line_number, line in enumerate(content.splitlines(), start=1):
        for rule, pattern in SECRET_PATTERNS.items():
            if pattern.search(line):
                findings.append(
                    {
                        "path": path,
                        "line": line_number,
                        "rule": rule,
                        "match": "[REDACTED]",
                    }
                )
    return findings


def is_text_candidate(path: str) -> bool:
    candidate = Path(path)
    return candidate.name.startswith(".env") or candidate.suffix.lower() in TEXT_EXTENSIONS


def git_bytes(repo_root: Path, *args: str) -> bytes:
    return subprocess.run(
        ["git", *args], cwd=repo_root, check=True, capture_output=True
    ).stdout


def staged_paths(repo_root: Path) -> list[str]:
    output = git_bytes(
        repo_root,
        "diff",
        "--cached",
        "--name-only",
        "--diff-filter=ACMR",
        "-z",
    )
    return [item.decode("utf-8") for item in output.split(b"\0") if item]


def tracked_paths(repo_root: Path) -> list[str]:
    output = git_bytes(repo_root, "ls-files", "-z")
    return [item.decode("utf-8") for item in output.split(b"\0") if item]


def scan_staged(repo_root: Path) -> dict[str, Any]:
    return scan_paths(
        repo_root,
        staged_paths(repo_root),
        source="staged",
        read_from_index=True,
    )


def scan_paths(
    repo_root: Path,
    paths: list[str],
    *,
    source: str,
    read_from_index: bool = False,
) -> dict[str, Any]:
    findings = []
    scanned = 0
    skipped_large = 0
    for path in paths:
        if not is_text_candidate(path):
            continue
        candidate = repo_root / path
        if read_from_index:
            size = int(git_bytes(repo_root, "cat-file", "-s", f":{path}").decode().strip())
        else:
            if not candidate.is_file():
                continue
            size = candidate.stat().st_size
        if size > MAX_SCAN_BYTES:
            skipped_large += 1
            continue
        blob = (
            git_bytes(repo_root, "show", f":{path}")
            if read_from_index
            else candidate.read_bytes()
        )
        try:
            content = blob.decode("utf-8")
        except UnicodeDecodeError:
            continue
        scanned += 1
        findings.extend(scan_text(path, content))
    return {
        "passed": not findings,
        "scanned_files": scanned,
        "skipped_large_files": skipped_large,
        "findings": findings,
        "source": source,
        "privacy_note": "Credential values are never printed.",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tracked",
        action="store_true",
        help="scan every Git-tracked text file (intended for CI)",
    )
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    report = (
        scan_paths(repo_root, tracked_paths(repo_root), source="tracked")
        if args.tracked
        else scan_staged(repo_root)
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

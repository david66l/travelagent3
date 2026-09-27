"""Run the H007 full-Agent-Loop evaluation entirely on the cloud host.

The controller is intentionally locked to one run id.  It starts an isolated
vLLM process group, records immutable runtime evidence, executes every stage
once, validates report completeness, and always cleans up the server.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen


RUN_ID = "h007-cloud-full-agent-loop-v1-20260905-r4"
REPO_ROOT = Path("/root/autodl-tmp/TravelAgent2-h005-eval-20260903")
RUN_DIR = REPO_ROOT / "artifacts/native-react-posttraining" / RUN_ID
EVAL_PYTHON = Path(
    "/root/autodl-tmp/venvs/travelagent-eval-h007-20260905/bin/python"
)
SERVE_PYTHON = Path("/root/miniconda3/bin/python")
ADAPTER_DIR = (
    REPO_ROOT
    / "artifacts/native-react-posttraining/"
    "h007-targeted-corrective-sft-v1-seed20260904"
)
BASE_MODEL_DIR = Path("/root/autodl-tmp/models/Qwen3-1.7B")
TEMPLATE_PATH = (
    REPO_ROOT
    / "backend/src/agentic/templates/qwen3_agent_prefix_preserving_v1.jinja"
)
POLICY_MODEL = "travel-h007-final"
POLICY_BACKEND = "http://127.0.0.1:8000/v1"
INTENT_MODEL = "deepseek-v4-flash"
INTENT_BACKEND = "cloud-openai-compatible"
EXPECTED_ADAPTER_MANIFEST = (
    "45be85d4d8c233eb17c4083d7000487b412c675a8b28049f15469d1317da0193"
)
EXPECTED_BASE_MANIFEST = (
    "aa81aeaaee547ef68a6a993763cb3ef38a8223eafe2f2eeef139a47f2715110d"
)
EXPECTED_TEMPLATE_SHA256 = (
    "dfe4e379b6439a9f01e881660c5f1ea57cd8026a8831553492723b38d15c9e63"
)
EXPECTED_SERVING_VERSIONS = {
    "peft": "0.20.0",
    "torch": "2.6.0",
    "transformers": "4.57.6",
    "vllm": "0.8.5.post1",
}
MONITOR_INTERVAL_SECONDS = 30
STAGE_TIMEOUT_SECONDS = 1800
SERVER_TIMEOUT_SECONDS = 600

RUNTIME_PATHS = (
    REPO_ROOT / "backend/src",
    REPO_ROOT / "scripts/evaluate_full_agent_loop.py",
    REPO_ROOT / "scripts/evaluate_full_agent_loop_recovery.py",
    REPO_ROOT / "evals/promotion-val-v1",
)


def utc_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def append_event(event: str, **data: Any) -> None:
    row = {"timestamp": utc_now(), "event": event, **data}
    with (RUN_DIR / "monitor.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def root_file_manifest(path: Path) -> dict[str, Any]:
    rows: list[str] = []
    files: list[dict[str, Any]] = []
    for item in sorted((p for p in path.iterdir() if p.is_file()), key=lambda p: p.name):
        digest = sha256_file(item)
        rows.append(f"{digest}  ./{item.name}\n")
        files.append({"path": item.name, "sha256": digest, "size": item.stat().st_size})
    combined = hashlib.sha256("".join(rows).encode()).hexdigest()
    return {"path": str(path), "combined_sha256": combined, "files": files}


def runtime_snapshot(phase: str) -> dict[str, Any]:
    files: list[dict[str, Any]] = []
    for root in RUNTIME_PATHS:
        candidates = [root] if root.is_file() else sorted(root.rglob("*"))
        for path in candidates:
            if not path.is_file() or path.suffix in {".pyc", ".pyo"}:
                continue
            if "__pycache__" in path.parts:
                continue
            files.append(
                {
                    "path": path.relative_to(REPO_ROOT).as_posix(),
                    "sha256": sha256_file(path),
                    "size": path.stat().st_size,
                }
            )
    files.sort(key=lambda row: row["path"])
    combined_bytes = "".join(
        f"{row['sha256']}  {row['path']}\n" for row in files
    ).encode()
    return {
        "schema_version": "h007-cloud-runtime-snapshot.v1",
        "kind": "runtime_snapshot",
        "phase": phase,
        "run_id": RUN_ID,
        "captured_at": utc_now(),
        "python": sys.version,
        "python_executable": sys.executable,
        "combined_sha256": hashlib.sha256(combined_bytes).hexdigest(),
        "files": files,
    }


def get_gpu_row() -> dict[str, Any]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    parts = [part.strip() for part in result.stdout.strip().split(",")]
    if len(parts) != 5:
        raise RuntimeError(f"unexpected nvidia-smi output: {result.stdout!r}")
    return {
        "name": parts[0],
        "memory_total_mib": int(parts[1]),
        "memory_used_mib": int(parts[2]),
        "memory_free_mib": int(parts[3]),
        "utilization_percent": int(parts[4]),
    }


def serving_package_versions() -> dict[str, str]:
    code = (
        "import importlib.metadata as m,json; "
        "names=['peft','torch','transformers','vllm']; "
        "print(json.dumps({name:m.version(name) for name in names},sort_keys=True))"
    )
    result = subprocess.run(
        [str(SERVE_PYTHON), "-c", code],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return json.loads(result.stdout)


def verify_port_free(port: int) -> None:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", port))


def request_json(
    url: str,
    *,
    body: dict[str, Any] | None = None,
    timeout: float = 10,
) -> dict[str, Any]:
    data = None if body is None else json.dumps(body).encode()
    request = Request(url, data=data)
    if data is not None:
        request.add_header("Content-Type", "application/json")
    with urlopen(request, timeout=timeout) as response:
        payload = response.read()
    return json.loads(payload) if payload else {}


def rss_bytes(pid: int) -> int:
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    except (FileNotFoundError, ProcessLookupError):
        pass
    return 0


def stop_process_group(process: subprocess.Popen[Any], *, reason: str) -> None:
    if process.poll() is not None:
        return
    append_event("PROCESS_GROUP_STOP", name=reason, pid=process.pid)
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=30)
        return
    except subprocess.TimeoutExpired:
        append_event("PROCESS_GROUP_FORCE_STOP", name=reason, pid=process.pid)
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait(timeout=10)


def run_monitored(
    name: str,
    command: list[str],
    *,
    timeout_seconds: int,
    check_seconds: int,
    env: dict[str, str],
) -> dict[str, Any]:
    stdout_path = RUN_DIR / f"{name}.stdout.log"
    stderr_path = RUN_DIR / f"{name}.stderr.log"
    started_wall = time.monotonic()
    started_at = utc_now()
    with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            env=env,
            stdout=stdout,
            stderr=stderr,
            start_new_session=True,
        )
        append_event(
            "PROCESS_STARTED",
            name=name,
            pid=process.pid,
            command=command,
            timeout_seconds=timeout_seconds,
        )
        peak_rss = rss_bytes(process.pid)
        last_size = 0
        last_output_change = time.monotonic()
        stall_advisory_emitted = False
        timed_out = False
        while process.poll() is None:
            elapsed = time.monotonic() - started_wall
            if elapsed >= timeout_seconds:
                timed_out = True
                append_event("HARD_TIMEOUT", name=name, pid=process.pid, elapsed_seconds=elapsed)
                stop_process_group(process, reason=f"hard-timeout:{name}")
                break
            time.sleep(min(check_seconds, max(1, timeout_seconds - elapsed)))
            total_size = sum(
                path.stat().st_size if path.exists() else 0
                for path in (stdout_path, stderr_path)
            )
            if total_size != last_size:
                last_size = total_size
                last_output_change = time.monotonic()
                stall_advisory_emitted = False
            peak_rss = max(peak_rss, rss_bytes(process.pid))
            append_event(
                "PROCESS_CHECK",
                name=name,
                pid=process.pid,
                elapsed_seconds=round(time.monotonic() - started_wall, 3),
                output_bytes=total_size,
                rss_bytes=rss_bytes(process.pid),
            )
            quiet = time.monotonic() - last_output_change
            if quiet >= 90 and not stall_advisory_emitted:
                stall_advisory_emitted = True
                append_event(
                    "OUTPUT_STALL_ADVISORY",
                    name=name,
                    pid=process.pid,
                    seconds_without_output=round(quiet, 3),
                )
        exit_code = 124 if timed_out else int(process.wait())
    result = {
        "name": name,
        "command": command,
        "started_at": started_at,
        "ended_at": utc_now(),
        "elapsed_seconds": round(time.monotonic() - started_wall, 3),
        "exit_code": exit_code,
        "timed_out": timed_out,
        "peak_rss_bytes": peak_rss,
        "stdout": stdout_path.name,
        "stderr": stderr_path.name,
    }
    append_event("PROCESS_FINISHED", **result)
    return result


def optional(record: dict[str, Any], name: str) -> Any:
    return record.get(name)


def validate_stage(report_path: Path, expected_total: int, stage_name: str) -> dict[str, Any]:
    if not report_path.is_file():
        raise RuntimeError(f"{stage_name} did not create {report_path}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    records = report.get("records") or []
    if int((report.get("summary") or {}).get("total", -1)) != expected_total:
        raise RuntimeError(f"{stage_name} summary total is incomplete")
    if len(records) != expected_total:
        raise RuntimeError(f"{stage_name} records are incomplete")
    if report.get("policy_model") != POLICY_MODEL or report.get("policy_backend") != POLICY_BACKEND:
        raise RuntimeError(f"{stage_name} policy identity changed")
    if report.get("intent_model") != INTENT_MODEL:
        raise RuntimeError(f"{stage_name} intent model changed")
    for record in records:
        case_id = record.get("case_id", "unknown")
        if optional(record, "error") or optional(record, "agent_status") == "exception":
            raise RuntimeError(f"{stage_name} infrastructure exception in {case_id}")
        if any(str(item).startswith("EXCEPTION:") for item in (record.get("failures") or [])):
            raise RuntimeError(f"{stage_name} caught exception in {case_id}")
        intent = record.get("intent") or {}
        inference = record.get("intent_inference") or {}
        if intent.get("parse_source") != "llm":
            raise RuntimeError(f"{stage_name} intent fallback in {case_id}")
        if inference.get("model") != INTENT_MODEL or inference.get("backend") != INTENT_BACKEND:
            raise RuntimeError(f"{stage_name} intent backend changed in {case_id}")
        if record.get("runtime_error"):
            raise RuntimeError(f"{stage_name} runtime error in {case_id}")
        if any(bool(value) for value in (record.get("runtime_errors") or {}).values()):
            raise RuntimeError(f"{stage_name} nested runtime error in {case_id}")
    return report


def server_command() -> list[str]:
    return [
        str(SERVE_PYTHON),
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        str(BASE_MODEL_DIR),
        "--served-model-name",
        "travel-qwen3-1.7b-base",
        "--host",
        "127.0.0.1",
        "--port",
        "8000",
        "--chat-template",
        str(TEMPLATE_PATH),
        "--dtype",
        "auto",
        "--max-model-len",
        "6144",
        "--gpu-memory-utilization",
        "0.90",
        "--enable-auto-tool-choice",
        "--tool-call-parser",
        "hermes",
        "--enable-lora",
        "--lora-modules",
        f"{POLICY_MODEL}={ADAPTER_DIR}",
        "--max-lora-rank",
        "16",
        "--max-loras",
        "1",
        "--max-cpu-loras",
        "1",
        "--enable-prefix-caching",
        "--enforce-eager",
        "--disable-log-requests",
    ]


def preflight_environment() -> tuple[dict[str, Any], dict[str, str]]:
    if sys.executable != str(EVAL_PYTHON):
        raise RuntimeError(f"wrong evaluation interpreter: {sys.executable}")
    if not (REPO_ROOT / ".env").is_file():
        raise RuntimeError("cloud .env is missing")
    for path in (ADAPTER_DIR, BASE_MODEL_DIR, TEMPLATE_PATH, SERVE_PYTHON):
        if not path.exists():
            raise RuntimeError(f"required runtime path is missing: {path}")
    verify_port_free(8000)
    adapter = root_file_manifest(ADAPTER_DIR)
    base = root_file_manifest(BASE_MODEL_DIR)
    template_hash = sha256_file(TEMPLATE_PATH)
    serving_versions = serving_package_versions()
    if adapter["combined_sha256"] != EXPECTED_ADAPTER_MANIFEST:
        raise RuntimeError("H007 adapter manifest mismatch")
    if base["combined_sha256"] != EXPECTED_BASE_MANIFEST:
        raise RuntimeError("base-model manifest mismatch")
    if template_hash != EXPECTED_TEMPLATE_SHA256:
        raise RuntimeError("chat-template hash mismatch")
    if serving_versions != EXPECTED_SERVING_VERSIONS:
        raise RuntimeError(f"serving versions changed: {serving_versions}")
    gpu = get_gpu_row()
    if (
        gpu["name"] != "NVIDIA GeForce RTX 4080 SUPER"
        or gpu["memory_total_mib"] < 30000
        or gpu["memory_free_mib"] < 30000
    ):
        raise RuntimeError(f"GPU identity/free memory preflight failed: {gpu}")
    env = os.environ.copy()
    env.update(
        {
            "PYTHONPATH": str(REPO_ROOT / "backend/src"),
            "LOCAL_LLM_ENABLED": "false",
            "LANGSMITH_TRACING": "false",
            "LANGSMITH_TRACING_V2": "false",
            "LANGCHAIN_TRACING": "false",
            "LANGCHAIN_TRACING_V2": "false",
            "PRIVACY_ENCRYPTION_KEY": (
                "4b0a135c8bf3ea9efe077219231444a4318cd6e752bb29e128c8af2365a3d81f"
            ),
        }
    )
    env.pop("LANGSMITH_API_KEY", None)
    env.pop("LANGCHAIN_API_KEY", None)
    evidence = {
        "checked_at": utc_now(),
        "adapter": adapter,
        "base_model": base,
        "template_sha256": template_hash,
        "serving_versions": serving_versions,
        "gpu": gpu,
        "evaluation_python": str(EVAL_PYTHON),
        "external_tracing_enabled": False,
        "api_credentials_logged": False,
    }
    return evidence, env


def wait_for_server(process: subprocess.Popen[Any]) -> None:
    deadline = time.monotonic() + SERVER_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"vLLM exited before health check: {process.returncode}")
        try:
            request_json("http://127.0.0.1:8000/health", timeout=5)
            append_event("SERVER_HEALTHY", pid=process.pid)
            return
        except (URLError, TimeoutError, json.JSONDecodeError):
            time.sleep(5)
    raise RuntimeError("vLLM health check timed out")


def verify_server(process: subprocess.Popen[Any], baseline_gpu_mib: int) -> None:
    models = request_json("http://127.0.0.1:8000/v1/models", timeout=10)
    model_ids = [str(item.get("id")) for item in models.get("data", [])]
    if POLICY_MODEL not in model_ids:
        raise RuntimeError("H007 model alias is missing")
    smoke = request_json(
        "http://127.0.0.1:8000/v1/chat/completions",
        body={
            "model": POLICY_MODEL,
            "messages": [{"role": "user", "content": "请查询北京的历史文化景点。"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "search_pois",
                        "description": "Search points of interest",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "keywords": {"type": "array", "items": {"type": "string"}}
                            },
                            "required": ["keywords"],
                            "additionalProperties": False,
                        },
                    },
                }
            ],
            "tool_choice": "required",
            "temperature": 0.0,
            "max_tokens": 128,
            "chat_template_kwargs": {"enable_thinking": False},
        },
        timeout=60,
    )
    tool_calls = smoke["choices"][0]["message"].get("tool_calls") or []
    if len(tool_calls) != 1 or tool_calls[0]["function"]["name"] != "search_pois":
        raise RuntimeError("native tool-call smoke failed")
    write_json(
        RUN_DIR / "server-health.json",
        {
            "checked_at": utc_now(),
            "health_passed": True,
            "pid": process.pid,
            "pgid": os.getpgid(process.pid),
            "model_ids": model_ids,
            "model_alias_present": True,
            "tool_smoke_passed": True,
            "smoke_response_model": smoke.get("model"),
            "smoke_tool_name": tool_calls[0]["function"]["name"],
            "gpu_baseline_used_mib": baseline_gpu_mib,
            "max_model_len": 6144,
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )


def main() -> None:
    if len(sys.argv) != 2 or sys.argv[1] != RUN_ID:
        raise SystemExit(f"usage: {EVAL_PYTHON} {__file__} {RUN_ID}")
    if RUN_DIR.exists():
        raise SystemExit(f"run directory already exists: {RUN_DIR}")
    RUN_DIR.mkdir(parents=True)
    controller_log = (RUN_DIR / "controller.log").open("a", encoding="utf-8", buffering=1)
    os.dup2(controller_log.fileno(), sys.stdout.fileno())
    os.dup2(controller_log.fileno(), sys.stderr.fileno())
    print(f"[{utc_now()}] starting {RUN_ID}", flush=True)
    server: subprocess.Popen[Any] | None = None
    stage_rows: list[dict[str, Any]] = []
    run_error: str | None = None
    baseline_gpu_mib: int | None = None
    pre_snapshot: dict[str, Any] | None = None
    try:
        evidence, env = preflight_environment()
        baseline_gpu_mib = int(evidence["gpu"]["memory_used_mib"])
        write_json(RUN_DIR / "remote-model-manifest.json", evidence)
        write_json(
            RUN_DIR / "run-contract.json",
            {
                "schema_version": "h007-cloud-full-agent-loop-run-contract.v1",
                "run_id": RUN_ID,
                "scope": "formal same-stack full-Agent-Loop diagnostic; no sealed set opened",
                "policy_model": POLICY_MODEL,
                "policy_backend": POLICY_BACKEND,
                "intent_model": INTENT_MODEL,
                "intent_backend": INTENT_BACKEND,
                "execution_location": "cloud",
                "stage_timeout_seconds": STAGE_TIMEOUT_SECONDS,
                "monitor_interval_seconds": MONITOR_INTERVAL_SECONDS,
                "auto_retry": False,
                "external_tracing_enabled": False,
                "promotion_val_opened": False,
                "sealed_160_opened": False,
                "production_promotion_eligible": False,
            },
        )
        pre_snapshot = runtime_snapshot("pre")
        write_json(RUN_DIR / "runtime-pre.json", pre_snapshot)

        tests = [
            str(EVAL_PYTHON),
            "-m",
            "pytest",
            "-p",
            "no:cacheprovider",
            "-W",
            "error",
            "backend/tests/unit/evaluation/test_full_agent_loop_benchmark.py",
            "backend/tests/unit/evaluation/test_full_agent_loop_recovery.py",
            "backend/tests/unit/evaluation/test_full_agent_loop_report.py",
            "backend/tests/unit/evaluation/test_h006_full_agent_loop_run_audit.py",
            "backend/tests/unit/evaluation/test_posttraining_promotion_protocol.py",
            "backend/tests/unit/agentic/test_policy.py",
            "backend/tests/unit/agentic/test_loop.py",
            "backend/tests/unit/agentic/test_action_executor.py::test_search_merges_city_knowledge_and_preserves_explicit_must_visits",
            "backend/tests/unit/tools/test_tool_executor.py::test_search_pois_supplements_broad_supply_with_hard_required_entities",
            "backend/tests/unit/tools/test_tool_executor.py::test_amap_required_query_records_rank_one_alias_provenance",
            "backend/tests/unit/core/test_schemas.py",
            "backend/tests/unit/core/test_conversation_turn.py",
            "backend/tests/unit/core/test_langsmith_trace.py",
            "-q",
        ]
        test_result = run_monitored(
            "preflight-tests", tests, timeout_seconds=300, check_seconds=10, env=env
        )
        write_json(RUN_DIR / "test-result.json", test_result)
        if test_result["exit_code"] != 0:
            raise RuntimeError("preflight tests failed")

        stdout = (RUN_DIR / "vllm.stdout.log").open("wb")
        stderr = (RUN_DIR / "vllm.stderr.log").open("wb")
        server = subprocess.Popen(
            server_command(),
            cwd=REPO_ROOT,
            env=env,
            stdout=stdout,
            stderr=stderr,
            start_new_session=True,
        )
        write_json(
            RUN_DIR / "server-process.json",
            {
                "started_at": utc_now(),
                "pid": server.pid,
                "pgid": os.getpgid(server.pid),
                "command": server_command(),
            },
        )
        append_event("SERVER_STARTED", pid=server.pid, pgid=os.getpgid(server.pid))
        wait_for_server(server)
        verify_server(server, baseline_gpu_mib)

        stages = [
            (
                "core-10",
                10,
                [
                    str(EVAL_PYTHON),
                    "scripts/evaluate_full_agent_loop.py",
                    "--output",
                    str(RUN_DIR / "core-10.json"),
                    "--episode-output",
                    str(RUN_DIR / "core-10-episodes.jsonl"),
                    "--rollout-id",
                    f"{RUN_ID}-core",
                    "--suite",
                    "core",
                    "--policy-model",
                    POLICY_MODEL,
                    "--policy-base-url",
                    POLICY_BACKEND,
                    "--policy-temperature",
                    "0.0",
                ],
            ),
            (
                "expanded-20",
                20,
                [
                    str(EVAL_PYTHON),
                    "scripts/evaluate_full_agent_loop.py",
                    "--output",
                    str(RUN_DIR / "expanded-20.json"),
                    "--episode-output",
                    str(RUN_DIR / "expanded-20-episodes.jsonl"),
                    "--rollout-id",
                    f"{RUN_ID}-expanded",
                    "--suite",
                    "expanded",
                    "--policy-model",
                    POLICY_MODEL,
                    "--policy-base-url",
                    POLICY_BACKEND,
                    "--policy-temperature",
                    "0.0",
                ],
            ),
            (
                "recovery-8x4",
                32,
                [
                    str(EVAL_PYTHON),
                    "scripts/evaluate_full_agent_loop_recovery.py",
                    "--output",
                    str(RUN_DIR / "recovery-8x4.json"),
                    "--episode-output",
                    str(RUN_DIR / "recovery-8x4-episodes.jsonl"),
                    "--group-size",
                    "4",
                    "--seed",
                    "421",
                    "--rollout-id",
                    RUN_ID,
                    "--policy-model",
                    POLICY_MODEL,
                    "--policy-base-url",
                    POLICY_BACKEND,
                    "--policy-temperature",
                    "0.8",
                ],
            ),
        ]
        reports: dict[str, dict[str, Any]] = {}
        for name, expected_total, command in stages:
            result = run_monitored(
                name,
                command,
                timeout_seconds=STAGE_TIMEOUT_SECONDS,
                check_seconds=MONITOR_INTERVAL_SECONDS,
                env=env,
            )
            stage_rows.append(result)
            write_json(RUN_DIR / "stage-processes.json", {"stages": stage_rows})
            if result["exit_code"] != 0:
                raise RuntimeError(
                    f"{name} exited with {result['exit_code']}; no retry attempted"
                )
            reports[name] = validate_stage(
                RUN_DIR / f"{name}.json", expected_total, name
            )
            write_json(
                RUN_DIR / f"{name}-validated.json",
                {
                    "validated_at": utc_now(),
                    "expected_total": expected_total,
                    "infrastructure_pass": True,
                    "summary": reports[name]["summary"],
                },
            )

        post_snapshot = runtime_snapshot("post")
        write_json(RUN_DIR / "runtime-post.json", post_snapshot)
        source_stable = (
            pre_snapshot["combined_sha256"] == post_snapshot["combined_sha256"]
        )
        if not source_stable:
            raise RuntimeError("runtime source changed during evaluation")
        write_json(
            RUN_DIR / "final-summary.json",
            {
                "schema_version": "h007-cloud-full-agent-loop-summary.v1",
                "run_id": RUN_ID,
                "completed_at": utc_now(),
                "infrastructure_pass": True,
                "source_stable": True,
                "auto_retry_attempted": False,
                "policy_model": POLICY_MODEL,
                "policy_backend": POLICY_BACKEND,
                "intent_model": INTENT_MODEL,
                "intent_backend": INTENT_BACKEND,
                "stages": {
                    name: reports[name]["summary"] for name in reports
                },
                "promotion_val_opened": False,
                "sealed_160_opened": False,
                "production_promotion_eligible": False,
            },
        )
        print(f"[{utc_now()}] evaluation completed", flush=True)
    except BaseException as exc:
        run_error = f"{type(exc).__name__}: {exc}"
        write_json(
            RUN_DIR / "controller-error.json",
            {
                "failed_at": utc_now(),
                "error": run_error,
                "stages": stage_rows,
                "auto_retry_attempted": False,
            },
        )
        print(f"[{utc_now()}] ERROR {run_error}", flush=True)
        raise
    finally:
        if server is not None:
            stop_process_group(server, reason="vllm-cleanup")
        cleanup_gpu = None
        port_released = False
        try:
            cleanup_gpu = get_gpu_row()
            verify_port_free(8000)
            port_released = True
        except BaseException as cleanup_error:
            print(f"[{utc_now()}] cleanup verification warning: {cleanup_error}", flush=True)
        write_json(
            RUN_DIR / "server-cleanup.json",
            {
                "checked_at": utc_now(),
                "server_pid": None if server is None else server.pid,
                "server_exited": server is None or server.poll() is not None,
                "remote_port_released": port_released,
                "gpu_baseline_used_mib": baseline_gpu_mib,
                "gpu_after": cleanup_gpu,
                "run_error": run_error,
            },
        )


if __name__ == "__main__":
    main()

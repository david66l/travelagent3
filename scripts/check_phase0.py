#!/usr/bin/env python3
"""Run the reproducible Phase 0 quality baseline from the repository root."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
FRONTEND = ROOT / "frontend"


def run(label: str, command: list[str], cwd: Path) -> None:
    print(f"\n==> {label}", flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--with-db",
        action="store_true",
        help="Include tests marked requires_db (PostgreSQL must be available).",
    )
    parser.add_argument(
        "--strict-toolchains",
        action="store_true",
        help="Fail when optional frontend or gateway toolchains are unavailable.",
    )
    args = parser.parse_args()

    python = sys.executable
    run("Backend lint", [python, "-m", "ruff", "check", "src", "tests"], BACKEND)
    run(
        "Backend format check",
        [python, "-m", "ruff", "format", "--check", "src", "tests"],
        BACKEND,
    )
    run(
        "Backend type check",
        [python, "-m", "mypy", "src", "--ignore-missing-imports"],
        BACKEND,
    )
    run("Backend compile check", [python, "-m", "compileall", "-q", "src"], BACKEND)

    pytest_command = [python, "-m", "pytest", "tests/unit", "-q", "--tb=short"]
    if not args.with_db:
        pytest_command.extend(["-m", "not requires_db"])
    run("Backend unit tests", pytest_command, BACKEND)

    node = shutil.which("node")
    tsc = FRONTEND / "node_modules" / "typescript" / "bin" / "tsc"
    if node and tsc.exists():
        run("Frontend type check", [node, str(tsc), "--noEmit"], FRONTEND)
    elif args.strict_toolchains:
        raise RuntimeError("Node.js/TypeScript dependencies are unavailable")
    else:
        print("\n==> Frontend type check skipped (Node.js or node_modules missing)")

    go = shutil.which("go")
    if go:
        run("Gateway tests", [go, "test", "./..."], ROOT / "gateway")
    elif args.strict_toolchains:
        raise RuntimeError("Go toolchain is unavailable")
    else:
        print("\n==> Gateway tests skipped (Go toolchain missing)")

    print("\nPhase 0 baseline passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

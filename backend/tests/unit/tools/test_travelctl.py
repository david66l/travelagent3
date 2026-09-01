"""Tests for the travelctl naming-convention dispatcher."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_TRAVELCTL = Path(__file__).resolve().parents[4] / "scripts" / "travelctl.py"
_spec = importlib.util.spec_from_file_location("travelctl", _TRAVELCTL)
assert _spec is not None and _spec.loader is not None
travelctl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(travelctl)


def test_discover_groups_supported_verbs_with_docstring_help():
    registry = travelctl.discover_scripts()
    # The repository ships at least these verb families.
    for verb in ("evaluate", "audit", "build"):
        assert registry.get(verb), f"missing verb family: {verb}"
    # Help text comes from the script docstring's first line.
    evaluate = {entry["subject"]: entry for entry in registry["evaluate"]}
    assert "tradeoff-grounding-sft" in evaluate
    assert evaluate["tradeoff-grounding-sft"]["file"] == "evaluate_tradeoff_grounding_sft.py"
    assert evaluate["tradeoff-grounding-sft"]["help"]


def test_resolve_script_maps_alias_and_dashes():
    script = travelctl.resolve_script("eval", "tradeoff-grounding-sft")
    assert script is not None and script.name == "evaluate_tradeoff_grounding_sft.py"

    assert travelctl.resolve_script("audit", "verifier-reason-quality") is not None
    assert travelctl.resolve_script("evaluate", "does-not-exist") is None


def test_run_script_forwards_argv_and_exit_code(tmp_path, monkeypatch):
    fake = tmp_path / "evaluate_fake.py"
    fake.write_text(
        'import sys\n'
        'raise SystemExit(7 if "--flag" in sys.argv else 3)\n',
        encoding="utf-8",
    )
    code = travelctl.run_script(fake, ["--flag"])
    assert code == 7

    fake_ok = tmp_path / "audit_ok.py"
    fake_ok.write_text("print('ok')\n", encoding="utf-8")
    assert travelctl.run_script(fake_ok, []) == 0


def test_run_script_handles_string_system_exit_and_restores_argv(tmp_path, capsys):
    fake = tmp_path / "audit_boom.py"
    fake.write_text('raise SystemExit("boom: config missing")\n', encoding="utf-8")
    argv_before = list(sys.argv)
    code = travelctl.run_script(fake, [])
    assert code == 1
    assert "boom: config missing" in capsys.readouterr().err
    assert sys.argv == argv_before


def test_first_docstring_line_skips_shebang(tmp_path):
    script = tmp_path / "evaluate_shebang.py"
    script.write_text(
        '#!/usr/bin/env python3\n"""Help after shebang."""\n\nx = 1\n',
        encoding="utf-8",
    )
    assert travelctl._first_docstring_line(script) == "Help after shebang."

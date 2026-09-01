"""Tests for the travelctl naming-convention dispatcher."""

from __future__ import annotations

import importlib.util
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

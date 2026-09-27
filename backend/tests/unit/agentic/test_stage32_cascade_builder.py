import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[4] / "scripts" / "build_stage32_cascade_distillation.py"
SPEC = importlib.util.spec_from_file_location("build_stage32_cascade_distillation", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

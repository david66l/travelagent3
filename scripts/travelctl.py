"""Unified front door for the post-training tooling under ``scripts/``.

Every legacy entrypoint stays runnable exactly as before
(``python scripts/<name>.py ...``); historical evidence references keep
working.  ``travelctl`` only adds a conventional dispatcher on top:

    python scripts/travelctl.py list                 # auto-generated command index
    python scripts/travelctl.py list evaluate        # one verb family
    python scripts/travelctl.py eval tradeoff-grounding-sft --dataset-dir ...

Subcommand resolution is purely naming-convention based:

    travelctl <verb> <subject>   ->   scripts/<verb>_<subject-dashes>.py

where dashes in the subject map to underscores in the filename.  The registry
is discovered at call time, so adding/removing a script requires no
registration step and the index never goes stale.
"""

from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent

# Friendly short forms for the most common verbs.
VERB_ALIASES = {
    "eval": "evaluate",
    "bench": "benchmark",
    "gen": "generate",
}

# Verbs the dispatcher is willing to expose.  ``build`` is included but the
# long-tail one-off stage scripts remain directly runnable by filename.
SUPPORTED_VERBS = (
    "audit",
    "benchmark",
    "build",
    "check",
    "compare",
    "derive",
    "evaluate",
    "export",
    "generate",
    "merge",
    "repair",
    "reroute",
    "select",
    "smoke",
    "validate",
)


def _subject_to_stem(subject: str) -> str:
    return subject.replace("-", "_")


def _stem_to_subject(stem: str) -> str:
    return stem.replace("_", "-")


def normalize_verb(verb: str) -> str:
    return VERB_ALIASES.get(verb, verb)


def discover_scripts() -> dict[str, list[dict[str, str]]]:
    """Return ``{verb: [{subject, file, help}]}`` for the supported verbs."""
    registry: dict[str, list[dict[str, str]]] = {verb: [] for verb in SUPPORTED_VERBS}
    seen: set[tuple[str, str]] = set()
    for path in sorted(SCRIPTS_DIR.glob("*.py")):
        stem = path.stem
        raw_verb, _, rest = stem.partition("_")
        verb = normalize_verb(raw_verb)
        if verb not in registry or not rest:
            continue
        subject = _stem_to_subject(rest)
        # A canonical file (evaluate_x) and an alias file (eval_x) with the
        # same subject must not produce duplicate index entries.
        if (verb, subject) in seen:
            continue
        seen.add((verb, subject))
        registry[verb].append(
            {
                "subject": subject,
                "file": path.name,
                "help": _first_docstring_line(path),
            }
        )
    return registry


def _first_docstring_line(path: Path) -> str:
    try:
        content = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    # Skip shebang/coding lines so scripts that put #! first still get help.
    lines = content.splitlines()
    while lines and lines[0].startswith(("#!", "# -*-")):
        lines = lines[1:]
    if not lines or not lines[0].lstrip().startswith('"""'):
        return ""
    first = lines[0].lstrip()[3:]
    if first.endswith('"""'):
        first = first[:-3]
    first = first.strip()
    if first:
        return _console_safe(first)
    for line in lines[1:]:
        if '"""' in line:
            break
        if line.strip():
            return _console_safe(line.strip())
    return ""


def _console_safe(text: str) -> str:
    """Keep the index printable under legacy console encodings (e.g. GBK)."""
    encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
    return text.encode(encoding, errors="replace").decode(encoding, errors="replace")


def resolve_script(verb: str, subject: str) -> Path | None:
    """Map ``(verb, subject)`` onto a concrete script path, or ``None``."""
    verb = normalize_verb(verb)
    candidate = SCRIPTS_DIR / f"{verb}_{_subject_to_stem(subject)}.py"
    return candidate if candidate.is_file() else None


def run_script(script: Path, forward_args: list[str]) -> int:
    """Execute a legacy script in-process with forwarded arguments.

    The target script's ``sys.argv`` is restored afterwards; other process
    state it may mutate (``sys.path``, signal handlers, atexit hooks) is not
    isolated, so do not chain multiple dispatches in one process.
    """
    saved_argv = sys.argv
    sys.argv = [str(script), *forward_args]
    try:
        runpy.run_path(str(script), run_name="__main__")
    except SystemExit as exc:
        # argparse errors exit with 2; scripts may exit with ints or messages.
        if exc.code is None:
            return 0
        if isinstance(exc.code, int):
            return exc.code
        print(exc.code, file=sys.stderr)
        return 1
    finally:
        sys.argv = saved_argv
    return 0


def _print_index(registry: dict[str, list[dict[str, str]]], verb: str | None) -> None:
    verbs = [verb] if verb else [v for v in SUPPORTED_VERBS if registry.get(v)]
    for current in verbs:
        entries = registry.get(current, [])
        if not entries:
            continue
        print(f"[{current}] ({len(entries)})")
        for entry in entries:
            help_text = f"  — {entry['help']}" if entry["help"] else ""
            print(f"  travelctl {current} {entry['subject']}{help_text}")
        print()


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="travelctl",
        description=(
            "Dispatch post-training tooling by naming convention. "
            "Run `travelctl list` for the full index."
        ),
    )
    parser.add_argument("verb", help="verb family, e.g. eval / audit / build")
    parser.add_argument("subject", nargs="?", help="script subject with dashes")
    parser.add_argument(
        "forward", nargs=argparse.REMAINDER,
        help="arguments forwarded verbatim to the script",
    )
    args = parser.parse_args()

    registry = discover_scripts()
    verb = normalize_verb(args.verb)

    if args.verb == "list":
        _print_index(registry, normalize_verb(args.subject) if args.subject else None)
        return 0
    if args.subject is None:
        if args.forward and args.forward[0].startswith("-"):
            parser.error(
                "a subject is required before forwarded flags: "
                f"travelctl {args.verb} <subject> {' '.join(args.forward)}"
            )
        _print_index(registry, verb)
        return 0

    script = resolve_script(verb, args.subject)
    if script is None:
        available = ", ".join(e["subject"] for e in registry.get(verb, []))
        parser.error(
            f"no script matches '{verb} {args.subject}'. "
            f"Available subjects for '{verb}': {available or '(none)'}"
        )
    return run_script(script, args.forward)


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
"""Task runner for the BAB replication project.

Replaces the `make` targets CLAUDE.md referenced from the start of the project
but which never existed -- there was no Makefile, and `make` is not installed
on this Windows environment either, so every session that tried `make test`
failed and improvised its own pytest invocation. A Python runner fixes that
with no new dependency to install.

Usage:
    python run.py <task> [extra pytest args...]
    python run.py help

Why this exists rather than a bare pytest command in CLAUDE.md: it pins the
interpreter. This project has TWO virtualenvs and picking the wrong one fails
in a confusing way -- `.venv` is missing yaml/polars/pandas, so
`.venv/Scripts/python.exe -c "import yaml"` raises ModuleNotFoundError while
the identical command under `.venv-wrds` works. Encoding the choice here means
nobody has to remember it.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

# .venv-wrds, NOT .venv -- see module docstring. Falls back to the current
# interpreter if the venv is missing (e.g. a fresh clone, or CI), rather than
# failing with a confusing "file not found" on the executable itself.
_WRDS_PYTHON = PROJECT_ROOT / ".venv-wrds" / "Scripts" / "python.exe"
_WRDS_PYTHON_POSIX = PROJECT_ROOT / ".venv-wrds" / "bin" / "python"


def _python() -> str:
    for candidate in (_WRDS_PYTHON, _WRDS_PYTHON_POSIX):
        if candidate.exists():
            return str(candidate)
    print(
        "WARNING: .venv-wrds not found, falling back to the current "
        f"interpreter ({sys.executable}). If imports fail (yaml, polars, "
        "pandas, wrds), that is why -- .venv is NOT a substitute.",
        file=sys.stderr,
    )
    return sys.executable


# Each task: (description, argv-after-python, note-printed-before-running).
# The `note` field exists for tasks whose result needs interpreting rather than
# just reading -- see `leakage`.
_LEAKAGE_NOTE = """\
NOTE: this task currently reports 6 NotImplementedError failures. They are
stubs, not regressions -- the four adapter functions in
tests/leakage/test_no_lookahead.py are unimplemented, so the truncation test
(described in its own docstring as "the single most valuable test in this
file") has never actually executed.

CLAUDE.md calls this must-pass-before-any-commit. It is NOT doing that job
today. A permanently-red gate is worse than a missing one: it trains everyone
to ignore it. Wired in three stages (E1 with Phase A, E2 with D, E3 with F)
-- see docs/03_roadmap.md.

The two leakage tests that ARE real (test_universe_panel_crsp_pit_bounds.py,
test_ccm_link_date_bounds.py) do pass, which is why this runs without -x:
stopping at the first stub would hide them.
"""

TASKS: dict[str, tuple[str, list[str], str | None]] = {
    "test": ("full test suite", ["-m", "pytest", "tests/", "-q"], None),
    "leakage": (
        "leakage tests only (SEE NOTE)",
        ["-m", "pytest", "tests/leakage/", "-q"],
        _LEAKAGE_NOTE,
    ),
    "gates": (
        "validation gates (slow, needs cached data)",
        ["-m", "pytest", "tests/gates/", "-q"],
        None,
    ),
    "unit": (
        "unit tests only (fast, no WRDS data needed)",
        ["-m", "pytest", "tests/unit/", "-q"],
        None,
    ),
    "lint": ("ruff check", ["-m", "ruff", "check", "src/", "tests/"], None),
    "typecheck": ("mypy", ["-m", "mypy", "src/"], None),
}


def _run(argv: list[str]) -> int:
    cmd = [_python(), *argv]
    print(f"$ {' '.join(cmd)}\n", flush=True)
    return subprocess.call(cmd, cwd=PROJECT_ROOT)


def _help() -> int:
    print(__doc__.split("Usage:")[0].strip())
    print("\nTasks:")
    width = max(len(name) for name in TASKS) + 2
    for name, (desc, _, _) in TASKS.items():
        print(f"  {name:<{width}} {desc}")
    print(f"  {'check':<{width}} lint + typecheck + test")
    print("\nExtra arguments are passed through to the underlying command:")
    print("  python run.py test -x -k universe")
    return 0


def main(argv: list[str]) -> int:
    if not argv or argv[0] in {"help", "-h", "--help"}:
        return _help()

    task, extra = argv[0], argv[1:]

    if task == "check":
        # Sequential, stop on first failure -- a type error makes the test
        # results harder to interpret, so there is no value in pressing on.
        for name in ("lint", "typecheck", "test"):
            _, args, _ = TASKS[name]
            code = _run(args)
            if code != 0:
                print(f"\n'{name}' failed (exit {code}) -- stopping.", file=sys.stderr)
                return code
        return 0

    if task not in TASKS:
        print(f"Unknown task: {task}\n", file=sys.stderr)
        _help()
        return 2

    _, args, note = TASKS[task]
    if note:
        print(note, file=sys.stderr)
    return _run([*args, *extra])


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

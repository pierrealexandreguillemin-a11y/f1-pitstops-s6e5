"""Pre-commit static check for Jupyter notebooks.

For each .ipynb passed on argv:
1. Parse via nbformat (valid JSON + nbformat schema).
2. For each code cell, compile() the source — catches SyntaxError and other parse-time bugs
   BEFORE the notebook is ever executed. Cheap, fast, no kernel needed.

Does NOT execute the notebook. Runtime errors (wrong column name, API misuse, etc.) are
caught at notebook-run time, not at commit time. This script is the cheap gate.

Exit 0 = clean ; exit 1 = at least one notebook failed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import nbformat


def check_one(nb_path: Path) -> list[str]:
    """Return list of error strings (empty = clean)."""
    errors: list[str] = []
    try:
        nb = nbformat.read(nb_path, as_version=nbformat.NO_CONVERT)
    except Exception as e:
        return [f"{nb_path}: nbformat.read failed: {e}"]

    for idx, cell in enumerate(nb.cells):
        if cell.cell_type != "code":
            continue
        source = cell.source
        if not source.strip():
            continue
        # Strip jupyter magics + shell ! lines before compile (they're not valid Python).
        cleaned_lines = []
        for line in source.splitlines():
            stripped = line.lstrip()
            if stripped.startswith(("%", "!", "?")) or stripped.endswith("?"):
                cleaned_lines.append("pass  # stripped magic/shell line")
            else:
                cleaned_lines.append(line)
        cleaned = "\n".join(cleaned_lines)
        try:
            compile(cleaned, f"{nb_path}::cell-{idx}", "exec")
        except SyntaxError as e:
            errors.append(
                f"{nb_path}::cell-{idx} SyntaxError line {e.lineno}: {e.msg}\n  {e.text or ''}"
            )
        except Exception as e:
            errors.append(f"{nb_path}::cell-{idx} compile error: {type(e).__name__}: {e}")
    return errors


def main() -> int:
    paths = [Path(p) for p in sys.argv[1:]]
    if not paths:
        return 0

    # No venv guard here: this script is invoked by pre-commit's isolated venv
    # (deps declared via additional_dependencies in .pre-commit-config.yaml).
    # If nbformat is missing, the import at top of file fails fast with ImportError.

    total_errors: list[str] = []
    for p in paths:
        if p.suffix != ".ipynb":
            continue
        total_errors.extend(check_one(p))

    if total_errors:
        print("Notebook static check FAILED:", file=sys.stderr)
        for err in total_errors:
            print(f"  {err}", file=sys.stderr)
        return 1

    print(f"Notebook static check OK ({len(paths)} files)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

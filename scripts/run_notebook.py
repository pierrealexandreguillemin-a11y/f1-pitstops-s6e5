"""Execute a Jupyter notebook in-process via nbconvert Python API.

Why this exists
---------------
The `jupyter <subcmd>` CLI dispatches to `jupyter-<subcmd>.exe` via PATH lookup. On a laptop
with a polluted user-site (`C:/Users/<u>/AppData/Roaming/Python/Python313/Scripts/`), PATH
finds the user-site exe BEFORE the venv's. That exe uses a different Python that does NOT
have our deps installed. Bug observed on f1-pitstops-s6e5 bootstrap 2026-05-26.

This script bypasses the contamination structurally :
- Invoked as `.venv/Scripts/python.exe scripts/run_notebook.py <nb.ipynb>`
- Uses nbconvert's ExecutePreprocessor Python API (no jupyter CLI involved)
- Kernel resolution via jupyter_core.paths which prioritizes venv (`sys.prefix`) over user-site
- ALSO runs nbstripout to clear outputs (deterministic git diff)

The dispatcher PATH lookup never happens. Contamination is impossible.

Usage
-----
    .venv/Scripts/python.exe scripts/run_notebook.py notebooks/00_setup.ipynb
"""

from __future__ import annotations

import sys
from pathlib import Path

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor


def run(nb_path: Path, timeout: int = 600) -> None:
    if not nb_path.exists():
        raise FileNotFoundError(f"Notebook not found: {nb_path}")

    print(f"[run_notebook] Executing {nb_path}")
    nb = nbformat.read(nb_path, as_version=4)

    ep = ExecutePreprocessor(timeout=timeout, kernel_name="python3")
    ep.preprocess(nb, {"metadata": {"path": str(nb_path.parent.resolve())}})

    nbformat.write(nb, nb_path)
    print(f"[run_notebook] Execution OK, outputs written back to {nb_path}")


def strip(nb_path: Path) -> None:
    """Clear outputs via nbstripout API (no CLI dispatch)."""
    import nbstripout

    nb = nbformat.read(nb_path, as_version=nbformat.NO_CONVERT)
    nbstripout.strip_output(nb, keep_output=False, keep_count=False, keep_id=True)
    nbformat.write(nb, nb_path)
    print(f"[run_notebook] Outputs stripped from {nb_path}")


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: python scripts/run_notebook.py <notebook.ipynb> [--no-strip]")
        return 2

    nb_path = Path(sys.argv[1]).resolve()
    no_strip = "--no-strip" in sys.argv[2:]

    if ".venv" not in sys.executable:
        print(f"ERROR: Python executable is NOT venv-scoped: {sys.executable}", file=sys.stderr)
        print("Invoke via: .venv/Scripts/python.exe scripts/run_notebook.py ...", file=sys.stderr)
        return 3

    run(nb_path)
    if not no_strip:
        strip(nb_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())

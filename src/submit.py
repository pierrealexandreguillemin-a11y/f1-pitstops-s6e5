"""Submission helper - write CSV + call kaggle CLI."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
SUBMISSIONS_DIR = REPO_ROOT / "submissions"


def write_submission(
    test_ids: pd.Series, y_pred: pd.Series, step_num: int, approach_tag: str
) -> Path:
    """Write submission CSV named NN_<tag>.csv. Returns the path."""
    SUBMISSIONS_DIR.mkdir(exist_ok=True)
    filename = f"{step_num:02d}_{approach_tag}.csv"
    path = SUBMISSIONS_DIR / filename
    pd.DataFrame({"id": test_ids, "PitNextLap": y_pred}).to_csv(path, index=False)
    print(f"Wrote {path} ({len(test_ids)} rows)")
    return path


def submit_to_kaggle(csv_path: Path, message: str, dry_run: bool = False) -> None:
    """Submit via kaggle CLI. dry_run=True just prints the command.

    Uses .venv/Scripts/kaggle.exe explicitly to stay venv-scoped (no global pollution).
    """
    kaggle_exe = REPO_ROOT / ".venv" / "Scripts" / "kaggle.exe"
    cmd = [
        str(kaggle_exe),
        "competitions",
        "submit",
        "playground-series-s6e5",
        "-f",
        str(csv_path),
        "-m",
        message,
    ]
    print(f"$ {' '.join(cmd)}")
    if not dry_run:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        print(result.stdout)
        if result.returncode != 0:
            print(f"STDERR: {result.stderr}")
            raise RuntimeError(f"kaggle submit failed: exit {result.returncode}")

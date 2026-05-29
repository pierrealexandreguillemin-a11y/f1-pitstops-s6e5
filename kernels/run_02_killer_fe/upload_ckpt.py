"""Cross-run checkpoint handoff for submit-02 (timeout-proof relaunch).

Between kernel runs, this script:
  1. downloads the completed run's output (ckpt/ artifacts),
  2. stages the resume-essential files (fold_*.npz + *_meta.json — NOT the .cbm
     models, kept small for a fast upload),
  3. creates or versions the Kaggle dataset that the next run mounts via
     dataset_sources to resume.

Run locally (never on Kaggle):
    .venv/Scripts/python.exe kernels/run_02_killer_fe/upload_ckpt.py

Then add the dataset to kernel-metadata.json "dataset_sources" and re-push.
See docs/plans/2026-05-29-submit-02-checkpoint-resume-plan.md.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

KERNEL_SLUG = "pguillemin/f1-pitstops-s6e5-submit-02-killer-fe"
CKPT_DATASET_SLUG = "pguillemin/f1-s6e5-submit02-ckpt"
CKPT_DATASET_TITLE = "f1-s6e5 submit02 ckpt"
KAGGLE = "kaggle"  # resolve via venv Scripts if not on PATH


def _run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    print("  $", " ".join(cmd))
    return subprocess.run(cmd, text=True, capture_output=True, check=False, **kw)


def main() -> None:
    staging = Path(tempfile.mkdtemp(prefix="s6e5_ckpt_"))
    dl = staging / "download"
    dl.mkdir()

    print(f"[1/3] download kernel output {KERNEL_SLUG}")
    out = _run([KAGGLE, "kernels", "output", KERNEL_SLUG, "-p", str(dl)])
    print(out.stdout, out.stderr)

    # Resume-essential files only (preds + meta); models stay in kernel output.
    src_files = sorted(dl.rglob("fold_*.npz")) + sorted(dl.rglob("fold_*_meta.json"))
    if not src_files:
        sys.exit(
            "  no fold_*.npz found in output — run not COMPLETE, or no folds done yet.\n"
            f"  check: {KAGGLE} kernels status {KERNEL_SLUG}"
        )
    ds_dir = staging / "dataset" / "ckpt"
    ds_dir.mkdir(parents=True)
    for f in src_files:
        (ds_dir / f.name).write_bytes(f.read_bytes())
    folds = sorted({int(f.stem.split("_")[1]) for f in src_files if f.suffix == ".npz"})
    print(f"  staged folds {folds} ({len(src_files)} files)")

    meta_path = staging / "dataset" / "dataset-metadata.json"
    meta_path.write_text(
        json.dumps(
            {
                "title": CKPT_DATASET_TITLE,
                "id": CKPT_DATASET_SLUG,
                "licenses": [{"name": "CC0-1.0"}],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"[2/3] probe dataset {CKPT_DATASET_SLUG}")
    status = _run([KAGGLE, "datasets", "status", CKPT_DATASET_SLUG])
    exists = status.returncode == 0 and status.stdout.strip() in ("ready", "creating")

    print(f"[3/3] {'version' if exists else 'create'} dataset")
    ds_root = str(meta_path.parent)
    if exists:
        res = _run(
            [KAGGLE, "datasets", "version", "-p", ds_root, "-r", "zip", "-m", f"folds {folds}"]
        )
    else:
        res = _run([KAGGLE, "datasets", "create", "-p", ds_root, "-r", "zip"])
    print(res.stdout, res.stderr)
    print(f'\nDONE. Add "{CKPT_DATASET_SLUG}" to dataset_sources and re-push to resume.')


if __name__ == "__main__":
    main()

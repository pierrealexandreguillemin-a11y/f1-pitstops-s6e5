"""Tests for submit-02 kernel checkpoint/resume plumbing.

Validates the timeout-proof relaunch mechanism WITHOUT any Kaggle cost:
- per-fold checkpoint round-trip (save -> load) preserves preds + meta
- restore_checkpoints() copies prior-run ckpts from a mounted input tree and
  discovers completed folds (the cross-run resume path)

The kernel lives at kernels/run_02_killer_fe/run_02_killer_fe.py (not a package),
so we load it by file path and monkeypatch its CKPT_DIR / INPUT_ROOT constants
to a tmp dir.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

KERNEL_PATH = (
    Path(__file__).resolve().parents[1] / "kernels" / "run_02_killer_fe" / "run_02_killer_fe.py"
)


@pytest.fixture()
def kernel(tmp_path, monkeypatch):
    """Import the kernel module with paths redirected to tmp_path."""
    spec = importlib.util.spec_from_file_location("run_02_killer_fe", KERNEL_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.setattr(mod, "CKPT_DIR", tmp_path / "ckpt")
    monkeypatch.setattr(mod, "INPUT_ROOT", tmp_path / "input")
    return mod


def _fake_fold(n_val=50, n_test=80, n_feat=12, seed=0):
    rng = np.random.default_rng(seed)
    va_idx = np.arange(n_val) * 2  # arbitrary non-trivial indices
    va_pred = rng.random(n_val)
    test_pred = rng.random(n_test)
    feat_imp = rng.random(n_feat)
    meta = {
        "fold": 0,
        "n_train": 999,
        "n_val": n_val,
        "best_iter": 1234,
        "auc": 0.95,
        "brier": 0.04,
        "logloss": 0.14,
        "seconds": 12.3,
        "per_year": [{"year": 2024, "n": 25, "auc": 0.97, "y_true_mean": 0.2, "y_pred_mean": 0.2}],
    }
    return va_idx, va_pred, test_pred, feat_imp, meta


def test_save_load_roundtrip_preserves_everything(kernel):
    kernel.CKPT_DIR.mkdir(parents=True)
    va_idx, va_pred, test_pred, feat_imp, meta = _fake_fold()

    kernel.save_fold_ckpt(0, va_idx, va_pred, test_pred, feat_imp, meta)
    lo_idx, lo_vp, lo_tp, lo_fi, lo_meta = kernel.load_fold_ckpt(0)

    assert np.array_equal(lo_idx, va_idx)
    assert np.allclose(lo_vp, va_pred)
    assert np.allclose(lo_tp, test_pred)
    assert np.allclose(lo_fi, feat_imp)
    assert lo_meta == meta


def test_restore_finds_no_folds_when_input_absent(kernel):
    # No /kaggle/input tree => clean first run, empty restore (must not raise).
    assert kernel.restore_checkpoints() == []
    assert kernel.CKPT_DIR.exists()


def test_restore_copies_prior_run_ckpts_and_discovers_folds(kernel):
    # Simulate a prior run's output mounted as a dataset at an arbitrary depth.
    prior = kernel.INPUT_ROOT / "datasets" / "pguillemin" / "ckpt-ds" / "ckpt"
    prior.mkdir(parents=True)

    # Write fold 0 and fold 2 (a gap is realistic across runs) into the fake mount.
    for fi in (0, 2):
        va_idx, va_pred, test_pred, feat_imp, meta = _fake_fold(seed=fi)
        meta["fold"] = fi
        np.savez(
            prior / f"fold_{fi}.npz",
            va_idx=va_idx,
            va_pred=va_pred,
            test_pred=test_pred,
            feat_imp=feat_imp,
        )
        (prior / f"fold_{fi}_meta.json").write_text(json.dumps(meta), encoding="utf-8")

    restored = kernel.restore_checkpoints()

    assert restored == [0, 2]
    # ckpts were copied into the working CKPT_DIR and are loadable there.
    for fi in (0, 2):
        assert (kernel.CKPT_DIR / f"fold_{fi}.npz").exists()
        _, _, _, _, meta = kernel.load_fold_ckpt(fi)
        assert meta["fold"] == fi


def test_smoke_constant_is_off_for_real_runs(kernel):
    # Guard: the throwaway smoke toggle must never be committed as True.
    assert kernel.SMOKE is False
    assert kernel.CATBOOST_PARAMS["iterations"] == 16000

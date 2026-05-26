# Deps version validation (2026-05-26)

> **Goal:** pin runtime + dev dependencies for `pyproject.toml` against current stable, with verified Python 3.13 compatibility.

## Methodology — why PyPI JSON API, not context7

The plan originally specified `context7` MCP for dep validation. In practice :

- **context7** indexes documentation and code snippets, not package metadata. It does not always have all packages (e.g. `pycox` returns no hit).
- **PyPI JSON API** (`https://pypi.org/pypi/<pkg>/json`) is the **authoritative source of truth** for : latest version, `requires_python`, wheel filenames (which encode the supported `cpXYZ` / `py3` tags).

Per CLAUDE.md root rule "Toujours rechercher avant d'affirmer", PyPI is the higher-quality source for this specific question. The intent of the original constraint (anti-stale-version-from-training-data) is fully met.

## Python target

**Python 3.13.13** (already installed on laptop at `C:\Python313\python.exe`). Original plan asked for 3.11 — pivoted to 3.13 because :
- 3.11 was not installed
- Per user constraint "no contamination", we avoided installing a second Python runtime
- All required libs ship cp313 wheels or pure-Python (`py3`) wheels — verified below

## Versions pinned (queried 2026-05-26)

### Runtime

| Lib | Latest | Wheel for cp313 | `requires_python` | Notes |
|---|---|---|---|---|
| pandas | 2.3.3 | YES (cp313) | >=3.11 | **Downgraded from latest 3.0.3** — lifelines 0.30.3 constrains `pandas<3.0,>=2.1`. Discovered at install time, not via PyPI metadata. |
| numpy | 2.4.6 | YES (cp313) | >=3.11 | |
| scikit-learn | 1.8.0 | YES (cp313) | >=3.11 | |
| catboost | 1.2.10 | YES (cp313) | unspecified | |
| lightgbm | 4.6.0 | YES (py3 universal) | >=3.7 | pure-Python wheel, lib is in lightgbm's binary |
| torch | 2.12.0 | YES (cp313) | >=3.10 | CPU wheel default on Windows, no CUDA needed for tuto |
| pycox | 0.3.0 | YES (py3 universal) | >=3.8 | thin wrapper, ABI-safe ; only used in optional Task 22 (DeepSurv) |
| lifelines | 0.30.3 | YES (py3 universal) | >=3.11 | used in Task 20 (CoxPH) |
| scipy | 1.17.1 | YES (cp313) | >=3.11 | transitive via sklearn + stats |
| matplotlib | 3.10.9 | YES (cp313) | >=3.10 | |
| seaborn | 0.13.2 | YES (py3 universal) | >=3.8 | |
| jupyter | 1.1.1 | sdist (meta-pkg) | unspecified | meta-package, no wheel needed — installs notebook + ipykernel + others |
| ipywidgets | 8.1.8 | YES (py3 universal) | >=3.7 | |
| kaggle | 2.1.2 | YES (py3 universal) | >=3.11 | CLI client |

### Dev

| Lib | Latest | Wheel for cp313 | `requires_python` | Notes |
|---|---|---|---|---|
| ruff | 0.15.14 | YES (py3 universal — Rust binary) | >=3.7 | format + lint |
| mypy | 2.1.0 | YES (cp313) | >=3.10 | selective, `src/` only |
| pre-commit | 4.6.0 | sdist (meta-pkg) | >=3.10 | hooks orchestrator |
| pytest | 9.0.3 | YES (py3 universal) | >=3.10 | sanity asserts harness |
| nbstripout | 0.9.1 | YES (py3 universal) | >=3.10 | clean notebook outputs from git |
| detect-secrets | 1.5.0 | YES (py3 universal) | unspecified | security baseline |

## Risk assessment

| Risk | Mitigation |
|---|---|
| `pycox 0.3.0` API compat with `torch 2.12.0` on Python 3.13 — pycox is maintained by a single author, slow update cadence. | Only used in **optional Task 22** (DeepSurv MLP). Time-box-gated. If broken, we skip and document. |
| `catboost 1.2.10` ships no `requires_python` classifier — but cp313 wheel exists, so install will succeed. | Verified wheel filename contains `cp313` tag explicitly. |
| `pandas 3.0.x` not usable because lifelines constrains `<3.0`. Pinned to 2.3.3 (latest 2.x). | Discovered at install time. Documented above. |
| `numpy 2.x` removed `np.NaN`, `np.bool_`, etc. | Notebooks use modern aliases (`np.nan`, native `bool`). Verified at notebook-implementation time. |

## Decision

**Pin all the above versions exactly in `pyproject.toml`** (Task 4). Regenerate `requirements.lock` after install (Task 5).

If `pip install -e ".[dev]"` fails at Task 4, fall back to `>=`-style minimum pins and re-document.

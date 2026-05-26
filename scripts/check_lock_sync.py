"""Verify requirements.lock contains every dep declared in pyproject.toml."""

from __future__ import annotations

import sys
import tomllib
from pathlib import Path


def _normalize(name: str) -> str:
    """PEP 503 normalization: lowercase, replace runs of -/_/. with -."""
    import re

    return re.sub(r"[-_.]+", "-", name).lower()


def parse_pyproject_deps() -> set[str]:
    data = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    deps = data["project"]["dependencies"]
    dev_groups = data.get("dependency-groups", {}).get("dev", [])
    dev_optional = data.get("project", {}).get("optional-dependencies", {}).get("dev", [])
    dev = dev_groups + dev_optional
    return {
        _normalize(pkg.split("==")[0].split(">=")[0].split("[")[0].strip()) for pkg in deps + dev
    }


def parse_lock_pkgs() -> set[str]:
    lock = Path("requirements.lock")
    if not lock.exists():
        return set()
    pkgs = set()
    for raw_line in lock.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        pkgs.add(_normalize(line.split("==")[0].split(">=")[0].split(" ")[0]))
    return pkgs


def main() -> int:
    declared = parse_pyproject_deps()
    locked = parse_lock_pkgs()
    missing = declared - locked
    if missing:
        print(f"ERROR: requirements.lock missing {len(missing)} declared deps: {sorted(missing)}")
        print(
            "Re-run: .venv/Scripts/python.exe -m pip freeze --exclude-editable > requirements.lock"
        )
        return 1
    print("OK: lock-sync verified")
    return 0


if __name__ == "__main__":
    sys.exit(main())

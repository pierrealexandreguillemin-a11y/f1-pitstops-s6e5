#!/usr/bin/env bash
# doctor.sh - venv-contamination audit for f1-pitstops-s6e5
#
# Verifies that all critical Python tools resolve to .venv/Scripts/ after activation.
# If any tool resolves outside the venv, we have PATH contamination from user-site or
# system installs (see CLAUDE.md "CONTAMINATION CONNUE" section).
#
# Usage:  bash scripts/doctor.sh
# Exit:   0 = clean ; 1 = contamination detected

set -u

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT" || exit 2

# shellcheck disable=SC1091
source .venv/Scripts/activate

EXPECTED_PREFIX="$REPO_ROOT/.venv/Scripts"

TOOLS=(python pip pytest ruff mypy jupyter jupyter-nbconvert kaggle nbstripout pre-commit detect-secrets)

fail=0
for tool in "${TOOLS[@]}"; do
    resolved="$(which "$tool" 2>/dev/null || echo "NOT_FOUND")"
    # Normalize Windows path forward-slashes
    resolved_norm="$(echo "$resolved" | sed 's|\\|/|g')"
    expected_norm="$(echo "$EXPECTED_PREFIX" | sed 's|\\|/|g')"
    if [[ "$resolved_norm" == "$expected_norm"/* ]] || [[ "$resolved_norm" == "/$(echo "$expected_norm" | sed 's|^/c/|c/|')"/* ]]; then
        printf "  OK   %-22s -> %s\n" "$tool" "$resolved"
    else
        printf "  FAIL %-22s -> %s\n" "$tool" "$resolved"
        fail=1
    fi
done

if [[ $fail -eq 0 ]]; then
    echo "doctor: all tools resolve to venv. Clean."
    exit 0
else
    echo "doctor: CONTAMINATION DETECTED. See CLAUDE.md 'CONTAMINATION CONNUE'."
    exit 1
fi

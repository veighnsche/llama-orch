#!/usr/bin/env bash
set -euo pipefail

# Moves the root TODO.md to .docs/DONE/TODO-[auto-increment].md
# Usage: bash ci/scripts/archive_todo.sh

# Resolve repository root (works whether script is run from repo root or elsewhere)
REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || true)"
if [[ -z "${REPO_ROOT}" ]]; then
  # Fallback: assume script lives under <repo>/ci/scripts/
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
fi

SRC_TODO="${REPO_ROOT}/TODO.md"
DONE_DIR="${REPO_ROOT}/.docs/DONE"

if [[ ! -f "${SRC_TODO}" ]]; then
  echo "ERROR: No TODO.md found at repo root: ${SRC_TODO}" >&2
  exit 1
fi

mkdir -p "${DONE_DIR}"

# Determine next auto-increment number based on existing files: TODO-<n>.md
NEXT=0
shopt -s nullglob
existing=("${DONE_DIR}"/TODO-*.md)
if (( ${#existing[@]} > 0 )); then
  # Extract numeric suffixes and find max
  max=-1
  for f in "${existing[@]}"; do
    base="$(basename "$f")"
    num="${base#TODO-}"
    num="${num%.md}"
    if [[ "$num" =~ ^[0-9]+$ ]]; then
      if (( num > max )); then max=$num; fi
    fi
  done
  if (( max >= 0 )); then
    NEXT=$((max + 1))
  fi
fi
shopt -u nullglob

DST_TODO="${DONE_DIR}/TODO-${NEXT}.md"

mv "${SRC_TODO}" "${DST_TODO}"
echo "Archived: ${DST_TODO}"

# Optionally create a fresh empty TODO.md
cat > "${SRC_TODO}" <<'EOF'
# TODO — Active Tracker (Spec→Contract→Tests→Code)

This is the single active TODO tracker for the repository. Maintain execution order and update after each task with what changed, where, and why.

## P0 — Blockers (in order)

- [ ]

## Progress Log (what changed)

EOF

echo "Created fresh TODO.md at: ${SRC_TODO}"

#!/usr/bin/env bash
set -euo pipefail

# Configuration
EXCLUDE_EXTS="md,MD"
IGNORE_DIRS="reference"

# Collect all tracked top-level dirs (excluding ignored ones)
top_dirs=$(
  git ls-files | awk -F/ 'NF>1 {print $1}' | sort -u \
  | grep -vE "^($(echo "$IGNORE_DIRS" | tr ' ' '|'))$" || true
)

# Root-level tracked files (no slash)
root_files=$(git ls-files | awk -F/ 'NF==1')

# Root files (if any)
if [[ -n "${root_files}" ]]; then
  echo
  echo "============================================================"
  echo "[ROOT]"
  echo "============================================================"
  printf "%s\n" "${root_files}" \
    | cloc --list-file=- --exclude-ext="${EXCLUDE_EXTS}" --hide-rate
fi

# Each top-level directory except ignored ones
for d in ${top_dirs}; do
  echo
  echo "============================================================"
  echo "[${d}]"
  echo "============================================================"
  git ls-files "${d}/**" \
    | cloc --list-file=- --exclude-ext="${EXCLUDE_EXTS}" --hide-rate
done

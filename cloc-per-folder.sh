#!/usr/bin/env bash
set -euo pipefail

# Count per top-level folder, respecting .gitignore (by using git's file list),
# and excluding Markdown files (both md/MD).
# Requires: git, cloc

EXCLUDE_EXTS="md,MD"

# Top-level dirs that have tracked files
top_dirs=$(
  git ls-files | awk -F/ 'NF>1 {print $1}' | sort -u
)

# Root-level tracked files (no slash)
root_files=$(git ls-files | awk -F/ 'NF==1')

# Root files (if any)
if [[ -n "${root_files}" ]]; then
  echo
  echo "============================================================"
  echo "[ROOT]"
  echo "============================================================"
  printf "%s\n" "${root_files}" | cloc --list-file=- --exclude-ext="${EXCLUDE_EXTS}" --hide-rate
fi

# Each top-level directory
for d in ${top_dirs}; do
  echo
  echo "============================================================"
  echo "[${d}]"
  echo "============================================================"
  # Feed only files under this dir (tracked by git), exclude md/MD via cloc
  git ls-files "${d}/**" \
    | cloc --list-file=- --exclude-ext="${EXCLUDE_EXTS}" --hide-rate
done

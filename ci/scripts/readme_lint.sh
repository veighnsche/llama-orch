#!/usr/bin/env bash
set -euo pipefail

# Lint generated READMEs for:
# - Max line length (<= 100)
# - Required sections presence

fail=0

required_sections=(
  "## 1. Name & Purpose"
  "## 2. Why it exists (Spec traceability)"
  "## 3. Public API surface"
  "## 4. How it fits"
  "## 5. Build & Test"
  "## 6. Contracts"
  "## 7. Config & Env"
  "## 8. Metrics & Logs"
  "## 9. Runbook (Dev)"
  "## 10. Status & Owners"
  "## 11. Changelog pointers"
  "## 12. Footnotes"
  "## What this crate is not"
)

# Only lint README.md files under crate directories and root
while IFS= read -r -d '' file; do
  # Section presence (skip root README for sections, only length check)
  if [[ "$file" != "./README.md" ]]; then
    for sec in "${required_sections[@]}"; do
      if ! grep -qF "$sec" "$file"; then
        echo "Missing section '$sec' in $file" >&2
        fail=1
      fi
    done
  fi

  # Line length check
  # Ignore code fences by tracking state
  in_code=0
  lineno=0
  while IFS= read -r line; do
    lineno=$((lineno+1))
    if [[ "${line:0:3}" == '```' ]]; then
      if [[ $in_code -eq 0 ]]; then in_code=1; else in_code=0; fi
      continue
    fi
    if [[ $in_code -eq 1 ]]; then
      continue
    fi
    # Skip docs.rs badges and shields
    if [[ "$line" == \[\!\[docs.rs* ]] || [[ "$line" == *"img.shields.io"* ]]; then
      continue
    fi
    # Count characters (tab expanded to 2 spaces)
    exp_line=${line//$'\t'/"  "}
    if (( ${#exp_line} > 100 )); then
      echo "Line too long ($(( ${#exp_line} )) > 100) in $file:$lineno" >&2
      fail=1
    fi
  done < "$file"

done < <(find . -type f -name README.md -print0)

exit $fail

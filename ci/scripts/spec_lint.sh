#!/usr/bin/env bash
set -euo pipefail

# SPEC lint: ensure RFC-2119 usage, stable IDs, and no duplicates across .specs/*.md
# IDs supported: ORCH-##### and OC-AREA-#### (AREA is [A-Z0-9-]+)

shopt -s nullglob
specs=(.specs/*.md)

if (( ${#specs[@]} == 0 )); then
  echo "No spec files found under .specs/" >&2
  exit 1
fi

all_ids=()
rc=0

for f in "${specs[@]}"; do
  # RFC-2119 words (case-insensitive) must appear
  if ! grep -Eiq '\b(MUST|SHOULD|MAY)\b' "$f"; then
    echo "Missing RFC-2119 words in $f" >&2
    rc=1
  fi
  # Extract IDs
  ids=$(grep -Eo 'ORCH-[0-9]{3,5}|OC-[A-Z0-9-]+-[0-9]{3,5}' "$f" || true)
  if [[ -z "$ids" ]]; then
    echo "No requirement IDs found in $f" >&2
    rc=1
  fi
  while IFS= read -r id; do
    [[ -z "$id" ]] && continue
    all_ids+=("$id|$f")
  done <<< "$ids"

  # Verify links in file exist via existing link checker (anchors ignored there)
  :

done

# Check duplicates across all specs
if (( ${#all_ids[@]} > 0 )); then
  dups=$(printf '%s
' "${all_ids[@]}" | cut -d'|' -f1 | sort | uniq -d || true)
  if [[ -n "$dups" ]]; then
    echo "Duplicate requirement IDs found across specs:" >&2
    printf '  %s\n' $dups >&2
    rc=1
  fi
fi

exit $rc

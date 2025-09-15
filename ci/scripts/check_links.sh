#!/usr/bin/env bash
set -euo pipefail

# Simple internal link checker for Markdown files: verifies that local file links exist.
# Not a full link validator; good enough for pre-code CI.

fail=0
while IFS= read -r -d '' file; do
  while read -r link; do
    # Extract target between parentheses
    target=$(echo "$link" | sed -n 's/.*](\(.*\))/\1/p')
    # Skip URLs (http/https/mailto)
    if [[ "$target" =~ ^(http|https|mailto): ]]; then
      continue
    fi
    # Strip anchors
    base="${target%%#*}"
    # Resolve relative path
    dir=$(dirname "$file")
    abs=$(realpath -m "$dir/$base")
    if [ -n "$base" ] && [ ! -f "$abs" ]; then
      echo "Missing link target in $file: $target (resolved $abs)" >&2
      fail=1
    fi
  done < <(grep -o '\[[^]]\+\](\([^)]\+\))' "$file" || true)

done < <(find . -name "*.md" -print0)

exit $fail

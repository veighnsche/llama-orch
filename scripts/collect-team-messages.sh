#!/usr/bin/env bash
set -euo pipefail

# Collect all markdown files with TEAM in the filename
# and copy them to .team-messages/

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET_DIR="$REPO_ROOT/.team-messages"

echo "Collecting TEAM markdown files..."

# Create target directory
mkdir -p "$TARGET_DIR"

# Find and copy all markdown files with TEAM in the name
find "$REPO_ROOT" \
  -type f \
  -name "*TEAM*.md" \
  -not -path "*/.team-messages/*" \
  -not -path "*/.git/*" \
  -not -path "*/node_modules/*" \
  -not -path "*/.venv*/*" \
  -exec cp -v {} "$TARGET_DIR/" \;

echo "Done. Files collected in .team-messages/"

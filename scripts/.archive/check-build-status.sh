#!/usr/bin/env bash
# Check if Rust build is behind source code
# Returns 0 if build is up-to-date, 1 if rebuild needed

WORKSPACE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$WORKSPACE_ROOT"

echo "Checking build status..."

# Check if target directory exists
if [[ ! -d "target/debug" ]]; then
    echo "❌ No build artifacts found. Need to run: cargo build"
    exit 1
fi

# Get the timestamp of the most recent build artifact
LATEST_ARTIFACT=$(find target/debug -type f \( -name "*.rlib" -o -name "*.rmeta" \) 2>/dev/null | head -1)

if [[ -z "$LATEST_ARTIFACT" ]]; then
    echo "❌ No build artifacts found. Need to run: cargo build"
    exit 1
fi

echo "Latest artifact: $LATEST_ARTIFACT"

# Find any source files newer than the latest artifact
NEWER_COUNT=$(find . -type f \
    \( -name "*.rs" -o -name "Cargo.toml" -o -name "Cargo.lock" \) \
    -not -path "./target/*" \
    -not -path "./.git/*" \
    -newer "$LATEST_ARTIFACT" 2>/dev/null | wc -l)

if [[ "$NEWER_COUNT" -gt 0 ]]; then
    echo "❌ Build is BEHIND source code. $NEWER_COUNT files modified since last build."
    echo ""
    echo "Sample of modified files:"
    find . -type f \
        \( -name "*.rs" -o -name "Cargo.toml" -o -name "Cargo.lock" \) \
        -not -path "./target/*" \
        -not -path "./.git/*" \
        -newer "$LATEST_ARTIFACT" 2>/dev/null | head -5
    echo ""
    echo "Run: cargo build --workspace"
    exit 1
else
    echo "✅ Build is UP-TO-DATE with source code"
    exit 0
fi

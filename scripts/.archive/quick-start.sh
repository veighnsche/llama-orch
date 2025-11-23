#!/usr/bin/env bash
# TEAM-450: Quick start for development

set -e

echo "🐝 rbee Quick Start"
echo ""

# Install dependencies
echo "→ Installing dependencies..."
pnpm install

# Build UI package
echo "→ Building @rbee/ui..."
pnpm run build:ui

echo ""
echo "✓ Ready to develop!"
echo ""
echo "Dev commands:"
echo "  pnpm run dev:commercial  - Commercial + Marketplace"
echo "  pnpm run dev:ui          - Storybook"
echo "  pnpm run dev:all         - Everything"
echo ""

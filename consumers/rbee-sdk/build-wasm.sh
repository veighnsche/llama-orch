#!/bin/bash
# TEAM-286: Build script for WASM

set -e

echo "🔧 Building rbee-sdk for WASM..."

# Build for web (browser)
echo "📦 Building for web..."
wasm-pack build --target web --out-dir pkg/web

# Build for Node.js
echo "📦 Building for Node.js..."
wasm-pack build --target nodejs --out-dir pkg/nodejs

# Build for bundlers (webpack, vite, etc.)
echo "📦 Building for bundlers..."
wasm-pack build --target bundler --out-dir pkg/bundler

echo "✅ Build complete!"
echo ""
echo "Output:"
echo "  - pkg/web/     (for browser via <script type='module'>)"
echo "  - pkg/nodejs/  (for Node.js via require/import)"
echo "  - pkg/bundler/ (for webpack/vite/rollup)"

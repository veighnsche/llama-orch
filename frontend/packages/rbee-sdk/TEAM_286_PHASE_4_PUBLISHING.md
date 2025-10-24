# TEAM-286: Phase 4 - Publishing

**Phase:** 4 of 4  
**Duration:** 1 day  
**Status:** 📋 **READY TO START**

---

## Goal

Build the WASM package with wasm-pack, optimize for production, and publish to npm.

---

## Prerequisites

- ✅ All phases 1-3 completed
- ✅ All operations working
- ✅ Examples tested

---

## Deliverables

1. **Production Build**
   - Optimized WASM binary
   - JavaScript glue code
   - TypeScript types

2. **npm Package**
   - Published to npm registry
   - Version 0.1.0

3. **Documentation**
   - Complete README
   - CHANGELOG
   - Usage examples

---

## Step-by-Step Implementation

### Step 1: Optimize Cargo.toml (30 min)

**File:** `consumers/rbee-sdk/Cargo.toml` (add/update)

```toml
[profile.release]
# TEAM-286: Optimize for small WASM size
opt-level = "z"           # Optimize for size
lto = true                # Link-time optimization
codegen-units = 1         # Better optimization
panic = "abort"           # Smaller binary
strip = true              # Strip symbols

[profile.release.package."*"]
opt-level = "z"
```

---

### Step 2: Add Package Metadata (30 min)

**File:** `consumers/rbee-sdk/Cargo.toml` (update package section)

```toml
[package]
name = "rbee-sdk"
version = "0.1.0"
edition = "2021"
authors = ["rbee Team"]
license = "GPL-3.0-or-later"
description = "Rust SDK for rbee that compiles to WASM for browser/Node.js usage"
repository = "https://github.com/your-org/llama-orch"
homepage = "https://github.com/your-org/llama-orch/tree/main/consumers/rbee-sdk"
keywords = ["rbee", "llm", "wasm", "inference", "ai"]
categories = ["wasm", "api-bindings", "web-programming"]
readme = "README.md"

[package.metadata.wasm-pack.profile.release]
wasm-opt = ["-Oz", "--enable-mutable-globals"]
```

---

### Step 3: Build for All Targets (1 hour)

**File:** `consumers/rbee-sdk/build-all.sh`

```bash
#!/bin/bash
# TEAM-286: Build for all targets

set -e

echo "🔧 Building rbee-sdk for all targets..."

# Clean previous builds
rm -rf pkg/

# Build for web (browser via <script type="module">)
echo "📦 Building for web..."
wasm-pack build --release --target web --out-dir pkg/web

# Build for Node.js
echo "📦 Building for Node.js..."
wasm-pack build --release --target nodejs --out-dir pkg/nodejs

# Build for bundlers (webpack, vite, rollup, etc.)
echo "📦 Building for bundlers..."
wasm-pack build --release --target bundler --out-dir pkg/bundler

echo ""
echo "✅ Build complete!"
echo ""
echo "Sizes:"
du -h pkg/web/rbee_sdk_bg.wasm
du -h pkg/nodejs/rbee_sdk_bg.wasm
du -h pkg/bundler/rbee_sdk_bg.wasm
echo ""
echo "Output:"
echo "  - pkg/web/     - For browser <script type='module'>"
echo "  - pkg/nodejs/  - For Node.js require/import"
echo "  - pkg/bundler/ - For webpack/vite/rollup"
```

Make executable:
```bash
chmod +x consumers/rbee-sdk/build-all.sh
```

**Run it:**
```bash
cd consumers/rbee-sdk
./build-all.sh
```

**Expected sizes:**
- WASM binary: ~150-250KB (gzipped ~50-80KB)
- JS glue: ~10-20KB

---

### Step 4: Update package.json (1 hour)

wasm-pack auto-generates package.json, but we need to customize it:

**File:** `consumers/rbee-sdk/pkg/bundler/package.json` (edit after build)

```json
{
  "name": "@rbee/sdk",
  "version": "0.1.0",
  "description": "Rust SDK for rbee - compiles to WASM for browser/Node.js",
  "main": "rbee_sdk.js",
  "types": "rbee_sdk.d.ts",
  "files": [
    "rbee_sdk_bg.wasm",
    "rbee_sdk.js",
    "rbee_sdk.d.ts"
  ],
  "keywords": [
    "rbee",
    "llm",
    "wasm",
    "inference",
    "ai",
    "rust"
  ],
  "author": "rbee Team",
  "license": "GPL-3.0-or-later",
  "repository": {
    "type": "git",
    "url": "https://github.com/your-org/llama-orch.git"
  },
  "homepage": "https://github.com/your-org/llama-orch/tree/main/consumers/rbee-sdk"
}
```

**Better approach:** Use `wasm-pack` metadata in Cargo.toml:

**File:** `consumers/rbee-sdk/Cargo.toml` (add)

```toml
[package.metadata.wasm-pack]
# TEAM-286: Configure npm package
npm-name = "@rbee/sdk"
npm-scope = "rbee"
```

---

### Step 5: Create CHANGELOG (30 min)

**File:** `consumers/rbee-sdk/CHANGELOG.md`

```markdown
# Changelog

All notable changes to rbee-sdk will be documented in this file.

## [0.1.0] - 2025-10-24

### Added

**Core Features:**
- RbeeClient with WASM bindings
- Submit jobs and stream results via SSE
- Health check endpoint
- All 17 operations from operations-contract

**Operations:**
- System: Status
- Hive: List, Get, Status, RefreshCapabilities
- Worker: Spawn, ProcessList, ProcessGet, ProcessDelete
- Active Worker: List, Get, Retire
- Model: Download, List, Get, Delete
- Inference: Streaming and non-streaming

**Developer Experience:**
- Auto-generated TypeScript types
- OperationBuilder for easy operation construction
- Convenience methods (infer, status, listHives)
- Examples for browser and Node.js

**Architecture:**
- Reuses job-client shared crate
- Reuses operations-contract
- Zero duplication with backend
- WASM binary ~150-250KB

### Known Issues

- SSE streaming requires modern browser with EventSource support
- Node.js requires v18+ for native fetch
- WASM binary larger than pure JS (but faster)

## Future Plans

- v0.2.0: React hooks wrapper
- v0.3.0: Vue/Svelte integrations
- v1.0.0: Stable API
```

---

### Step 6: Publishing Process (2 hours)

**Pre-publish checklist:**

**File:** `consumers/rbee-sdk/pre-publish.sh`

```bash
#!/bin/bash
# TEAM-286: Pre-publish checks

set -e

echo "🔍 Running pre-publish checks..."

echo "1. Running Rust tests..."
cargo test

echo "2. Running WASM tests..."
wasm-pack test --headless --firefox

echo "3. Building for all targets..."
./build-all.sh

echo "4. Checking package sizes..."
ls -lh pkg/bundler/*.wasm

echo "5. Verifying TypeScript types..."
ls pkg/bundler/*.d.ts

echo "✅ All checks passed!"
echo ""
echo "Ready to publish!"
echo ""
echo "Commands:"
echo "  cd pkg/bundler"
echo "  npm publish --access public"
```

Make executable:
```bash
chmod +x consumers/rbee-sdk/pre-publish.sh
```

**Run checks:**
```bash
./pre-publish.sh
```

**Publish to npm:**

```bash
# Login to npm (first time only)
npm login

# Publish bundler target (most common)
cd pkg/bundler
npm publish --access public

# Or publish with specific tag
npm publish --access public --tag latest
```

**Verify publication:**
```bash
npm view @rbee/sdk
```

---

### Step 7: GitHub Release (30 min)

```bash
# Tag the release
git tag -a v0.1.0 -m "Release v0.1.0 - Initial WASM SDK"

# Push tag
git push origin v0.1.0
```

**Create GitHub release:**
1. Go to GitHub releases page
2. Click "Create new release"
3. Select tag v0.1.0
4. Title: "v0.1.0 - Initial Release"
5. Copy CHANGELOG entry
6. Publish release

---

### Step 8: Documentation Site (optional, 1 hour)

**Using GitHub Pages:**

**File:** `consumers/rbee-sdk/docs/index.html`

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>rbee SDK Documentation</title>
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/@picocss/pico@1/css/pico.min.css">
</head>
<body>
    <main class="container">
        <h1>🐝 rbee SDK</h1>
        <p>Rust SDK that compiles to WASM for browser/Node.js usage.</p>
        
        <h2>Installation</h2>
        <pre><code>npm install @rbee/sdk</code></pre>
        
        <h2>Quick Start</h2>
        <pre><code>import init, { RbeeClient } from '@rbee/sdk';

await init();
const client = new RbeeClient('http://localhost:8500');

await client.infer({
  model: 'llama-3-8b',
  prompt: 'Hello!',
  max_tokens: 100,
}, (token) => console.log(token));</code></pre>
        
        <h2>Links</h2>
        <ul>
            <li><a href="https://github.com/your-org/llama-orch">GitHub</a></li>
            <li><a href="https://www.npmjs.com/package/@rbee/sdk">npm</a></li>
            <li><a href="examples.html">Examples</a></li>
        </ul>
    </main>
</body>
</html>
```

---

## Verification Checklist

After completing Phase 4:

- [ ] All targets build successfully (web, nodejs, bundler)
- [ ] WASM binary is optimized (<250KB)
- [ ] TypeScript types are generated
- [ ] Pre-publish checks pass
- [ ] Package published to npm
- [ ] GitHub release created
- [ ] Documentation complete
- [ ] Examples work with published package
- [ ] All TEAM-286 signatures added

---

## Post-Publication

### Test the Published Package

```bash
# Create test project
mkdir test-rbee-sdk
cd test-rbee-sdk
npm init -y
npm install @rbee/sdk

# Create test file
cat > test.js << 'EOF'
const { RbeeClient } = require('@rbee/sdk');

async function main() {
    const client = new RbeeClient('http://localhost:8500');
    console.log('Client created:', client.base_url);
}

main();
EOF

# Run it
node test.js
```

### Update Main README

Add link to SDK in main project README:

```markdown
## SDKs

- **Rust + WASM**: [`@rbee/sdk`](./consumers/rbee-sdk) - For browser and Node.js
```

---

## Success Criteria

### MVP (v0.1.0) ✅

- Published to npm as `@rbee/sdk`
- WASM binary optimized
- TypeScript types included
- All 17 operations working
- Examples provided
- Documentation complete

### Future Versions

- **v0.2.0**: React hooks wrapper (pure JS package)
- **v0.3.0**: Vue/Svelte integrations
- **v1.0.0**: Stable API, no breaking changes

---

## Bundle Size Comparison

| Implementation | Size (gzipped) | Performance |
|----------------|----------------|-------------|
| Pure TypeScript | ~15-20KB | Good |
| **Rust + WASM** | **~50-80KB** | **Excellent** |
| With React hooks | +5-10KB | Good |

**Worth it?** YES - for:
- Zero type drift
- Code reuse with backend
- Better performance
- Single codebase maintenance

---

## Announcement Template

```markdown
🚀 rbee SDK v0.1.0 Released!

We're excited to announce the first release of rbee SDK - a Rust SDK that 
compiles to WASM for browser and Node.js!

**Features:**
✅ All 17 rbee operations
✅ Auto-generated TypeScript types
✅ Zero type drift with backend
✅ ~50-80KB gzipped

**Install:**
npm install @rbee/sdk

**Docs:**
https://github.com/your-org/llama-orch/tree/main/consumers/rbee-sdk

**Built with:**
- Rust + wasm-bindgen
- Reuses backend shared crates
- Zero code duplication
```

---

**Congratulations! The SDK is published! 🎉**

---

**Created by:** TEAM-286  
**Date:** Oct 24, 2025  
**Estimated Duration:** 1 day

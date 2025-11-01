# TEAM-381: WASM SDK & Config Package Watchers

**Date:** 2025-11-01  
**Status:** ⚠️ NEEDS ATTENTION

## Question
> "Don't we need to put watchers on SDK packages and config packages?"

## Answer: YES, but it's PARTIALLY working

### Current State

#### ✅ What's Already Watched (Vite Apps)
```json
// bin/20_rbee_hive/ui/app/package.json
{
  "scripts": {
    "dev": "vite"  // ← Watches .tsx, .ts, .css files
  }
}
```

**Vite watches:**
- ✅ App source files (`src/**/*.tsx`, `src/**/*.ts`)
- ✅ CSS files
- ✅ Config files (`vite.config.ts`, `tsconfig.json`)

#### ❌ What's NOT Watched (WASM SDK)
```json
// bin/20_rbee_hive/ui/packages/rbee-hive-sdk/package.json
{
  "scripts": {
    "build": "wasm-pack build --target bundler --out-dir pkg/bundler"
    // ❌ NO "dev" script - no watcher!
  }
}
```

**Problem:** WASM SDK packages are **Rust code** that compiles to WASM:
- Source: `src/lib.rs`, `src/operations.rs`, etc.
- Build: `wasm-pack build` (slow, ~5-10 seconds)
- Output: `pkg/bundler/*.js`, `pkg/bundler/*.wasm`

**Current behavior:**
- ✅ Vite detects when `pkg/bundler/*.js` changes
- ✅ Vite hot-reloads the app
- ❌ But `pkg/` doesn't auto-rebuild when `src/*.rs` changes!

#### ⚠️ Config Packages (Partially Watched)
```json
// frontend/packages/tailwind-config/package.json
{
  "exports": {
    ".": "./shared-styles.css"
  }
  // ❌ NO "dev" script
}
```

**Vite watches CSS imports:**
- ✅ If you change `shared-styles.css`, Vite detects it
- ✅ Hot reload works
- ✅ No build step needed (it's just CSS)

**Verdict:** Config packages are fine (no build step needed).

---

## The Problem: WASM SDK Needs Manual Rebuild

### Current Workflow (BROKEN)
```bash
# Terminal 1: Start Vite dev server
cd bin/20_rbee_hive/ui/app
pnpm dev

# Edit SDK Rust code
vim ../packages/rbee-hive-sdk/src/operations.rs

# ❌ PROBLEM: Changes don't appear in browser!
# You must manually rebuild:
cd ../packages/rbee-hive-sdk
pnpm build  # ← Manual step!

# ✅ NOW Vite detects pkg/ change and hot-reloads
```

### Why This Happens
1. **WASM SDK is Rust** - needs `wasm-pack` to compile
2. **Vite only watches JS/TS** - doesn't know about `.rs` files
3. **No watcher on SDK** - `wasm-pack` doesn't have a watch mode

---

## Solution: Add cargo-watch for WASM SDK

### Option 1: Add `dev` Script with cargo-watch (RECOMMENDED)

**Install cargo-watch** (if not already installed):
```bash
cargo install cargo-watch
```

**Add to SDK package.json:**
```json
// bin/20_rbee_hive/ui/packages/rbee-hive-sdk/package.json
{
  "scripts": {
    "build": "wasm-pack build --target bundler --out-dir pkg/bundler",
    "dev": "cargo watch -i 'pkg/*' -s 'wasm-pack build --target bundler --out-dir pkg/bundler'"
  }
}
```

**What this does:**
- `cargo watch` monitors `src/*.rs` files
- `-i 'pkg/*'` ignores output directory (prevents rebuild loop)
- `-s '...'` runs `wasm-pack build` on any Rust file change
- Vite detects `pkg/` change and hot-reloads

**Usage:**
```bash
# Terminal 1: Watch SDK
cd bin/20_rbee_hive/ui/packages/rbee-hive-sdk
pnpm dev  # ← Now watches Rust files!

# Terminal 2: Run Vite
cd ../../app
pnpm dev

# Edit src/operations.rs → Auto-rebuild → Auto-reload ✅
```

### Option 2: Use Turbo with SDK dev script

**Update root package.json:**
```json
{
  "scripts": {
    "dev:hive": "turbo dev --filter=@rbee/rbee-hive-sdk --filter=@rbee/rbee-hive-ui"
  }
}
```

**Then run:**
```bash
pnpm dev:hive
```

**Turbo will:**
- ✅ Start SDK watcher (if dev script exists)
- ✅ Start Vite dev server
- ✅ Coordinate both processes

---

## Recommended Setup

### For Each WASM SDK Package

**1. rbee-hive-sdk**
```bash
cd bin/20_rbee_hive/ui/packages/rbee-hive-sdk
```

**Add to package.json:**
```json
{
  "scripts": {
    "build": "wasm-pack build --target bundler --out-dir pkg/bundler",
    "dev": "cargo watch -i 'pkg/*' -s 'wasm-pack build --target bundler --out-dir pkg/bundler'"
  }
}
```

**2. queen-rbee-sdk**
```bash
cd bin/10_queen_rbee/ui/packages/queen-rbee-sdk
```

**Add to package.json:**
```json
{
  "scripts": {
    "build": "wasm-pack build --target bundler --out-dir pkg/bundler",
    "dev": "cargo watch -i 'pkg/*' -s 'wasm-pack build --target bundler --out-dir pkg/bundler'"
  }
}
```

**3. llm-worker-sdk** (if exists)
```bash
cd bin/30_llm_worker_rbee/ui/packages/llm-worker-sdk
```

**Add to package.json:**
```json
{
  "scripts": {
    "build": "wasm-pack build --target bundler --out-dir pkg/bundler",
    "dev": "cargo watch -i 'pkg/*' -s 'wasm-pack build --target bundler --out-dir pkg/bundler'"
  }
}
```

### Update Turbo Config

**Add SDK packages to turbo dev:**
```json
// package.json (root)
{
  "scripts": {
    "dev:hive": "turbo dev --filter=@rbee/rbee-hive-sdk --filter=@rbee/rbee-hive-ui",
    "dev:queen": "turbo dev --filter=@rbee/queen-rbee-sdk --filter=@rbee/queen-rbee-react --filter=@rbee/queen-rbee-ui",
    "dev:product": "turbo dev --filter='@rbee/*-sdk' --filter='@rbee/*-ui'"
  }
}
```

---

## Summary

### What Needs Watchers

| Package Type | Needs Watcher? | Current State | Solution |
|--------------|----------------|---------------|----------|
| **Vite Apps** (rbee-hive-ui) | ✅ Yes | ✅ Has `dev` script | Working |
| **WASM SDKs** (rbee-hive-sdk) | ✅ Yes | ❌ No `dev` script | **Add cargo-watch** |
| **React Packages** (rbee-hive-react) | ⚠️ Maybe | ❌ No `dev` script | Vite watches imports |
| **Config Packages** (tailwind-config) | ❌ No | ✅ Static CSS | Working |

### Action Items

**CRITICAL - Add SDK watchers:**
```bash
# 1. Install cargo-watch (if needed)
cargo install cargo-watch

# 2. Add dev scripts to all SDK packages
# See "Recommended Setup" section above

# 3. Update root package.json with new dev commands
# See "Update Turbo Config" section above

# 4. Test the workflow
pnpm dev:hive
# Edit bin/20_rbee_hive/ui/packages/rbee-hive-sdk/src/operations.rs
# Should auto-rebuild and hot-reload ✅
```

**OPTIONAL - Add React package watchers:**
If you want React packages to rebuild on change (not usually needed with Vite):
```json
// bin/20_rbee_hive/ui/packages/rbee-hive-react/package.json
{
  "scripts": {
    "dev": "tsc --watch"  // Watch TypeScript compilation
  }
}
```

But **Vite already handles this** - it compiles imported `.ts` files on-the-fly, so this is usually unnecessary.

---

## Why This Matters

**Without SDK watchers:**
- ❌ Edit Rust code → No rebuild
- ❌ Must manually run `pnpm build` in SDK package
- ❌ Slow feedback loop (30+ seconds)
- ❌ Easy to forget to rebuild

**With SDK watchers:**
- ✅ Edit Rust code → Auto-rebuild (5-10 seconds)
- ✅ Vite detects change → Hot-reload (instant)
- ✅ Fast feedback loop (~10 seconds total)
- ✅ No manual steps

**This is the missing piece for true hot reload!**

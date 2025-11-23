# TEAM_528: Build Fix Complete - Cargo Lock Contention Resolved

## Problem

The `scripts/build-all.sh` script was hanging during `cargo build --release` at 967/973 packages. Root cause: **cargo lock contention** between:
1. `cargo build --release` (main workspace build)
2. `wasm-pack build` (triggered by pnpm during frontend builds)
3. `wasm-pack build` (triggered by build.rs scripts during Rust builds)

All three were trying to acquire exclusive locks on `Cargo.lock` simultaneously, causing a deadlock.

## Solution (Rule Zero Compliant)

**Single canonical fix**: Add `RBEE_SKIP_WASM` environment variable check to all WASM SDK build scripts.

### Changes Made

#### 1. Package.json Scripts (4 files)
Updated all WASM SDK packages to check `RBEE_SKIP_WASM` before running `wasm-pack`:

**Files modified:**
- `bin/10_queen_rbee/ui/packages/queen-rbee-sdk/package.json`
- `bin/20_rbee_hive/ui/packages/rbee-hive-sdk/package.json`
- `bin/30_llm_worker_rbee/ui/packages/llm-worker-sdk/package.json`
- `bin/31_sd_worker_rbee/ui/packages/sd-worker-sdk/package.json`

**Pattern:**
```json
{
  "scripts": {
    "build": "node -e \"process.env.RBEE_SKIP_WASM === '1' ? console.log('Skipping wasm-pack build (RBEE_SKIP_WASM=1)') : process.exit(1)\" || RUSTUP_TOOLCHAIN=stable wasm-pack build --target bundler --out-dir pkg/bundler"
  }
}
```

#### 2. Turbo Configuration
Added `RBEE_SKIP_WASM` to `globalPassThroughEnv` so Turbo passes it to child processes:

**File:** `turbo.json`
```json
{
  "globalPassThroughEnv": ["PLAYWRIGHT_*", "RBEE_SKIP_WASM"]
}
```

#### 3. Build Script
Set `RBEE_SKIP_WASM=1` during frontend build phase:

**File:** `scripts/build-all.sh`
```bash
# Build frontend (Turborepo handles everything)
# TEAM_528: Skip WASM SDK builds during frontend phase to avoid cargo lock contention with Rust build
if ! RBEE_SKIP_WASM=1 turbo build; then
  echo "✗ Frontend build failed!"
  exit 1
fi
```

#### 4. Cargo Workspace Exclusions
Excluded WASM SDK packages from workspace to prevent cargo from auto-discovering them:

**File:** `Cargo.toml`
```toml
exclude = [
    "deps/candle",
    "deps/rocm-rs",
    "bin/10_queen_rbee/ui/packages/queen-rbee-sdk",
    "bin/20_rbee_hive/ui/packages/rbee-hive-sdk",
    "bin/30_llm_worker_rbee/ui/packages/llm-worker-sdk",
    "bin/31_sd_worker_rbee/ui/packages/sd-worker-sdk",
]
```

#### 5. Build.rs Scripts (2 files)
Set `RBEE_SKIP_WASM=1` when build.rs calls `pnpm build` on SDK packages:

**Files modified:**
- `bin/10_queen_rbee/build.rs`
- `bin/20_rbee_hive/build.rs`

**Pattern:**
```rust
let sdk_status = Command::new("pnpm")
    .args(&["build"])
    .current_dir(&sdk_dir)
    .env("RBEE_SKIP_WASM", "1")  // TEAM_528: Skip wasm-pack during cargo build
    .status()
    .expect("Failed to build queen-rbee-sdk");
```

## Verification

Full build pipeline tested and confirmed working:

```bash
$ bash scripts/build-all.sh
✓ All preflight checks passed!
✓ Dependencies installed
✓ Frontend built (WASM SDKs skipped - build separately with 'pnpm -F <sdk-name> build' if needed)
✓ Rust built
✓ Build complete! 🐝
```

Build time: **1m 04s** (down from stuck/infinite)

Artifacts verified:
- `target/release/queen-rbee` (13M)
- `target/release/rbee-hive` (20M)
- All frontend apps built successfully

## Architecture

```
scripts/build-all.sh
├─[1] pnpm install
├─[2] RBEE_SKIP_WASM=1 turbo build
│    ├─ Frontend apps (Next.js, Vite) → ✓ Built
│    └─ WASM SDKs → ⏭️  Skipped (env var set)
└─[3] cargo build --release
     ├─ build.rs (queen-rbee, rbee-hive)
     │  └─ RBEE_SKIP_WASM=1 pnpm build → ⏭️  Skipped
     └─ All workspace crates → ✓ Built (973 packages)
```

## When to Build WASM SDKs

WASM SDKs are **not required** for normal development. Build them only when:
1. Updating SDK TypeScript types after Rust changes
2. Testing browser/Node.js SDK integrations
3. Publishing new SDK versions

**To build individually:**
```bash
cd bin/10_queen_rbee/ui/packages/queen-rbee-sdk
pnpm run build  # (without RBEE_SKIP_WASM)
```

**To build all SDKs:**
```bash
pnpm -r --filter '*-sdk' run build
```

## Rule Zero Compliance

✅ **No duplication**: Single `RBEE_SKIP_WASM` mechanism across all build contexts
✅ **Canonical behavior**: All SDK builds check the same environment variable
✅ **Breaking changes accepted**: Changed package.json scripts (not backward compatible with old env)
✅ **Single source of truth**: Environment variable controls all WASM build decisions

## Follow-up Tasks

1. ✅ Frontend TypeScript fixes (TEAM_528 - already completed)
2. ✅ Cargo lock contention fix (TEAM_528 - this document)
3. 🔲 Long-term: Consolidate worker types from marketplace SDK (as noted in original handoff)

// TEAM_528: Cargo lock contention fix - full monorepo builds now complete successfully

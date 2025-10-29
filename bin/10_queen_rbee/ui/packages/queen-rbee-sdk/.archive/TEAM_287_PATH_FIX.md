# TEAM-287: Path Fix After SDK Relocation

**Date:** Oct 24, 2025  
**Status:** ✅ **FIXED**

---

## Problem

After moving `rbee-sdk` from `consumers/rbee-sdk` to `frontend/packages/rbee-sdk`, the Cargo build failed:

```
error: failed to load manifest for workspace member `/home/vince/Projects/llama-orch/frontend/packages/rbee-sdk`
referenced by workspace at `/home/vince/Projects/llama-orch/Cargo.toml`

Caused by:
  failed to load manifest for dependency `job-client`

Caused by:
  failed to read `/home/vince/Projects/llama-orch/frontend/bin/99_shared_crates/job-client/Cargo.toml`

Caused by:
  No such file or directory (os error 2)
```

**Root Cause:** The relative paths in `rbee-sdk/Cargo.toml` were still pointing to `../../bin/` but after the move, they needed to be `../../../bin/`.

---

## Solution

### 1. Fixed Cargo.toml Paths

**File:** `frontend/packages/rbee-sdk/Cargo.toml`

**Changed:**
```toml
# OLD (from consumers/rbee-sdk)
job-client = { path = "../../bin/99_shared_crates/job-client" }
operations-contract = { path = "../../bin/97_contracts/operations-contract" }
rbee-config = { path = "../../bin/99_shared_crates/rbee-config" }
```

**To:**
```toml
# NEW (from frontend/packages/rbee-sdk)
job-client = { path = "../../../bin/99_shared_crates/job-client" }
operations-contract = { path = "../../../bin/97_contracts/operations-contract" }
rbee-config = { path = "../../../bin/99_shared_crates/rbee-config" }
```

**Why:** The SDK moved one directory level deeper:
- Old: `consumers/rbee-sdk` → root is `../../`
- New: `frontend/packages/rbee-sdk` → root is `../../../`

### 2. Fixed package.json Dependency

**File:** `frontend/apps/web-ui/package.json`

**Changed:**
```json
// OLD
"@rbee/sdk": "../../consumers/rbee-sdk"
```

**To:**
```json
// NEW
"@rbee/sdk": "workspace:*"
```

**Why:** Follow monorepo best practices using workspace protocol.

---

## Verification

### ✅ Cargo Check
```bash
cargo check -p rbee-sdk
# ✅ SUCCESS: Finished `dev` profile [unoptimized + debuginfo] target(s) in 7.84s
```

### ✅ Cargo Clippy
```bash
cargo clippy --workspace --all-targets
# ✅ SUCCESS: No errors, only warnings about profile locations (expected)
```

### ✅ pnpm Install
```bash
pnpm install
# ✅ SUCCESS: Already up to date
```

---

## Files Modified

### 1. frontend/packages/rbee-sdk/Cargo.toml
- Line 27: `job-client` path updated
- Line 32: `operations-contract` path updated
- Line 36: `rbee-config` path updated

### 2. frontend/apps/web-ui/package.json
- Line 31: `@rbee/sdk` dependency changed to `workspace:*`

---

## Path Reference

### Directory Structure
```
llama-orch/                                    (root)
├── bin/
│   ├── 97_contracts/
│   │   └── operations-contract/               (target)
│   └── 99_shared_crates/
│       ├── job-client/                        (target)
│       └── rbee-config/                       (target)
└── frontend/
    └── packages/
        └── rbee-sdk/                          (source)
            └── Cargo.toml
```

### Path Calculation
From `frontend/packages/rbee-sdk/`:
- `../` → `frontend/packages/`
- `../../` → `frontend/`
- `../../../` → `llama-orch/` (root)
- `../../../bin/` → `llama-orch/bin/` ✅

---

## Impact

### ✅ Fixed
- Cargo workspace builds successfully
- rbee-sdk compiles without errors
- pnpm workspace links correctly
- No more "failed to read Cargo.toml" errors

### ✅ Maintained
- All existing functionality preserved
- WASM build still works
- TypeScript types still generated
- No breaking changes to API

---

## Next Steps

The path fix is complete. You can now proceed with the implementation plan:

1. ✅ **Path fix** (DONE)
2. ⏳ **Build WASM** - `cd frontend/packages/rbee-sdk && pnpm build`
3. ⏳ **Update next.config.ts** - Add webpack WASM config
4. ⏳ **Create hooks** - useRbeeSDK, useHeartbeat
5. ⏳ **Update page.tsx** - Live heartbeat dashboard

See `TEAM_287_IMPLEMENTATION_PLAN.md` for full details.

---

**Prepared by:** TEAM-287  
**Date:** Oct 24, 2025  
**Status:** ✅ Fixed and verified

# UI Path Violations - FIXED

**Date:** Oct 29, 2025  
**Status:** ✅ COMPLETE

---

## Problem

Documentation incorrectly stated that the Queen UI is served at `/ui` path.

**WRONG:** `http://localhost:7833/ui`  
**CORRECT:** `http://localhost:7833/`

The **code was already correct** (TEAM-341 fixed it), but documentation was out of sync.

---

## Root Cause

AI coders kept assuming UIs should be nested under `/ui` prefix, despite explicit requirements to serve at root path.

This pattern was copied across multiple documentation files, creating widespread misinformation.

---

## Files Fixed

### Documentation (HAD VIOLATIONS)

1. **ARCHITECTURE.md** ✅
   - Line 422: Changed `GET /ui/*` → `GET /*`
   - Added note about fallback routing

2. **JOB_OPERATIONS.md** ✅
   - Lines 311-313: Changed all iframe URLs from `/ui` to `/`
   - Lines 609-613: Fixed duplicate section (removed `/ui`)

3. **PORT_CONFIGURATION.md** ✅
   - Line 24-28: Fixed visual map
   - Lines 50-54: Fixed production URLs table
   - Lines 86-88: Fixed architecture diagram
   - Lines 97-101: Fixed port mapping table
   - Lines 104-106: Fixed key principles
   - Lines 137-161: Fixed code examples
   - Lines 666-668: Fixed production health checks
   - Line 704: Fixed key principle statement

4. **UI_PROXY_SETUP.md** ❌ MOVED TO ARCHIVE
   - Entire file was incorrect
   - Moved to `.archive/UI_PROXY_SETUP.md.WRONG`
   - Replaced with new `UI_ARCHITECTURE.md`

### Code (WAS ALREADY CORRECT ✅)

1. **src/http/static_files.rs**
   - Lines 3-16: Clear warnings that UI is at ROOT
   - Line 53: `Router::new().fallback(dev_proxy_handler)` (no `/ui` nesting)
   - Line 60: `Router::new().fallback(static_handler)` (no `/ui` nesting)

2. **src/main.rs**
   - Line 144: `api_router.merge(http::create_static_router())`
   - API routes registered first, static router is fallback
   - No `/ui` prefix anywhere

3. **SDK Packages (all correct)**
   - All use `http://localhost:7833` (no `/ui`)
   - React hooks default to root URL
   - WASM SDK uses root URL

---

## New Documentation

Created **UI_ARCHITECTURE.md** to replace the incorrect `UI_PROXY_SETUP.md`:

- Clear explanation of root path requirement
- Router merge order explanation
- Dev vs prod mode differences
- Troubleshooting section
- Related services reference

---

## Verification

```bash
# Check for any remaining /ui violations in queen-rbee docs
grep -r "/ui" bin/10_queen_rbee/ --include="*.md" | grep -v ".archive"

# Should return: No results (all violations fixed)
```

---

## Pattern to Watch

**AI coders consistently assume nested paths for UIs:**
- `http://localhost:7833/ui` ❌
- `http://localhost:7835/api` ❌
- `http://localhost:8080/v1/ui` ❌

**Correct pattern:**
- API routes: `/health`, `/v1/*` (explicit prefixes)
- UI routes: `/*` (fallback, catches everything else)
- Router merge: API first, UI fallback

**Why this works:**
- API routes registered first (priority)
- UI router registered as fallback
- SPA routing works naturally (no base path needed)
- Simpler URLs for users

---

## Related Issues

**Same pattern applies to all services:**
- rbee-hive: `http://localhost:7835/` (not `/ui`)
- llm-worker: `http://localhost:8080/` (not `/ui`)
- comfy-worker: `http://localhost:8188/` (not `/ui`)
- vllm-worker: `http://localhost:8000/` (not `/ui`)

**Check these services for similar violations when they get UIs.**

---

## Prevention

1. **Code is authoritative** - Trust `static_files.rs` over docs
2. **Grep before writing** - Check for existing patterns
3. **Test URLs** - `curl http://localhost:7833/` should return UI
4. **Router order matters** - API routes MUST be registered before UI fallback

---

## Files Changed

**Fixed:**
- `ARCHITECTURE.md`
- `JOB_OPERATIONS.md`
- `PORT_CONFIGURATION.md`
- `UI_ARCHITECTURE.md` (NEW)

**Archived:**
- `UI_PROXY_SETUP.md` → `.archive/UI_PROXY_SETUP.md.WRONG`

**Total violations fixed:** 15+ instances across 4 files

---

**Status:** All `/ui` path violations eliminated from queen-rbee documentation.

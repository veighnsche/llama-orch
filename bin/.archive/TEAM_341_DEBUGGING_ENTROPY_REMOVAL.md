# TEAM-341: Debugging Entropy Removal Checklist

**Date:** Oct 29, 2025  
**Status:** ✅ CLEANED

---

## Problem

During debugging of WASM MIME type issue, added configuration entropy that didn't solve the root cause.

**Rule:** Fix problems at the right layer, not by adding configuration entropy.

---

## Entropy Added (REMOVED)

### ❌ vite-plugin-top-level-await

**What was added:**
```typescript
// vite.config.ts
import topLevelAwait from 'vite-plugin-top-level-await'

plugins: [
  wasm(),
  topLevelAwait(),  // ← ENTROPY!
  react(),
]
```

**Why it was wrong:**
- Vite already handles top-level await in WASM files
- Problem was at the proxy layer (HTTP headers), not Vite config
- Added dependency that does nothing
- Increased configuration complexity

**Removed:**
```bash
pnpm remove vite-plugin-top-level-await
# Reverted vite.config.ts to clean state
```

---

## Correct Fix

### ✅ Fix at Proxy Layer

**File:** `bin/10_queen_rbee/src/http/static_files.rs`

**Problem:** Dev proxy forwards headers from Vite as-is, but browsers require strict MIME types for WASM module imports.

**Solution:** Override `Content-Type` header for `.wasm` files at proxy level:

```rust
// TEAM-341: Fix WASM MIME type for ES module imports
if path.ends_with(".wasm") {
    builder = builder.header(header::CONTENT_TYPE, "application/wasm");
}
```

**Why this is correct:**
- Fixes the problem at the source (HTTP proxy)
- No configuration entropy
- Works for both dev and prod (though prod uses embedded files)
- Single line of code in the right place

---

## Debugging Entropy Prevention

### When Debugging, Ask:

1. **What layer is the problem?**
   - UI/Frontend config? (Vite, React, etc.)
   - HTTP proxy? (Axum, static_files.rs)
   - Backend logic? (Rust handlers)

2. **Am I adding configuration or fixing logic?**
   - ❌ Configuration = entropy (adds complexity)
   - ✅ Logic fixes = correct solution

3. **Does this dependency solve my actual problem?**
   - If you're not sure, DON'T add it
   - Research first, understand what it does
   - Test if it actually fixes the issue

4. **Can I fix it with existing tools?**
   - Use what's already there
   - Don't add new dependencies for debugging

### Entropy Checklist

Before adding ANY configuration change while debugging:

- [ ] Do I understand the root cause?
- [ ] Is this the right layer to fix it?
- [ ] Does this dependency actually solve my problem?
- [ ] Am I adding complexity without benefit?
- [ ] Can I fix it with existing code?

**If you can't check ALL boxes, don't add it.**

---

## What To Remove After Debugging

### Always Remove These:

1. **Debug plugins** that didn't help
2. **Extra dependencies** that did nothing
3. **Configuration changes** that didn't fix the issue
4. **Workarounds** that hide the real problem

### Keep Only:

1. **The minimal fix** that solves the root cause
2. **At the correct layer** (don't fix UI problems in backend config)

---

## This Issue's Timeline

1. ❌ Added `vite-plugin-top-level-await` (entropy)
2. ❌ Tested - didn't fix the issue
3. ✅ Removed dependency immediately (clean up)
4. ✅ Fixed at correct layer (HTTP proxy)
5. ✅ Documented for future reference

**Result:** 1 line of code in the right place, not 3 lines of config entropy.

---

## Lesson Learned

**Configuration is not the solution to logic problems.**

When browsers complain about MIME types:
- Fix the HTTP headers (logic)
- Don't add plugins (configuration)

When proxies don't work right:
- Fix the proxy code
- Don't reconfigure the upstream server

**Fix problems at their source, not by layering workarounds.**

---

## Files Changed

**Reverted (entropy removed):**
- `bin/10_queen_rbee/ui/app/vite.config.ts` - Removed topLevelAwait import and usage
- `bin/10_queen_rbee/ui/app/package.json` - Removed vite-plugin-top-level-await dependency

**Fixed (correct solution):**
- `bin/10_queen_rbee/src/http/static_files.rs` - Override Content-Type for .wasm files in proxy

---

**TEAM-341 ENTROPY CLEANED** ✅

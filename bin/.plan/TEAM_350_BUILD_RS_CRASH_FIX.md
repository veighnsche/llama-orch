# TEAM-350: Fixed build.rs Crashing Dev Server

**Status:** ✅ COMPLETE

## The Bug You Found

**Excellent catch!** When you press "Rebuild" in the GUI while `dev:queen` is running, the Turbo dev server crashes.

### Root Cause

**File:** `bin/10_queen_rbee/build.rs`

The build script runs `vite build` (production build) **every time** `cargo build` is executed:

```rust
// OLD CODE (BROKEN)
let status = Command::new("pnpm")
    .args(&["exec", "vite", "build"])  // ← Runs EVERY cargo build!
    .current_dir(&ui_app_dir)
    .status()
```

**The Conflict:**

1. **Dev Server Running:** `pnpm run dev:queen` → Turbo watches `ui/app/dist/`
2. **GUI Rebuild Pressed:** Triggers backend rebuild → `cargo build`
3. **build.rs Runs:** Executes `vite build` → **Overwrites `dist/` files**
4. **Dev Server Crash:** Turbo sees file changes → **ELIFECYCLE Command failed**

### Why This Happens

- `build.rs` runs **every time** you compile the Rust binary
- It rebuilds the **entire UI** (production build)
- This **overwrites** the files that the dev server is watching/serving
- The dev server **can't reconcile** the sudden file changes
- Result: **CRASH** 💥

## The Fix

**TEAM-350: Skip UI build in dev mode if dist already exists**

```rust
// NEW CODE (FIXED)
let profile = std::env::var("PROFILE").unwrap_or_else(|_| "debug".to_string());
let is_dev = profile == "debug";

if is_dev && ui_dist.exists() {
    // Dev mode: Skip UI build to avoid conflicting with turbo dev server
    println!("cargo:warning=⚡ Dev mode: Skipping UI build (dist exists)");
    return;  // ← Exit early!
}

// Only build UI in production mode OR if dist doesn't exist
```

## Behavior After Fix

### Development Mode (cargo build)
```
⚡ Dev mode: Skipping UI build (dist exists, dev server should be running)
   If UI is stale, run: cd bin/10_queen_rbee/ui/app && pnpm exec vite build
```

**Result:** No conflict, dev server keeps running ✅

### Production Mode (cargo build --release)
```
🔨 Building queen-rbee UI for embedding...
✅ queen-rbee UI built successfully
```

**Result:** UI is built and embedded in binary ✅

### First Build (dist doesn't exist)
```
🔨 Building queen-rbee UI for embedding...
✅ queen-rbee UI built successfully
```

**Result:** UI is built once, then dev mode skips it ✅

## Testing the Fix

1. **Kill the current dev server** (it's probably crashed)
2. **Rebuild queen-rbee** to get the new build.rs:
   ```bash
   cargo build --bin queen-rbee
   ```
3. **Start dev server again:**
   ```bash
   pnpm run dev:queen
   ```
4. **Press "Rebuild" in GUI** → Should NOT crash now!

## Related Issue: Stale Cache

If you still see old logs after the fix, you need to:

1. **Hard refresh browser:** Ctrl+Shift+R (or Cmd+Shift+R on Mac)
2. **Or clear browser cache completely**

The dev server serves files with cache headers, so old bundles can stick around.

## Files Changed

1. **bin/10_queen_rbee/build.rs** - Skip UI build in dev mode

## Why This Pattern Is Correct

- **Dev mode:** Dev server handles live reloading, no need to rebuild UI
- **Production mode:** UI must be embedded in binary, so we build it
- **First build:** If dist doesn't exist, we need to create it once

This prevents the conflict while maintaining correct production builds.

---

**TEAM-350 Signature:** Fixed build.rs crashing dev server when GUI rebuild is pressed

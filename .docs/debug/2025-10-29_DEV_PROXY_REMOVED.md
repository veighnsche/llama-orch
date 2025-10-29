# Dev Proxy Removed - Simplified Workflow (2025-10-29)

## Decision

**REMOVED** the entire dev proxy system. Queen now **always serves embedded static files** from `ui/app/dist/`, even in debug mode.

## Rationale

The dev proxy added complexity and caused issues:
- ❌ Vite HMR WebSocket port confusion (7833 vs 7834)
- ❌ Keeper iframe loading issues
- ❌ Two different ways to access UI (confusing)
- ❌ RBEE_SKIP_UI_BUILD environment variable complexity
- ❌ Turbo dev server conflicts during install

**New workflow is simpler:**
- ✅ One way to access UI: `http://localhost:7833`
- ✅ No port confusion
- ✅ No WebSocket issues
- ✅ No special environment variables
- ✅ Consistent behavior in debug and release

## New Workflow

### UI Development

**Every UI change requires a queen rebuild:**

```bash
# Make changes to UI files in bin/10_queen_rbee/ui/app/src/

# Rebuild queen (this builds UI and embeds it)
cargo build -p queen-rbee

# Restart queen
rbee queen stop
rbee queen start

# Access UI at http://localhost:7833
```

### Why This Is Better

1. **Simpler:** One workflow, no special cases
2. **Reliable:** No proxy, no WebSocket issues
3. **Consistent:** Same behavior in debug and release
4. **Predictable:** What you build is what you get
5. **No Surprises:** No HMR, no hot reload, no magic

### Trade-offs

**Lost:**
- ❌ Vite HMR (hot module replacement)
- ❌ Instant UI updates without rebuild

**Gained:**
- ✅ Simplicity
- ✅ Reliability
- ✅ No port confusion
- ✅ No iframe issues
- ✅ No Turbo conflicts

## Files Changed

### 1. `bin/10_queen_rbee/src/http/static_files.rs`

**Before:** 153 lines with dev proxy logic
**After:** 76 lines, always serve embedded files

**Removed:**
- `dev_proxy_handler()` function (71 lines)
- WebSocket blocking logic
- Vite HMR URL rewriting
- HTTP proxying to port 7834
- Debug vs release conditional compilation

**Kept:**
- `static_handler()` function
- SPA routing (fallback to index.html)
- MIME type detection
- Embedded asset serving

### 2. `bin/10_queen_rbee/build.rs`

**Before:** 47 lines with RBEE_SKIP_UI_BUILD logic
**After:** 30 lines, always build UI

**Removed:**
- RBEE_SKIP_UI_BUILD environment variable check
- Debug mode skip logic
- Release mode existing dist check
- cargo:rerun-if-env-changed directive

**Kept:**
- Always build UI via `pnpm exec vite build`
- Verify dist folder exists
- cargo:rerun-if-changed directives

### 3. `bin/99_shared_crates/daemon-lifecycle/src/build.rs`

**Before:** Set RBEE_SKIP_UI_BUILD=1 for queen builds
**After:** No special handling for queen

**Removed:**
- `if daemon_name == "queen-rbee"` check
- `command.env("RBEE_SKIP_UI_BUILD", "1")`
- Narration about skipping UI build

## Code Comparison

### static_files.rs

**Before (153 lines):**
```rust
#[cfg(debug_assertions)]
{
    Router::new().fallback(dev_proxy_handler)
}

#[cfg(not(debug_assertions))]
{
    Router::new().fallback(static_handler)
}

async fn dev_proxy_handler(...) {
    // 71 lines of proxy logic
    // WebSocket blocking
    // HTML rewriting
    // HTTP proxying
}

#[cfg(not(debug_assertions))]
async fn static_handler(...) {
    // Serve embedded files
}
```

**After (76 lines):**
```rust
pub fn create_static_router() -> Router {
    Router::new().fallback(static_handler)
}

async fn static_handler(...) {
    // Serve embedded files
    // Works in both debug and release
}
```

### build.rs

**Before:**
```rust
if std::env::var("RBEE_SKIP_UI_BUILD").is_ok() {
    if cfg!(debug_assertions) {
        println!("Debug mode: UI will be served from Vite dev server");
        return;
    } else if ui_dist.exists() {
        println!("Release mode: Using existing dist folder");
        return;
    } else {
        panic!("Missing dist in release mode!");
    }
}

println!("Building queen-rbee UI (vite only, skipping tsc)...");
```

**After:**
```rust
println!("Building queen-rbee UI for embedding...");
// Always build, no conditions
```

## Documentation Updates Needed

### PORT_CONFIGURATION.md

**Remove:**
- References to port 7834 (Vite dev server)
- Dev proxy explanation
- "Development mode proxies to Vite" notes

**Update:**
- Queen always serves UI at 7833
- UI changes require rebuild
- No separate dev server

### README / Developer Guide

**Add:**
```markdown
## UI Development Workflow

To make changes to the Queen UI:

1. Edit files in `bin/10_queen_rbee/ui/app/src/`
2. Rebuild queen: `cargo build -p queen-rbee`
3. Restart queen: `rbee queen stop && rbee queen start`
4. Access UI: http://localhost:7833

Note: UI changes require a full queen rebuild. There is no hot reload.
```

## Testing

### Verification Steps

1. **Build queen:**
   ```bash
   cargo build -p queen-rbee
   ```
   - Should see: `🔨 Building queen-rbee UI for embedding...`
   - Should complete successfully
   - Should create `ui/app/dist/` folder

2. **Start queen:**
   ```bash
   rbee queen start
   ```
   - Should start successfully
   - Should bind to port 7833

3. **Access UI in browser:**
   ```bash
   open http://localhost:7833
   ```
   - Should load Queen UI
   - Should see no console errors
   - Should see no WebSocket errors

4. **Access UI in Keeper:**
   - Open Keeper GUI
   - Navigate to Queen page
   - Should load iframe successfully
   - Should see no errors

5. **Make UI change:**
   - Edit `bin/10_queen_rbee/ui/app/src/App.tsx`
   - Add a console.log or change text
   - Rebuild: `cargo build -p queen-rbee`
   - Restart: `rbee queen stop && rbee queen start`
   - Refresh browser
   - Should see changes

### Expected Behavior

✅ **Browser (localhost:7833):**
- UI loads correctly
- No WebSocket errors
- No CORS errors
- No "Failed to load" errors

✅ **Keeper iframe:**
- UI loads correctly
- No WebSocket errors
- No CORS errors
- No "Failed to load" errors

✅ **Build process:**
- Always builds UI
- No RBEE_SKIP_UI_BUILD checks
- No conditional logic
- Consistent behavior

## Performance Impact

### Build Times

**Before (with dev proxy):**
- First build: ~30-60 seconds (includes UI build)
- Subsequent builds with RBEE_SKIP_UI_BUILD: ~10-20 seconds
- UI changes: instant (HMR)

**After (no dev proxy):**
- Every build: ~30-60 seconds (always includes UI build)
- UI changes: ~30-60 seconds (full rebuild)

**Trade-off:** Slower iteration but simpler workflow.

### Runtime Performance

**No change:** Serving embedded files is the same in both approaches.

## Migration Guide

### For Developers

**Old workflow:**
```bash
# Terminal 1: Vite dev server
cd bin/10_queen_rbee/ui/app
pnpm dev

# Terminal 2: Queen
cargo run -p queen-rbee

# Access at http://localhost:7834 (dev) or http://localhost:7833 (proxy)
```

**New workflow:**
```bash
# Make UI changes
vim bin/10_queen_rbee/ui/app/src/App.tsx

# Rebuild queen
cargo build -p queen-rbee

# Restart queen
rbee queen stop && rbee queen start

# Access at http://localhost:7833
```

### For CI/CD

**No changes needed.** CI/CD always built UI anyway.

### For Documentation

**Update all references:**
- Remove port 7834 mentions
- Remove dev proxy explanations
- Add rebuild workflow
- Clarify single port (7833)

## Rollback Plan

If this causes issues, revert these commits:

```bash
git revert <commit-hash>  # This commit
```

Then restore:
1. Dev proxy logic in static_files.rs
2. RBEE_SKIP_UI_BUILD logic in build.rs
3. daemon-lifecycle env var setting

## Future Improvements

### Option 1: Vite Build Watch Mode

Add a watch mode that rebuilds UI automatically:

```bash
# Terminal 1: Watch UI changes
cd bin/10_queen_rbee/ui/app
pnpm exec vite build --watch

# Terminal 2: Watch Rust changes and restart queen
cargo watch -x 'build -p queen-rbee' -s 'rbee queen stop && rbee queen start'
```

This would give faster iteration without the complexity of dev proxy.

### Option 2: Separate UI Dev Server (Optional)

For rapid UI development, developers can still run Vite dev server:

```bash
cd bin/10_queen_rbee/ui/app
pnpm dev
# Access at http://localhost:5173 (Vite default)
```

This works for UI-only development but won't connect to queen API.

## Related Issues

- Original investigation: `.docs/debug/2025-10-29_queen-ui_cors_select_sdk_install_investigation.md`
- Keeper iframe fix: `.docs/debug/2025-10-29_KEEPER_IFRAME_FIX.md`
- All fixes: `.docs/debug/2025-10-29_FIXES_IMPLEMENTED.md`

## Summary

**Removed:**
- Dev proxy (71 lines)
- RBEE_SKIP_UI_BUILD logic (20 lines)
- Port 7834 usage
- WebSocket URL rewriting
- Conditional compilation for static serving

**Result:**
- Simpler codebase
- One way to access UI
- No port confusion
- No iframe issues
- Consistent behavior

**Trade-off:**
- Slower UI iteration (rebuild required)
- No HMR

**Decision:** Simplicity > Speed for this project.

---

**Status:** ✅ COMPLETE  
**Compilation:** ✅ PASS  
**Date:** 2025-10-29  
**Approved by:** User (explicitly requested)

# Environment-Aware Build System

**TEAM-341** | **Date:** 2025-10-29  
**Status:** ✅ COMPLETE

## Problem

The `daemon-lifecycle` crate was **always** building binaries with `--release` flag, regardless of whether the parent binary was a debug or release build.

This caused issues in development:
- Dev builds of queen-rbee would build release versions of child daemons
- Release daemons don't proxy to Vite dev servers
- Developers couldn't use hot-reload during development

## Solution

Made `build.rs` and `install.rs` **environment-aware** using Rust's `cfg!(debug_assertions)`:

### Build Mode Detection

```rust
#[cfg(debug_assertions)]
{
    n!("build_mode", "🔧 Building in DEBUG mode (dev environment)");
    // No --release flag in debug mode
}

#[cfg(not(debug_assertions))]
{
    n!("build_mode", "🚀 Building in RELEASE mode (production)");
    command.arg("--release");
}
```

### Binary Path Resolution

```rust
#[cfg(debug_assertions)]
let build_mode = "debug";
#[cfg(not(debug_assertions))]
let build_mode = "release";

let binary_path = if let Some(target_triple) = target {
    PathBuf::from(format!("target/{}/{}/{}", target_triple, build_mode, daemon_name))
} else {
    PathBuf::from(format!("target/{}/{}", build_mode, daemon_name))
};
```

## Behavior

| Parent Binary | Child Daemon Build | Binary Path | Vite Proxy |
|---------------|-------------------|-------------|------------|
| `cargo build` (debug) | Debug build | `target/debug/rbee-hive` | ✅ Works |
| `cargo build --release` | Release build | `target/release/rbee-hive` | N/A (embedded files) |

## Pattern Match

This matches the pattern used in `queen-rbee/src/http/static_files.rs`:

```rust
#[cfg(debug_assertions)]
{
    // Development mode: Proxy to Vite dev server
    Router::new().nest("/ui", Router::new().fallback(dev_proxy_handler))
}

#[cfg(not(debug_assertions))]
{
    // Production mode: Serve embedded static files
    Router::new().nest("/ui", Router::new().fallback(static_handler))
}
```

## Files Modified

1. **`src/build.rs`** (TEAM-341)
   - Added `cfg!(debug_assertions)` check for build mode
   - Conditionally adds `--release` flag
   - Resolves binary path based on build mode
   - Added narration for build mode visibility

2. **`src/install.rs`** (TEAM-341)
   - Updated documentation to reflect environment-aware builds
   - Added notes about debug vs release paths
   - No code changes (uses `build_daemon()` which is now environment-aware)

## Testing

```bash
# Debug build (no --release flag)
cargo build -p queen-rbee
# → Builds child daemons in target/debug/

# Release build (with --release flag)
cargo build --release -p queen-rbee
# → Builds child daemons in target/release/
```

## Benefits

✅ **Development workflow:** Debug builds proxy to Vite dev servers  
✅ **Production builds:** Release builds serve embedded static files  
✅ **Consistency:** Same pattern across all daemons  
✅ **Hot-reload:** Works in development mode  
✅ **Performance:** Production builds are optimized  

## Related

- **PORT_CONFIGURATION.md** - Port mapping for dev vs prod
- **queen-rbee/src/http/static_files.rs** - Static file serving pattern
- **daemon-lifecycle/src/build.rs** - Build implementation
- **daemon-lifecycle/src/install.rs** - Remote installation

---

**Compilation:** ✅ PASS  
**Pattern:** Matches queen-rbee static file serving  
**Impact:** Enables hot-reload in development

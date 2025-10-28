# TEAM-328: `#[with_job_id]` Attribute Macro - Complete

**Status:** ✅ COMPLETE  
**Date:** Oct 27, 2025

## Summary

Created a procedural macro attribute `#[with_job_id]` that eliminates 15-17 lines of repetitive boilerplate from async functions that need job_id context for SSE routing.

## Problem Solved

**Before:** Every function needed this boilerplate:
```rust
pub async fn some_function(config: SomeConfig) -> Result<()> {
    // 15-17 lines of boilerplate
    let ctx = config.job_id.as_ref().map(|jid| NarrationContext::new().with_job_id(jid));
    let impl_fn = async {
        // Actual implementation
        n!("action", "Doing thing");
        Ok(())
    };
    if let Some(ctx) = ctx {
        with_narration_context(ctx, impl_fn).await
    } else {
        impl_fn.await
    }
}
```

**After:** Just add one attribute:
```rust
#[with_job_id]  // Auto-detects "config" parameter
pub async fn some_function(config: SomeConfig) -> Result<()> {
    n!("action", "Doing thing");
    Ok(())
}
```

## Usage

### Auto-detection (recommended)
```rust
use observability_narration_macros::with_job_id;

#[with_job_id]  // Finds first param with "config" in name
pub async fn build_daemon(config: RebuildConfig) -> Result<String> {
    n!("start", "Building...");
    Ok(binary_path)
}
```

### Explicit parameter
```rust
#[with_job_id(config_param = "rebuild_config")]
pub async fn rebuild(rebuild_config: RebuildConfig, other: Other) -> Result<()> {
    n!("start", "Rebuilding...");
    Ok(())
}
```

## Code Reduction

**Example from `rebuild.rs`:**
- Before: 126 lines (32 lines boilerplate = 25%)
- After: 96 lines (2 lines macro)
- **Savings: 30 lines (24% reduction)**

**Potential impact across monorepo:**
- daemon-lifecycle: ~100-150 lines
- queen-rbee-hive-lifecycle: ~200-300 lines
- auto-update: ~50-100 lines
- **Total: 300-500 lines saved**

## Files Created/Modified

### New Files
- `bin/99_shared_crates/narration-macros/src/with_job_id.rs` (117 LOC)
- `bin/99_shared_crates/narration-macros/TEAM_328_WITH_JOB_ID_MACRO.md` (full docs)

### Modified Files
- `bin/99_shared_crates/narration-macros/src/lib.rs` (+43 lines - macro export)
- `bin/99_shared_crates/daemon-lifecycle/Cargo.toml` (+1 dependency)
- `bin/99_shared_crates/daemon-lifecycle/src/rebuild.rs` (-30 lines - migrated 2 functions)

## Benefits

1. ✅ **Less boilerplate** - 15 lines → 1 line per function
2. ✅ **Harder to get wrong** - Macro handles it correctly every time
3. ✅ **Clearer intent** - Business logic not buried in boilerplate
4. ✅ **Easier to maintain** - Change once in macro, applies everywhere
5. ✅ **Type-safe** - Compiler verifies everything
6. ✅ **Zero runtime overhead** - Pure compile-time transformation

## Testing

```bash
# Verify compilation
cargo check -p daemon-lifecycle
✅ PASS

# Verify tests still work
cargo test -p daemon-lifecycle --lib rebuild
✅ PASS (2 tests)
```

## Migration Candidates

**daemon-lifecycle (5 more functions):**
- `src/shutdown.rs` - 2 functions
- `src/health.rs` - 1 function
- `src/list.rs` - 1 function
- `src/uninstall.rs` - 1 function

**Other crates:**
- `queen-rbee-hive-lifecycle` - 9 operations
- `auto-update` - 3+ functions

## Migration Steps

1. Add import: `use observability_narration_macros::with_job_id;`
2. Add attribute: `#[with_job_id]` above function
3. Remove boilerplate (ctx creation, async wrapper, if/else)
4. Remove unused imports: `with_narration_context`, `NarrationContext`
5. Verify: `cargo check`

## Example Migration

**Before (20 lines):**
```rust
use observability_narration_core::{n, with_narration_context, NarrationContext};

pub async fn install(config: InstallConfig) -> Result<()> {
    let ctx = config.job_id.as_ref().map(|jid| NarrationContext::new().with_job_id(jid));
    let impl_fn = async {
        n!("install_start", "Installing...");
        // ... implementation
        n!("install_complete", "✅ Done");
        Ok(())
    };
    if let Some(ctx) = ctx {
        with_narration_context(ctx, impl_fn).await
    } else {
        impl_fn.await
    }
}
```

**After (8 lines):**
```rust
use observability_narration_core::n;
use observability_narration_macros::with_job_id;

#[with_job_id]
pub async fn install(config: InstallConfig) -> Result<()> {
    n!("install_start", "Installing...");
    // ... implementation
    n!("install_complete", "✅ Done");
    Ok(())
}
```

**Result: 60% reduction**

## Documentation

Full documentation in:
- `bin/99_shared_crates/narration-macros/TEAM_328_WITH_JOB_ID_MACRO.md`
- Inline docs in `bin/99_shared_crates/narration-macros/src/lib.rs`
- Inline docs in `bin/99_shared_crates/narration-macros/src/with_job_id.rs`

## Next Steps

1. **Migrate remaining functions** in daemon-lifecycle (5 functions)
2. **Migrate queen-rbee-hive-lifecycle** (9 operations)
3. **Migrate auto-update** (3+ functions)
4. **Consider similar macros:**
   - `#[with_timeout]` - Auto-wrap with TimeoutEnforcer
   - `#[with_retry]` - Auto-retry on failure
   - `#[with_metrics]` - Auto-emit duration metrics

---

**TEAM-328 Signature:** All code tagged with TEAM-328 comments

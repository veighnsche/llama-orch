# TEAM-311: Queen-Lifecycle Migration to n!() Macro

**Status:** 🔄 IN PROGRESS  
**Date:** October 26, 2025  
**Goal:** Migrate all queen-lifecycle files from deprecated `NarrationFactory` to `n!()` macro

---

## Migration Progress

### ✅ Completed Files

1. **status.rs** - ✅ MIGRATED
   - Removed `NarrationFactory`
   - All 3 narrations converted to `n!()`
   - Simple format: `n!("queen_status", "message", args...)`

2. **start.rs** - ✅ MIGRATED
   - Removed `NarrationFactory`
   - 1 narration converted to `n!()`

3. **info.rs** - ✅ MIGRATED  
   - Removed `NarrationFactory`
   - 3 narrations converted to `n!()`

4. **install.rs** - ✅ MIGRATED
   - Removed `NarrationFactory`
   - 8 narrations converted to `n!()`
   - Removed `.error_kind()` calls (not supported in n!() yet)

### 🔄 Remaining Files

5. **uninstall.rs** - TODO
   - Uses `NarrationFactory`
   - Estimated 3-4 narrations

6. **rebuild.rs** - TODO
   - Uses `NarrationFactory`
   - Estimated 5-6 narrations

7. **stop.rs** - TODO
   - Uses `NarrationFactory`
   - Estimated 2-3 narrations

8. **types.rs** - TODO
   - Uses `NarrationFactory`
   - Check if any narrations

9. **health.rs** - TODO
   - Uses `NarrationFactory`
   - Check narrations

10. **ensure.rs** - TODO
    - Uses `NarrationFactory`
    - Likely has many narrations with `.error_kind()`
    - Most complex file

---

## Migration Pattern

### Before (Deprecated)
```rust
use observability_narration_core::NarrationFactory;

const NARRATE: NarrationFactory = NarrationFactory::new("queen-life");

// Simple narration
NARRATE.action("queen_install").human("📦 Installing...").emit();

// With context
NARRATE
    .action("queen_install")
    .context(path.display().to_string())
    .human("✅ Installed at: {}")
    .emit();

// With error_kind
NARRATE
    .action("queen_install")
    .human("❌ Failed")
    .error_kind("build_failed")
    .emit();
```

### After (Current)
```rust
use observability_narration_core::n;

// Simple narration
n!("queen_install", "📦 Installing...");

// With arguments
n!("queen_install", "✅ Installed at: {}", path.display());

// Error (no error_kind support yet)
n!("queen_install", "❌ Failed");
```

---

## Key Changes

1. **Import change**: `NarrationFactory` → `n`
2. **Remove const**: No more `const NARRATE` declarations
3. **Simple syntax**: `n!("action", "message", args...)`
4. **Auto-detected actor**: Crate name automatically used as actor
5. **`.error_kind()` removed**: Not supported in n!() macro (future enhancement)

---

## Benefits

1. **Less boilerplate**: No const declarations
2. **Cleaner code**: Simpler macro syntax
3. **Auto actor**: No need to specify "queen-life" every time
4. **Consistent**: Same API across entire codebase
5. **Standard format**: Uses Rust's `format!()` syntax

---

## Next Steps

1. Finish remaining 6 files:
   - uninstall.rs
   - rebuild.rs  
   - stop.rs
   - types.rs
   - health.rs
   - ensure.rs

2. Test compilation: `cargo check -p queen-lifecycle`

3. Test functionality: Run queen commands to verify narration still works

---

## Commands to Verify

After migration:

```bash
# Check compilation
cargo check -p queen-lifecycle

# Test queen operations
./rbee queen status
./rbee queen info  
./rbee queen start
./rbee queen stop
```

---

## Team Signature

**TEAM-311:** Queen-lifecycle migration to n!() macro

**Status:** 4/10 files complete, continuing with remaining files...

# TEAM-311: NARRATE → n!() Macro Migration

**Status:** 🚧 IN PROGRESS  
**Date:** Oct 26, 2025  
**Mission:** Replace deprecated `NARRATE` pattern with modern `n!()` macro across entire codebase

---

## Summary

Migrating from the deprecated `NarrationFactory` pattern to the new `n!()` macro introduced in narration-core v0.7.0.

### Migration Pattern

**OLD (deprecated):**
```rust
const NARRATE: NarrationFactory = NarrationFactory::new("actor");

let mut narration = NARRATE.action("action").context(&value);
if let Some(ref job_id) = job_id {
    narration = narration.job_id(job_id);
}
narration.human("Message: {}").emit();
```

**NEW (n!() macro):**
```rust
use observability_narration_core::{n, with_narration_context, NarrationContext};

// For functions with job_id parameter:
let ctx = job_id.as_ref().map(|jid| NarrationContext::new().with_job_id(jid));

let impl_fn = async {
    n!("action", "Message: {}", value);
    // ... rest of logic
};

if let Some(ctx) = ctx {
    with_narration_context(ctx, impl_fn).await
} else {
    impl_fn.await
}
```

---

## Progress

### ✅ Completed: daemon-lifecycle crate (18 usages)

**Files migrated:**
- `src/install.rs` - Install/uninstall daemon operations
- `src/ensure.rs` - Ensure daemon running pattern
- `src/shutdown.rs` - Graceful shutdown utilities
- `src/manager.rs` - Daemon process manager
- `src/health.rs` - Health checking with polling
- `src/status.rs` - Status check operations
- `src/get.rs` - Get daemon by ID
- `src/list.rs` - List all daemons

**Verification:**
```bash
cargo check -p daemon-lifecycle  # ✅ PASS
```

---

## Remaining Work

### 🚧 In Progress

**Total remaining:** 303 usages across 43 files

### High Priority (Core Infrastructure)

1. **queen-rbee/src/job_router.rs** (16 usages)
   - Job routing and operation dispatch
   - SSE channel management
   - Inference scheduling

2. **rbee-hive/src/job_router.rs** (28 usages)
   - Worker process management
   - Model catalog operations
   - Hive-level job routing

3. **scheduler/src/simple.rs** (14 usages)
   - Job scheduling logic
   - Worker selection
   - Inference execution

### Medium Priority (Lifecycle Crates)

4. **hive-lifecycle/** (40+ usages)
   - `src/start.rs` (13 usages)
   - `src/stop.rs` (11 usages)
   - `src/ssh.rs` (9 usages)
   - `src/install.rs` (6 usages)
   - `src/uninstall.rs` (4 usages)

5. **queen-lifecycle/** (42+ usages)
   - `src/install.rs` (10 usages)
   - `src/rebuild.rs` (9 usages)
   - `src/ensure.rs` (7 usages)
   - `src/health.rs` (5 usages)
   - `src/stop.rs` (5 usages)
   - `src/info.rs` (4 usages)
   - `src/status.rs` (4 usages)
   - `src/uninstall.rs` (4 usages)
   - `src/start.rs` (2 usages)
   - `src/types.rs` (2 usages)

6. **worker-lifecycle/** (27+ usages)
   - `src/start.rs` (9 usages)
   - `src/stop.rs` (9 usages)
   - `src/get.rs` (5 usages)
   - `src/install.rs` (4 usages)
   - `src/list.rs` (4 usages)
   - `src/uninstall.rs` (4 usages)

### Lower Priority

7. **timeout-enforcer/src/lib.rs** (5 usages)
8. **job-server/src/lib.rs** (9 usages)
9. **rbee-keeper/src/job_client.rs** (5 usages)
10. **rbee-keeper/src/handlers/hive.rs** (3 usages)
11. **ssh-config/src/lib.rs** (1 usage)
12. **narration.rs files** (3 usages total)

---

## Migration Guidelines

### For Functions with `job_id` Parameter

```rust
// 1. Remove NARRATE constant
// OLD:
const NARRATE: NarrationFactory = NarrationFactory::new("actor");

// 2. Update imports
use observability_narration_core::{n, with_narration_context, NarrationContext};

// 3. Wrap async function body
pub async fn my_function(config: Config) -> Result<()> {
    let ctx = config.job_id.as_ref().map(|jid| NarrationContext::new().with_job_id(jid));
    
    let impl_fn = async {
        n!("action", "Message: {}", value);
        // ... rest of function logic
        Ok(())
    };
    
    if let Some(ctx) = ctx {
        with_narration_context(ctx, impl_fn).await
    } else {
        impl_fn.await
    }
}
```

### For Simple Narration (No job_id)

```rust
// Just replace directly
// OLD:
NARRATE.action("action").context(&value).human("Message: {}").emit();

// NEW:
n!("action", "Message: {}", value);
```

### For Multiple Variables

```rust
// OLD:
NARRATE
    .action("action")
    .context(&var1)
    .context(&var2)
    .human("Message: {} and {}")
    .emit();

// NEW:
n!("action", "Message: {} and {}", var1, var2);
```

### For Error Narration

```rust
// OLD:
NARRATE
    .action("error")
    .context(&error)
    .human("Failed: {}")
    .error_kind("error_type")
    .emit_error();

// NEW:
n!("error", "Failed: {}", error);
// Note: error_kind is not supported in n!() macro
```

---

## Testing Strategy

After each file migration:

1. **Compile check:**
   ```bash
   cargo check -p <package-name>
   ```

2. **Run tests:**
   ```bash
   cargo test -p <package-name>
   ```

3. **Integration tests:**
   ```bash
   cargo xtask e2e:queen
   cargo xtask e2e:hive
   ```

---

## Benefits of n!() Macro

1. **Ultra-concise:** 1 line instead of 5+ lines
2. **Auto-detected actor:** Uses `env!("CARGO_CRATE_NAME")`
3. **Standard Rust format!():** No custom `{0}`, `{1}` syntax
4. **Automatic context:** job_id propagates via thread-local storage
5. **Type-safe:** Compile-time validation of format strings

---

## Breaking Changes

### Removed Features

1. **`.error_kind()`** - Not supported in n!() macro
2. **`.table()`** - Use formatted JSON string instead
3. **`.context()` with {0}, {1}** - Use standard Rust format!() syntax

### Migration Examples

**Table display:**
```rust
// OLD:
NARRATE.action("list").table(&json_value).emit();

// NEW:
let table_str = serde_json::to_string_pretty(&json_value).unwrap_or_default();
n!("list", "Results:\n{}", table_str);
```

**Error kinds:**
```rust
// OLD:
NARRATE.action("error").error_kind("not_found").emit_error();

// NEW:
n!("error", "Resource not found");
// Error kind is implicit in the action name
```

---

## Compilation Status

### ✅ Passing
- `daemon-lifecycle` - All 8 files migrated

### 🚧 Pending
- All other crates (43 files remaining)

---

## Next Steps

1. Migrate `queen-rbee/src/job_router.rs` (16 usages)
2. Migrate `rbee-hive/src/job_router.rs` (28 usages)
3. Migrate `scheduler/src/simple.rs` (14 usages)
4. Migrate lifecycle crates (hive, queen, worker)
5. Migrate remaining utility crates
6. Run full test suite
7. Update documentation

---

## Estimated Effort

- **daemon-lifecycle:** ✅ 2 hours (COMPLETE)
- **Core routers:** 🚧 3 hours (queen + hive + scheduler)
- **Lifecycle crates:** 🚧 4 hours (hive + queen + worker)
- **Remaining crates:** 🚧 2 hours
- **Testing & verification:** 🚧 1 hour

**Total:** ~12 hours (2 hours complete, 10 hours remaining)

---

## Team Signature

**TEAM-311:** Narration modernization - Replace deprecated NARRATE with n!() macro

All code changes include `// TEAM-311: Migrated to n!() macro` comments.

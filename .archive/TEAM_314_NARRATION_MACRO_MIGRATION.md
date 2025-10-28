# TEAM-314: Narration Macro Migration for `hive check` Flow

**Status:** ✅ COMPLETE  
**Date:** 2025-10-27  
**Purpose:** Migrate all files in the `./rbee hive check -a workstation` code flow from deprecated `NarrationFactory` pattern to the new `n!()` macro

---

## Command Flow Analysis

The `./rbee hive check -a workstation` command flows through:

```
rbee-keeper CLI
    ↓
bin/00_rbee_keeper/src/cli/hive.rs (HiveAction::Check)
    ↓
bin/00_rbee_keeper/src/handlers/hive.rs (handle_hive)
    ↓
bin/00_rbee_keeper/src/job_client.rs (submit_and_stream_job)
    ↓
[HTTP POST to hive] http://localhost:7835/v1/jobs
    ↓
bin/20_rbee_hive/src/job_router.rs (route_job)
    ↓
bin/20_rbee_hive/src/hive_check.rs (handle_hive_check)
```

---

## Files Modified

### 1. bin/00_rbee_keeper/src/handlers/hive.rs

**Changes:**
- Removed deprecated `NarrationFactory` import
- Added `n` macro import
- Removed `const NARRATE: NarrationFactory` declaration
- Migrated `HiveAction::List` narration calls to `n!()` macro

**Before:**
```rust
use observability_narration_core::NarrationFactory;

const NARRATE: NarrationFactory = NarrationFactory::new("keeper");

// In HiveAction::List
NARRATE
    .action("hive_list")
    .human("⚠️  No SSH hosts found in ~/.ssh/config")
    .emit();

NARRATE
    .action("hive_list")
    .human(format!("Found {} SSH target(s)", targets.len()))
    .table(&json_value)
    .emit();
```

**After:**
```rust
use observability_narration_core::n;

// In HiveAction::List
n!("hive_list", "⚠️  No SSH hosts found in ~/.ssh/config");

n!("hive_list", "Found {} SSH target(s)", targets.len());
// Print table as JSON for now (n!() doesn't support .table() yet)
println!("{}", serde_json::to_string_pretty(&json_value)?);
```

**Note:** The `.table()` method is not yet supported by `n!()` macro, so we use `println!()` with pretty-printed JSON as a workaround.

### 2. bin/20_rbee_hive/src/job_router.rs

**Changes:**
- Added `n` macro import alongside existing `NarrationFactory`
- Kept `NARRATE` constant for worker operations (not yet migrated)
- Migrated `HiveCheck` operation handler to `n!()` macro

**Before:**
```rust
use observability_narration_core::NarrationFactory;

const NARRATE: NarrationFactory = NarrationFactory::new("hv-router");

// In Operation::HiveCheck
NARRATE
    .action("hive_check_start")
    .job_id(&job_id)
    .human("🔍 Starting hive narration check")
    .emit();

// ... handler code ...

NARRATE
    .action("hive_check_complete")
    .job_id(&job_id)
    .human("✅ Hive narration check complete")
    .emit();
```

**After:**
```rust
use observability_narration_core::{n, NarrationFactory};

// TEAM-314: Keeping for worker operations that haven't been migrated yet
const NARRATE: NarrationFactory = NarrationFactory::new("hv-router");

// In Operation::HiveCheck
n!("hive_check_start", "🔍 Starting hive narration check");

// ... handler code ...

n!("hive_check_complete", "✅ Hive narration check complete");
```

**Note:** The `job_id` is automatically propagated through the `NarrationContext` set by `with_narration_context()`, so we don't need to manually specify it with `n!()`.

---

## Files Already Using n!() Macro

These files were already correctly using the new `n!()` macro:

### 1. bin/00_rbee_keeper/src/job_client.rs ✅
- Lines 48, 75, 97, 99: All narration uses `n!()` macro
- No changes needed

### 2. bin/20_rbee_hive/src/hive_check.rs ✅
- Lines 26-104: All narration uses `n!()` macro
- No changes needed

---

## Migration Pattern

### Old Pattern (Deprecated)
```rust
use observability_narration_core::NarrationFactory;

const NARRATE: NarrationFactory = NarrationFactory::new("actor");

NARRATE
    .action("action_name")
    .job_id(&job_id)  // Optional, for SSE routing
    .human("Message with {}")
    .context(&value)
    .emit();
```

### New Pattern (Recommended)
```rust
use observability_narration_core::n;

// Simple narration
n!("action_name", "Message with {}", value);

// With narration context (for SSE routing)
use observability_narration_core::{with_narration_context, NarrationContext};

let ctx = NarrationContext::new().with_job_id(&job_id);
with_narration_context(ctx, async {
    n!("action_name", "Message with {}", value);
    // ... more code ...
}).await?;
```

---

## Benefits of n!() Macro

1. **Concise:** Single line instead of builder pattern
2. **Type-safe:** Compile-time format string checking
3. **Auto-detection:** Actor name auto-detected from crate name
4. **Context-aware:** Automatically uses narration context when set
5. **Modern:** Aligns with Rust macro conventions

---

## Limitations

### .table() Method Not Supported

The old `NarrationFactory` had a `.table()` method for structured data output:

```rust
// Old pattern
NARRATE
    .action("action")
    .table(&json_value)
    .emit();
```

The `n!()` macro doesn't support this yet. **Workaround:**

```rust
// New pattern
n!("action", "Message");
println!("{}", serde_json::to_string_pretty(&json_value)?);
```

---

## Verification

✅ **Compilation:** `cargo build --bin rbee-keeper --bin rbee-hive` - SUCCESS  
✅ **Code Flow:** All files in `hive check` flow updated  
✅ **Backwards Compatibility:** Old `NARRATE` constant kept for worker operations

---

## Future Work

### Remaining Migrations

The following files still use the old `NarrationFactory` pattern and should be migrated in future work:

**bin/20_rbee_hive/src/job_router.rs:**
- Worker operations (WorkerSpawn, WorkerList, WorkerGet, WorkerDelete)
- Model operations (ModelDownload, ModelList, ModelGet, ModelDelete)

**Other binaries:**
- bin/10_queen_rbee/src/job_router.rs (if any remain)
- bin/30_llm_worker_rbee/src/* (if any)

---

## Testing

To test the migration:

```bash
# Build binaries
cargo build --bin rbee-keeper --bin rbee-hive

# Start hive on workstation (if not already running)
./target/debug/rbee-keeper hive start -a workstation

# Run hive check
./target/debug/rbee-keeper hive check -a localhost
```

Expected output should show narration messages using the new format, with proper SSE streaming.

---

## Related Work

- **TEAM-311:** Initial n!() macro migration in queen-lifecycle
- **TEAM-313:** HiveCheck command implementation
- **TEAM-314:** Port configuration update + narration migration

---

**Maintained by:** TEAM-314  
**Last Updated:** 2025-10-27

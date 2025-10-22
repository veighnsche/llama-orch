# TEAM-185 Handoff: rbee-keeper Consolidation & Narration Improvements

**Date:** 2025-10-21  
**Team:** TEAM-185  
**Scope:** rbee-keeper binary consolidation, narration enhancements, and CLI improvements

---

## Summary

TEAM-185 consolidated the rbee-keeper architecture by:
1. **Merged queen-lifecycle crate** into rbee-keeper binary as a module
2. **Enhanced narration-core** with `operation` and `hive_id` fields for better observability
3. **Renamed actions → operations** for clearer distinction between job operations and lifecycle actions
4. **Improved CLI** with comprehensive inference parameters and clearer naming
5. **Removed 05_rbee_keeper_crates** directory from workspace (consolidated into binary)

---

## Changes Made

### 1. Queen Lifecycle Consolidation

**Before:**
```
bin/05_rbee_keeper_crates/queen-lifecycle/  (separate crate)
bin/00_rbee_keeper/Cargo.toml               (depends on queen-lifecycle)
```

**After:**
```
bin/00_rbee_keeper/src/queen_lifecycle.rs   (integrated module)
bin/00_rbee_keeper/Cargo.toml               (direct dependencies: daemon-lifecycle, timeout-enforcer)
```

**Rationale:** The queen-lifecycle crate was only used by rbee-keeper. Consolidating it reduces indirection and simplifies the codebase.

**Files Modified:**
- `bin/00_rbee_keeper/src/queen_lifecycle.rs` - Moved from separate crate
- `bin/00_rbee_keeper/src/main.rs` - Added `mod queen_lifecycle;`
- `bin/00_rbee_keeper/src/job_client.rs` - Updated imports
- `bin/00_rbee_keeper/Cargo.toml` - Removed queen-lifecycle dependency, added direct deps
- `Cargo.toml` (workspace) - Removed 05_rbee_keeper_crates from members

---

### 2. Actions → Operations Rename

**Before:**
```rust
// bin/00_rbee_keeper/src/actions.rs
pub const ACTION_WORKER_SPAWN: &str = "worker_spawn";
pub const ACTION_INFER: &str = "infer";
```

**After:**
```rust
// bin/00_rbee_keeper/src/operations.rs
pub const OP_WORKER_SPAWN: &str = "worker_spawn";
pub const OP_INFER: &str = "infer";
pub const ACTION_JOB_SUBMIT: &str = "job_submit";
pub const ACTION_JOB_STREAM: &str = "job_stream";
pub const ACTION_JOB_COMPLETE: &str = "job_complete";
```

**Rationale:** 
- `OP_*` constants represent **job operations** (what the job does: worker_spawn, infer, etc.)
- `ACTION_*` constants represent **narration lifecycle actions** (job_submit, job_stream, job_complete)
- This distinction makes it clear when we're talking about the operation vs. the lifecycle event

**Files Modified:**
- `bin/00_rbee_keeper/src/operations.rs` - Renamed from actions.rs, reorganized constants
- `bin/00_rbee_keeper/src/main.rs` - Updated all operation strings to use OP_* constants
- `bin/00_rbee_keeper/src/job_client.rs` - Uses ACTION_* constants for narration

---

### 3. Narration-Core Enhancements

**New Fields Added:**

```rust
// bin/99_shared_crates/narration-core/src/lib.rs
pub struct NarrationFields {
    // ... existing fields ...
    
    /// TEAM-185: Added hive_id for multi-hive rbee operations
    pub hive_id: Option<String>,
    
    /// TEAM-185: Added operation field for job-based systems to track dynamic operation names
    /// Unlike action (which is static), this can be dynamic and operation-specific
    pub operation: Option<String>,
}
```

**New Builder Methods:**

```rust
// bin/99_shared_crates/narration-core/src/builder.rs
impl Narration {
    /// TEAM-185: Added for multi-hive rbee operations
    pub fn hive_id(mut self, id: impl Into<String>) -> Self { ... }
    
    /// TEAM-185: Added for job-based systems to track dynamic operation names
    pub fn operation(mut self, op: impl Into<String>) -> Self { ... }
}
```

**Usage Example:**

```rust
// Before (operation name embedded in human message)
Narration::new(ACTOR_RBEE_KEEPER, "job_submit", job_id)
    .human(format!("📋 Job {} submitted: {}", job_id, operation))
    .emit();

// After (operation as structured field)
Narration::new(ACTOR_RBEE_KEEPER, ACTION_JOB_SUBMIT, job_id)
    .operation(operation)
    .hive_id(hive_id)
    .human(format!("📋 Job {} submitted", job_id))
    .emit();
```

**Benefits:**
- **Queryable fields** - Can filter logs by operation type or hive
- **Cleaner human messages** - No need to repeat operation name
- **Better observability** - Structured data instead of text parsing

**Files Modified:**
- `bin/99_shared_crates/narration-core/src/lib.rs` - Added fields, updated emit macro
- `bin/99_shared_crates/narration-core/src/builder.rs` - Added builder methods
- `bin/00_rbee_keeper/src/job_client.rs` - Updated to use new fields

---

### 4. CLI Improvements

#### 4.1 Worker Spawn: `backend` → `worker`

**Before:**
```bash
rbee worker --hive-id localhost spawn --model M --backend cuda --device 0
```

**After:**
```bash
rbee worker --hive-id localhost spawn --model M --worker cuda --device 0
```

**Rationale:** The term "worker" is more accurate since we're specifying the worker type (cpu/cuda/metal), not a backend.

#### 4.2 Comprehensive Inference Parameters

**Before:**
```rust
Commands::Infer {
    model: String,
    prompt: String,
    max_tokens: Option<u32>,
    temperature: Option<f32>,
}
```

**After:**
```rust
Commands::Infer {
    hive_id: String,
    model: String,
    prompt: String,
    max_tokens: u32,
    temperature: f32,
    top_p: Option<f32>,      // NEW
    top_k: Option<u32>,      // NEW
    device: Option<String>,  // NEW (cpu/cuda/metal filter)
    worker_id: Option<String>, // NEW (specific worker)
    stream: bool,            // NEW
}
```

**Example:**
```bash
rbee infer --hive-id localhost --model M "prompt" \
  --max-tokens 20 --temperature 0.7 \
  --top-p 0.9 --top-k 50 \
  --device cuda \
  --worker-id worker-123 \
  --stream true
```

**Files Modified:**
- `bin/00_rbee_keeper/src/main.rs` - Updated Infer command struct and handler
- `bin/API_REFERENCE.md` - Updated CLI reference

---

### 5. Job Client Narration Updates

**Key Changes:**

1. **Operation extraction from payload:**
```rust
let operation = job_payload["operation"].as_str().unwrap_or("unknown");
let hive_id = job_payload["hive_id"].as_str();
```

2. **Structured narration:**
```rust
Narration::new(ACTOR_RBEE_KEEPER, ACTION_JOB_SUBMIT, job_id)
    .operation(operation)
    .hive_id(hive_id)
    .human(format!("📋 Job {} submitted", job_id))
    .emit();
```

3. **Three lifecycle events:**
- `ACTION_JOB_SUBMIT` - When job is submitted
- `ACTION_JOB_STREAM` - When streaming starts
- `ACTION_JOB_COMPLETE` - When operation completes

**Files Modified:**
- `bin/00_rbee_keeper/src/job_client.rs`

---

## Architecture Decisions

### Why Consolidate queen-lifecycle?

**Problem:** The queen-lifecycle crate was a single-purpose crate only used by rbee-keeper, adding unnecessary indirection.

**Solution:** Move it into rbee-keeper as a module. This:
- Reduces cognitive overhead (one less crate to navigate)
- Simplifies dependency graph
- Maintains clear module boundaries within the binary

### Why Separate OP_* and ACTION_* Constants?

**Problem:** Using "action" for both job operations and narration lifecycle events was confusing.

**Solution:**
- `OP_*` = Job operations (worker_spawn, infer, model_download)
- `ACTION_*` = Narration lifecycle (job_submit, job_stream, job_complete)

This makes it immediately clear what level of abstraction we're working at.

### Why Add `operation` Field to Narration?

**Problem:** The `action` field must be `&'static str`, so we couldn't use dynamic operation names. We were embedding operation names in human messages, making them unparseable.

**Solution:** Add `operation: Option<String>` field that can hold dynamic values. Now:
- `action` = Static lifecycle action (job_submit)
- `operation` = Dynamic job operation (worker_spawn)
- `target` = Job ID or resource ID
- `human` = Clean, concise message

---

## Testing

### Manual Testing Performed

1. **Queen lifecycle:**
```bash
rbee queen start  # Should auto-start queen if not running
rbee queen stop   # Should stop queen gracefully
```

2. **Worker operations:**
```bash
rbee worker --hive-id localhost spawn --model M --worker cuda --device 0
rbee worker --hive-id localhost list
```

3. **Inference:**
```bash
rbee infer --hive-id localhost --model M "Hello" \
  --max-tokens 20 --temperature 0.7 --stream true
```

4. **Build verification:**
```bash
cargo build --bin rbee-keeper  # Should compile without errors
```

### Expected Narration Output

```json
{
  "actor": "🧑‍🌾 rbee-keeper",
  "action": "job_submit",
  "target": "job-abc123",
  "operation": "worker_spawn",
  "hive_id": "localhost",
  "human": "📋 Job job-abc123 submitted"
}
```

---

## Migration Guide

### For Future Teams

If you need to add a new operation:

1. **Add operation constant:**
```rust
// bin/00_rbee_keeper/src/operations.rs
pub const OP_YOUR_OPERATION: &str = "your_operation";
```

2. **Add CLI command:**
```rust
// bin/00_rbee_keeper/src/main.rs
Commands::YourCommand { ... } => {
    let job_payload = serde_json::json!({
        "operation": OP_YOUR_OPERATION,
        "hive_id": hive_id,
        // ... other params
    });
    submit_and_stream_job(&client, &queen_url, job_payload).await
}
```

3. **Update API reference:**
```markdown
<!-- bin/API_REFERENCE.md -->
rbee your-command --hive-id {id} ...
```

### Breaking Changes

⚠️ **CLI Breaking Changes:**
- `--backend` renamed to `--worker` in worker spawn command
- Inference now requires `--hive-id` parameter

⚠️ **Crate Removal:**
- `rbee-keeper-queen-lifecycle` crate removed from workspace
- Code moved to `bin/00_rbee_keeper/src/queen_lifecycle.rs`

---

## Files Changed Summary

### Created
- `bin/00_rbee_keeper/src/operations.rs` (renamed from actions.rs)
- `bin/00_rbee_keeper/src/queen_lifecycle.rs` (moved from separate crate)
- `bin/00_rbee_keeper/TEAM_185_HANDOFF.md` (this file)

### Modified
- `bin/00_rbee_keeper/src/main.rs` - CLI updates, module imports, operation constants
- `bin/00_rbee_keeper/src/job_client.rs` - Narration updates, operation/hive_id fields
- `bin/00_rbee_keeper/Cargo.toml` - Dependency consolidation
- `bin/99_shared_crates/narration-core/src/lib.rs` - New fields (operation, hive_id)
- `bin/99_shared_crates/narration-core/src/builder.rs` - New builder methods
- `bin/API_REFERENCE.md` - Updated CLI reference
- `Cargo.toml` (workspace) - Removed 05_rbee_keeper_crates

### Deleted
- `bin/05_rbee_keeper_crates/` (entire directory removed from workspace)

---

## Known Issues / Future Work

### None Currently

All changes compile and manual testing passed.

### Potential Improvements

1. **Config consolidation:** The `config` crate in 05_rbee_keeper_crates was also unused and could be removed
2. **Polling crate:** Similar consolidation opportunity
3. **Device parameter clarity:** The `--device` parameter in inference is a device type filter (cpu/cuda/metal), not a device ID. Consider renaming to `--device-type` for clarity.

---

## Questions?

Contact TEAM-185 or refer to:
- `bin/API_REFERENCE.md` - CLI usage examples
- `bin/00_rbee_keeper/src/operations.rs` - Operation constants
- `bin/99_shared_crates/narration-core/` - Narration field documentation

---

**End of TEAM-185 Handoff**

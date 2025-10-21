# 🎀 Narration Core v0.3.0 Quick Reference

**TEAM-191 Upgrade** | **Date**: 2025-10-21

---

## 📊 NEW: Table Formatting

```rust
use observability_narration_core::Narration;

let data = serde_json::json!([
    {"id": "hive-1", "status": "ready", "workers": 3},
    {"id": "hive-2", "status": "starting", "workers": 0}
]);

Narration::new("queen-router", "status", "registry")
    .human("System Status:")
    .table(&data)
    .emit();
```

**Output**:
```
[queen-router]
  System Status:

  id     │ status   │ workers
  ───────┼──────────┼─────────
  hive-1 │ ready    │ 3
  hive-2 │ starting │ 0
```

---

## 👑 NEW: Queen-Rbee Constants

```rust
use observability_narration_core::{
    ACTOR_QUEEN_RBEE,
    ACTOR_QUEEN_ROUTER,
    ACTION_STATUS,
    ACTION_ROUTE_JOB,
    ACTION_HIVE_INSTALL,
    ACTION_HIVE_START,
    Narration,
};

// Main service
Narration::new(ACTOR_QUEEN_RBEE, ACTION_STATUS, "system")
    .human("Queen-rbee ready")
    .emit();

// Job routing
Narration::new(ACTOR_QUEEN_ROUTER, ACTION_ROUTE_JOB, "job-123")
    .human("Routing job to handler")
    .emit();

// Hive operations
Narration::new(ACTOR_QUEEN_ROUTER, ACTION_HIVE_INSTALL, "hive-1")
    .human("🔧 Installing hive 'hive-1'")
    .emit();
```

---

## 🎯 NEW: Action Constants

### Job Routing
- `ACTION_ROUTE_JOB` - Route job to handler
- `ACTION_PARSE_OPERATION` - Parse operation
- `ACTION_JOB_CREATE` - Create job

### Hive Management
- `ACTION_HIVE_INSTALL` - Install hive
- `ACTION_HIVE_UNINSTALL` - Uninstall hive
- `ACTION_HIVE_START` - Start hive
- `ACTION_HIVE_STOP` - Stop hive
- `ACTION_HIVE_STATUS` - Check status
- `ACTION_HIVE_LIST` - List hives

### System
- `ACTION_STATUS` - Get status
- `ACTION_START` - Start service
- `ACTION_LISTEN` - Listen
- `ACTION_READY` - Ready
- `ACTION_ERROR` - Error

---

## 😊 Emoji Support (Confirmed!)

```rust
Narration::new("queen-router", "hive_install", "hive-1")
    .human("🔧 Installing hive 'hive-1'")
    .emit();

Narration::new("queen-router", "status", "registry")
    .human("📊 Fetching live status")
    .emit();

Narration::new("queen-router", "hive_start", "hive-1")
    .human("✅ Hive started successfully")
    .emit();
```

**Recommended Emojis**:
- 📊 - Status/data operations
- ✅ - Success
- ❌ - Errors
- 🔧 - Installation/config
- 🏠 - Localhost operations
- ⚠️ - Warnings
- 🚀 - Starting/launching
- 🛑 - Stopping

---

## 📝 Multi-line Messages

```rust
Narration::new("queen-router", "hive_install_error", "hive-1")
    .human(
        "❌ Hive 'hive-1' not found in catalog.\n\
         \n\
         To install the hive:\n\
         \n\
           ./rbee hive install --id hive-1"
    )
    .emit();
```

**Guidelines**:
- Use `\n\` for line breaks (backslash prevents extra whitespace)
- Use `\n` for blank lines
- Indent commands with 2 spaces
- Keep total message under 500 characters

---

## 🎀 Complete Example

```rust
use observability_narration_core::{
    Narration,
    ACTOR_QUEEN_ROUTER,
    ACTION_STATUS,
};

// Get hive status from registry
let hives = vec![
    serde_json::json!({"id": "hive-1", "workers": 3, "state": "ready"}),
    serde_json::json!({"id": "hive-2", "workers": 0, "state": "starting"}),
];

Narration::new(ACTOR_QUEEN_ROUTER, ACTION_STATUS, "registry")
    .human(format!("📊 Live Status ({} hive(s)):", hives.len()))
    .table(&serde_json::Value::Array(hives))
    .emit();
```

**Output**:
```
[👑 queen-router]
  📊 Live Status (2 hive(s)):

  id     │ workers │ state
  ───────┼─────────┼──────────
  hive-1 │ 3       │ ready
  hive-2 │ 0       │ starting
```

---

## 🚀 Migration Guide

### Before (v0.2.0)
```rust
// Had to use string literals
Narration::new("queen-router", "status", "registry")
    .human("Status check")
    .emit();
```

### After (v0.3.0)
```rust
// Use constants for type safety
use observability_narration_core::{ACTOR_QUEEN_ROUTER, ACTION_STATUS};

Narration::new(ACTOR_QUEEN_ROUTER, ACTION_STATUS, "registry")
    .human("📊 Status check")
    .emit();
```

---

## 📚 Full Taxonomy

### Actors
- `ACTOR_ORCHESTRATORD` - Core orchestration
- `ACTOR_POOL_MANAGERD` - GPU pool manager
- `ACTOR_WORKER_ORCD` - Worker daemon
- `ACTOR_INFERENCE_ENGINE` - Inference engine
- `ACTOR_VRAM_RESIDENCY` - VRAM manager
- `ACTOR_QUEEN_RBEE` - Queen-rbee main ✨ NEW
- `ACTOR_QUEEN_ROUTER` - Queen router ✨ NEW

### Actions (Selected)
- Admission: `ACTION_ADMISSION`, `ACTION_ENQUEUE`, `ACTION_DISPATCH`
- Lifecycle: `ACTION_SPAWN`, `ACTION_READY_CALLBACK`, `ACTION_HEARTBEAT_SEND`, `ACTION_SHUTDOWN`
- Inference: `ACTION_INFERENCE_START`, `ACTION_INFERENCE_COMPLETE`, `ACTION_CANCEL`
- VRAM: `ACTION_VRAM_ALLOCATE`, `ACTION_SEAL`, `ACTION_VERIFY`
- Pool: `ACTION_REGISTER`, `ACTION_PROVISION`
- Jobs: `ACTION_ROUTE_JOB`, `ACTION_JOB_CREATE` ✨ NEW
- Hives: `ACTION_HIVE_INSTALL`, `ACTION_HIVE_START`, `ACTION_HIVE_STOP` ✨ NEW
- System: `ACTION_STATUS`, `ACTION_START`, `ACTION_READY` ✨ NEW

---

## 🎯 Best Practices

### 1. Use Constants
```rust
// ✅ Good
Narration::new(ACTOR_QUEEN_ROUTER, ACTION_STATUS, "registry")

// ❌ Bad
Narration::new("queen-router", "status", "registry")
```

### 2. Use Emojis
```rust
// ✅ Good - Clear visual indicators
.human("📊 Fetching status")
.human("✅ Operation successful")
.human("❌ Operation failed")

// ⚠️ OK but less clear
.human("Fetching status")
.human("Operation successful")
.human("Operation failed")
```

### 3. Use Tables for Structured Data
```rust
// ✅ Good - Readable table
.human("Found 3 hives:")
.table(&hives_json)

// ❌ Bad - Hard to read
.human(format!("Found hives: {:?}", hives))
```

### 4. Use Multi-line for Complex Messages
```rust
// ✅ Good - Clear and helpful
.human(
    "❌ Hive not found.\n\
     \n\
     To install:\n\
     \n\
       ./rbee hive install"
)

// ❌ Bad - Cramped and unclear
.human("Hive not found. Install with: ./rbee hive install")
```

---

## 🔗 Resources

- **Full Documentation**: `README.md`
- **Changelog**: `CHANGELOG.md`
- **Upgrade Plan**: `TEAM-191-UPGRADE-PLAN.md`
- **Summary**: `TEAM-191-SUMMARY.md`
- **Team Responsibilities**: `TEAM_RESPONSIBILITIES.md`

---

*May your logs be readable, your correlation IDs present, and your debugging experience absolutely DELIGHTFUL! 🎀✨*

— TEAM-191 (The Narration Core Team) 💝

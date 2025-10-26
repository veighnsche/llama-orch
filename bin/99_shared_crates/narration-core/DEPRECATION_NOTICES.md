# Deprecation Notices - TEAM-310 Format Centralization

**Date:** Oct 26, 2025  
**Team:** TEAM-310  
**Change:** Centralized all formatting logic into `format.rs` module

## Summary

All formatting functions and constants have been moved from scattered locations into a centralized `format.rs` module. This provides a single source of truth for narration formatting.

## Deprecated Code Locations

### 1. Constants (sse_sink.rs)

**Old Location:** `src/output/sse_sink.rs`

```rust
// ⚠️ DEPRECATED - REMOVED
const ACTOR_WIDTH: usize = 10;
const ACTION_WIDTH: usize = 15;
```

**New Location:** `src/format.rs`

```rust
pub const ACTOR_WIDTH: usize = 20;   // Increased from 10
pub const ACTION_WIDTH: usize = 20;  // Increased from 15
```

**Migration:**
```rust
use observability_narration_core::format::{ACTOR_WIDTH, ACTION_WIDTH};
```

---

### 2. Inline Formatting (sse_sink.rs)

**Old Code:** `src/output/sse_sink.rs` (REMOVED ~10 LOC)

```rust
// ⚠️ DEPRECATED - REMOVED
let formatted = format!(
    "[{:<10}] {:<15}: {}",
    fields.actor,
    fields.action,
    fields.human
);
```

**New Code:** Uses centralized function

```rust
use crate::format::format_message;

let formatted = format_message(fields.actor, fields.action, &fields.human);
// Result: Bold first line + message on newline
```

---

### 3. Table Formatting Functions (builder.rs)

**Old Location:** `src/api/builder.rs` (REMOVED ~60 LOC)

```rust
// ⚠️ DEPRECATED - REMOVED
fn format_array_table(items: &[Value]) -> String { ... }
fn format_object_table(map: &serde_json::Map<String, Value>) -> String { ... }
fn format_value_compact(value: &Value) -> String { ... }
```

**New Location:** `src/format.rs`

```rust
pub fn format_array_table(arr: &[Value]) -> String { ... }
pub fn format_object_table(map: &serde_json::Map<String, Value>) -> String { ... }
pub fn format_value_compact(value: &Value) -> String { ... }
```

**Migration:**
```rust
use observability_narration_core::format::{
    format_array_table,
    format_object_table,
    format_value_compact,
};
```

---

### 4. Job ID Shortening (builder.rs)

**Old Location:** `src/api/builder.rs` (REMOVED ~8 LOC)

```rust
// ⚠️ DEPRECATED - REMOVED
pub fn short_job_id(job_id: &str) -> String {
    if job_id.len() > 6 {
        format!("...{}", &job_id[job_id.len() - 6..])
    } else {
        job_id.to_string()
    }
}
```

**New Location:** `src/format.rs` (also re-exported from `lib.rs`)

```rust
pub fn short_job_id(job_id: &str) -> String { ... }
```

**Migration:**
```rust
// Option 1: Use re-export (recommended)
use observability_narration_core::short_job_id;

// Option 2: Direct import
use observability_narration_core::format::short_job_id;
```

---

### 5. Global stderr Output (emit.rs)

**Old Code:** `src/api/emit.rs` (REMOVED for security - TEAM-299)

```rust
// ⚠️ DEPRECATED - REMOVED (SECURITY FIX)
// This was removed in TEAM-299 for privacy/security reasons
eprintln!("[{:<10}] {:<15}: {}", fields.actor, fields.action, message);
```

**Reason for Removal:**
- Privacy violation in multi-tenant environments
- No job isolation
- Sensitive data exposed globally

**New Architecture:**
- SSE is PRIMARY output (job-scoped, secure)
- CLI uses tracing subscriber with `format_message()`
- No global stderr in narration-core

---

### 6. Context Interpolation (builder.rs)

**Old Code:** Inline in `human()`, `cute()`, `story()` methods

```rust
// ⚠️ DEPRECATED - Replaced with centralized function
let mut msg = msg.into();
for (i, value) in self.context_values.iter().enumerate() {
    msg = msg.replace(&format!("{{{}}}", i), value);
}
if let Some(first) = self.context_values.first() {
    msg = msg.replace("{}", first);
}
```

**New Code:** Uses centralized function

```rust
use crate::format::interpolate_context;

let msg = interpolate_context(&msg.into(), &self.context_values);
```

---

## Format Changes

### Old Format (pre-TEAM-310)
```
[actor     ] action         : message
```
- Actor: 10 chars
- Action: 15 chars
- Message: inline after colon
- No ANSI formatting

### New Format (TEAM-310)
```
[actor              ] action              
message
```
- Actor: 20 chars (bold)
- Action: 20 chars (bold)
- Message: on new line (no formatting)
- ANSI bold codes on first line

**ANSI Codes:**
- `\x1b[1m` - Start bold
- `\x1b[0m` - Reset formatting

---

## Code Removed

| File | LOC Removed | Description |
|------|-------------|-------------|
| `builder.rs` | ~80 | Table formatting + short_job_id |
| `sse_sink.rs` | ~10 | Inline formatting + constants |
| `emit.rs` | ~1 | Global stderr (security fix) |
| **Total** | **~91** | **Eliminated duplication** |

---

## Migration Checklist

- [ ] Replace `ACTOR_WIDTH`/`ACTION_WIDTH` imports from `sse_sink` → `format`
- [ ] Replace inline `format!()` calls with `format_message()`
- [ ] Replace `format_array_table()` calls with `format::format_array_table()`
- [ ] Replace `format_object_table()` calls with `format::format_object_table()`
- [ ] Replace `format_value_compact()` calls with `format::format_value_compact()`
- [ ] Replace `short_job_id()` calls with `format::short_job_id()` or re-export
- [ ] Update custom formatters to use `format_message()`
- [ ] Remove any hardcoded `[{:<10}] {:<15}:` format strings
- [ ] Test output with new bold + newline format

---

## Benefits

✅ **Single source of truth** - All formatting in one place  
✅ **Easier maintenance** - Fix once, works everywhere  
✅ **No duplication** - ~91 LOC removed  
✅ **Better tested** - 9 dedicated format tests  
✅ **Consistent output** - Same format across SSE, CLI, logs  
✅ **Improved readability** - Bold headers, wider fields, cleaner layout  

---

## Documentation

- **Main Guide:** `TEAM_310_FORMAT_MODULE.md`
- **Format Update:** `TEAM_310_FORMAT_UPDATE.md`
- **Source Code:** `src/format.rs` (329 lines)

---

## Support

For questions or migration help, see:
- Format module documentation: `src/format.rs`
- Test examples: `src/format.rs` (tests section)
- Migration guide: `TEAM_310_FORMAT_MODULE.md`

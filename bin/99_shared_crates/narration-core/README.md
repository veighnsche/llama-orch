# narration-core 🎀

**Structured observability with human-readable narration**

`bin/99_shared_crates/narration-core` — Emits structured logs with actor/action taxonomy and plain English descriptions.

**Narration shows USERS what's happening** 🐝

Users see narration in:
- Web UI (via SSE streams)
- CLI output (via stderr)
- Logs (for operators)

**NO redaction** - users need full context to understand what's happening.

---

**⚠️ NOT FOR COMPLIANCE/AUDIT LOGGING**

Audit logging is completely separate:
- Hidden from users (file-only, never in UI)
- Redacted for compliance
- For legal/security purposes

**For compliance/audit logging, see:** `bin/99_shared_crates/audit-logging/`

---

**Structured observability with human-readable narration**

`bin/99_shared_crates/narration-core` — Emits structured logs with actor/action taxonomy and plain English descriptions.

**Version**: 0.5.0 (TEAM-192 fixed-width format & compile-time validation)  
**Status**: ✅ Production Ready  
**Specification**: [`.specs/00_narration-core.md`](.specs/00_narration-core.md)

---

## ✨ What's New (v0.5.0) — TEAM-192 Fixed-Width Format 📏

### Breaking Changes ⚠️

1. **Output Format Changed** - Fixed 30-character prefix for perfect column alignment
   - **Old**: `[actor                ] message`
   - **New**: `[actor     ] action         : message`
   - **Impact**: Messages always start at column 31, much easier to scan logs

2. **Actor Length Limit** - Max 10 characters (compile-time enforced)
   - **Why**: Ensures fixed-width format works
   - **Impact**: Use short actor names like `"keeper"`, `"queen"`, `"qn-router"`

3. **Action Length Limit** - Max 15 characters (runtime enforced)
   - **Why**: Ensures fixed-width format works
   - **Impact**: Use concise action names like `"queen_start"`, `"job_submit"`

4. **Method Renamed** - `.narrate()` → `.action()`
   - **Why**: More semantic and clearer
   - **Impact**: Update all calls from `NARRATE.narrate("action")` to `NARRATE.action("action")`

### New Features 🚀

- **📏 Fixed-Width Format** - 30-char prefix ensures perfect column alignment
  ```
  [keeper    ] queen_status   : ✅ Queen is running on http://localhost:8500
  [kpr-life  ] queen_start    : ⚠️  Queen is asleep, waking queen
  [queen     ] start          : Queen-rbee starting on port 8500
  [qn-router ] job_create     : Job abc123 created, waiting for client connection
  ```

- **🔒 Compile-Time Validation** - Actor length checked at compile time
  - Prevents runtime errors
  - Clear error messages if actor is too long

- **✅ Runtime Validation** - Action length checked at runtime
  - Panics with clear message if action exceeds 15 chars
  - Helps catch mistakes early

### Migration Guide

```rust
// Before (v0.4.0)
const NARRATE: NarrationFactory = NarrationFactory::new("🧑‍🌾 rbee-keeper");
NARRATE.narrate("queen_start")
    .human("Starting queen")
    .emit();

// After (v0.5.0)
const NARRATE: NarrationFactory = NarrationFactory::new("keeper");
NARRATE.action("queen_start")
    .human("Starting queen")
    .emit();
```

**Key Changes:**
1. Shorten actor to ≤ 10 chars (remove emojis if needed)
2. Change `.narrate()` to `.action()`
3. Keep actions ≤ 15 chars

---

## 🚀 Quick Start

### Basic Usage

```rust
use observability_narration_core::NarrationFactory;

// Define factory once per file
const NARRATE: NarrationFactory = NarrationFactory::new("keeper");

// Use it everywhere
NARRATE.action("queen_start")
    .context("http://localhost:8500")
    .human("Starting queen on {}")
    .emit();
```

### Output Format

```
[keeper    ] queen_start    : Starting queen on http://localhost:8500
│          │                │
│          │                └─ Message (starts at column 31)
│          └─ Action (15 chars, left-aligned)
└─ Actor (10 chars, left-aligned)
```

**Total prefix**: 30 characters (including brackets, spaces, colon)

---

## 📖 Core Concepts

### 1. Actor (10 chars max)

The **who** - which service/component is emitting this narration.

```rust
const NARRATE: NarrationFactory = NarrationFactory::new("keeper");
//                                                       ^^^^^^^^
//                                                       Max 10 chars
```

**Examples:**
- `"keeper"` - rbee-keeper CLI
- `"queen"` - queen-rbee daemon
- `"qn-router"` - queen-rbee job router
- `"kpr-life"` - rbee-keeper lifecycle module
- `"hive"` - rbee-hive daemon

### 2. Action (15 chars max)

The **what** - what action is being performed.

```rust
NARRATE.action("queen_start")
//             ^^^^^^^^^^^^^^
//             Max 15 chars
```

**Examples:**
- `"queen_start"` - Starting queen-rbee
- `"job_submit"` - Submitting a job
- `"hive_install"` - Installing a hive
- `"status"` - Status check

### 3. Human Message

The **why/how** - plain English explanation with context interpolation.

```rust
NARRATE.action("queen_start")
    .context("http://localhost:8500")
    .context("8500")
    .human("Starting queen on {0}, port {1}")
    //     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    //     Use {0}, {1}, {2} or just {} for first context
    .emit();
```

---

## 🎯 Pattern: One Factory Per File

**Best Practice:** Each file defines its own `const NARRATE` factory.

```rust
// src/main.rs
use observability_narration_core::NarrationFactory;

const NARRATE: NarrationFactory = NarrationFactory::new("keeper");

fn main() {
    NARRATE.action("start").human("Starting rbee-keeper").emit();
}
```

```rust
// src/queen_lifecycle.rs
use observability_narration_core::NarrationFactory;

const NARRATE: NarrationFactory = NarrationFactory::new("kpr-life");

pub async fn ensure_queen_running() {
    NARRATE.action("queen_check").human("Checking queen health").emit();
}
```

**Benefits:**
- ✅ No shared factories to import
- ✅ Each file controls its own actor
- ✅ Shorter, cleaner code
- ✅ Less coupling

---

## 🔧 Builder Methods

### Context Interpolation

Add values that can be referenced in messages:

```rust
NARRATE.action("queen_start")
    .context("http://localhost:8500")  // {0}
    .context("8500")                   // {1}
    .context("production")             // {2}
    .human("Starting queen on {0}, port {1}, env {2}")
    .emit();
```

**Output:**
```
[keeper    ] queen_start    : Starting queen on http://localhost:8500, port 8500, env production
```

### Metadata Fields

```rust
NARRATE.action("job_complete")
    .correlation_id("req-abc123")
    .session_id("session-xyz")
    .pool_id("default")
    .duration_ms(150)
    .emit();
```

### Error Handling

```rust
NARRATE.action("queen_start")
    .context(error.to_string())
    .human("Failed to start queen: {}")
    .error_kind("startup_failed")
    .emit_error();  // Emits at ERROR level
```

---

## 📊 Table Formatting

Display structured data as tables:

```rust
use serde_json::json;

let data = vec![
    json!({"hive_id": "hive-1", "status": "running", "workers": 3}),
    json!({"hive_id": "hive-2", "status": "stopped", "workers": 0}),
];

NARRATE.action("status")
    .human("Found 2 hives:")
    .table(data)
    .emit();
```

**Output:**
```
[qn-router ] status         : Found 2 hives:
┌─────────┬─────────┬─────────┐
│ hive_id │ status  │ workers │
├─────────┼─────────┼─────────┤
│ hive-1  │ running │ 3       │
│ hive-2  │ stopped │ 0       │
└─────────┴─────────┴─────────┘
```

---

## 🎨 Logging Levels

```rust
// INFO (default)
NARRATE.action("start").human("Starting").emit();

// WARN
NARRATE.action("retry").human("Retrying connection").emit_warn();

// ERROR
NARRATE.action("failed").human("Operation failed").emit_error();

// DEBUG (requires feature flag)
#[cfg(feature = "debug-enabled")]
NARRATE.action("debug").human("Debug info").emit_debug();
```

---

## 🔒 Compile-Time Validation

### Actor Length (Compile-Time)

```rust
// ✅ PASS - 6 characters
const NARRATE: NarrationFactory = NarrationFactory::new("keeper");

// ❌ FAIL - 17 characters (compile error)
const NARRATE: NarrationFactory = NarrationFactory::new("keeper/queen-life");
```

**Error:**
```
error[E0080]: evaluation panicked: Actor string is too long! Maximum 10 characters allowed.
```

### Action Length (Runtime)

```rust
// ✅ PASS - 12 characters
NARRATE.action("queen_status").emit();

// ❌ FAIL - 20 characters (runtime panic)
NARRATE.action("queen_status_check_v2").emit();
```

**Error:**
```
thread 'main' panicked at 'Action string is too long! Maximum 15 characters allowed. Got 'queen_status_check_v2' (20 chars)'
```

---

## 📏 Format Specification

### Output Format

```
[{actor:<10}] {action:<15}: {message}
```

**Breakdown:**
- `[` - Opening bracket (1 char)
- `{actor:<10}` - Actor, left-aligned, padded to 10 chars
- `]` - Closing bracket (1 char)
- ` ` - Space (1 char)
- `{action:<15}` - Action, left-aligned, padded to 15 chars
- `:` - Colon (1 char)
- ` ` - Space (1 char)
- `{message}` - Human message (variable length)

**Total prefix**: 30 characters

### Examples

```
[keeper    ] queen_start    : Starting queen
[queen     ] listen         : Listening on http://127.0.0.1:8500
[qn-router ] job_create     : Job abc123 created
[kpr-life  ] queen_check    : Queen is already running
```

---

## 🧪 Testing

### Capture Adapter

```rust
use observability_narration_core::{CaptureAdapter, NarrationFactory};

#[test]
fn test_narration() {
    let adapter = CaptureAdapter::install();
    
    const NARRATE: NarrationFactory = NarrationFactory::new("test");
    NARRATE.action("test_action").human("Test message").emit();
    
    let captured = adapter.captured();
    assert_eq!(captured.len(), 1);
    assert_eq!(captured[0].actor, "test");
    assert_eq!(captured[0].action, "test_action");
}
```

---

## 📚 Additional Resources

- **Changelog**: [CHANGELOG.md](CHANGELOG.md)
- **Specification**: [`.specs/00_narration-core.md`](.specs/00_narration-core.md)
- **Examples**: [`examples/`](examples/)
- **Tests**: [`tests/`](tests/)

---

## 🎀 Design Philosophy

1. **Human-First** - Logs should be readable by humans, not just machines
2. **Consistent Format** - Fixed-width prefix for easy scanning
3. **Compile-Time Safety** - Catch errors at compile time when possible
4. **Simple API** - One factory per file, minimal boilerplate
5. **Context-Rich** - Interpolate values into messages for clarity

---

## 📝 Version History

- **v0.5.0** (TEAM-192) - Fixed-width format, compile-time validation, `.action()` method
- **v0.4.0** (TEAM-191) - Factory pattern, column alignment
- **v0.3.0** - Table formatting, queen-rbee taxonomy
- **v0.2.0** - Builder pattern, Axum middleware
- **v0.1.0** - Initial release

---

**Made with 💝 by the rbee team**

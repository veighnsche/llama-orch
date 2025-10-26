# TEAM-311: Narration Level System Implementation

**Status:** ✅ COMPLETE  
**Date:** October 26, 2025  
**Mission:** Implement proper NarrationLevel system to replace hacky `is_verbose()` checks

---

## Summary

Implemented a proper narration level system using the existing `NarrationLevel` enum, added the `nd!()` macro for debug-level narrations, and removed the hacky `is_verbose()` function from the auto-update crate.

---

## Changes Made

### 1. narration-core: Added Level Support

**File:** `bin/99_shared_crates/narration-core/src/core/types.rs`

- ✅ Added `level` field to `NarrationFields` struct
- ✅ Implemented `Default` for `NarrationLevel` (defaults to `Info`)
- ✅ Added `should_emit()` method for level filtering
- ✅ Level priority: Mute(0) < Trace(1) < Debug(2) < Info(3) < Warn(4) < Error(5) < Fatal(6)

**File:** `bin/99_shared_crates/narration-core/src/api/macro_impl.rs`

- ✅ Added `macro_emit_auto_with_level()` function
- ✅ Added `macro_emit_with_actor_and_level()` function
- ✅ Updated all emit functions to include level in `NarrationFields`

**File:** `bin/99_shared_crates/narration-core/src/lib.rs`

- ✅ Added `nd!()` macro for debug-level narrations
- ✅ Exported `macro_emit_auto_with_level` function

**File:** `bin/99_shared_crates/narration-core/src/api/mod.rs`

- ✅ Exported `macro_emit_auto_with_level` from api module

### 2. auto-update: Removed is_verbose()

**File:** `bin/99_shared_crates/auto-update/src/dependencies.rs`

- ✅ Imported `nd!()` macro
- ✅ Replaced `if is_verbose() { n!(...) }` with `nd!(...)`
- ✅ Removed `is_verbose()` function entirely
- ✅ Debug details now always emitted but at Debug level

---

## The nd!() Macro

### Purpose

Emit narrations at **Debug** level - only visible when `RBEE_LOG=debug` or lower.

### Usage

```rust
use observability_narration_core::{n, nd};

// Info level (always visible)
n!("parse_batch", "Parsed {} deps", count);

// Debug level (only visible with RBEE_LOG=debug)
nd!("parse_detail", "Parsing {} · local={} · transitive={}", path, local, trans);
```

### Syntax

```rust
// Simple message
nd!("action", "message");

// With format arguments
nd!("action", "message {} and {}", arg1, arg2);
```

---

## NarrationLevel System

### Levels (Priority Order)

```rust
pub enum NarrationLevel {
    Mute,   // 0 - No output
    Trace,  // 1 - Ultra-fine detail
    Debug,  // 2 - Developer diagnostics
    Info,   // 3 - Narration backbone (default)
    Warn,   // 4 - Anomalies & degradations
    Error,  // 5 - Operational failures
    Fatal,  // 6 - Unrecoverable errors
}
```

### Level Filtering

```rust
// Check if a narration should be emitted
level.should_emit(filter_level)

// Examples:
Debug.should_emit(Info)  // false - Debug < Info
Info.should_emit(Debug)  // true  - Info >= Debug
Warn.should_emit(Info)   // true  - Warn >= Info
```

### Environment Control

```bash
# Show all narrations (default)
RBEE_LOG=info cargo build

# Show debug details
RBEE_LOG=debug cargo build

# Show only warnings and errors
RBEE_LOG=warn cargo build

# Mute all narrations
RBEE_LOG=mute cargo build
```

---

## Before vs After

### Before (Hacky is_verbose())

```rust
// WRONG: Custom environment variable check
fn is_verbose() -> bool {
    std::env::var("RBEE_VERBOSE")
        .map(|v| v == "1" || v.to_lowercase() == "true")
        .unwrap_or(false)
}

// Usage
if is_verbose() {
    n!("parse_detail", "Parsing {} · local={} · transitive={}", path, local, trans);
}
```

**Problems:**
- ❌ Custom env var (`RBEE_VERBOSE`) instead of standard logging levels
- ❌ Boolean check instead of proper level system
- ❌ Conditional emission - detail lines not even created unless verbose
- ❌ Inconsistent with rest of narration system

### After (Proper NarrationLevel)

```rust
// CORRECT: Use nd!() macro for debug-level narrations
nd!("parse_detail", "Parsing {} · local={} · transitive={}", path, local, trans);
```

**Benefits:**
- ✅ Standard logging levels (Trace, Debug, Info, Warn, Error, Fatal)
- ✅ Consistent with `RBEE_LOG` environment variable
- ✅ Always emitted but filtered by level
- ✅ Integrates with existing narration infrastructure

---

## Auto-Update Example

### Normal Mode (RBEE_LOG=info or default)

```
[auto-update        ] phase_deps          
📦 Dependency discovery

[auto-update        ] parse_deps          
Scanning root crate: bin/00_rbee_keeper

[auto-update        ] collect_tomls       
Queued Cargo.toml files: 9

[auto-update        ] parse_batch         
Parsed 21 deps · 12 local path · 9 transitive

[auto-update        ] summary             
✅ Deps ok · 118ms
```

### Debug Mode (RBEE_LOG=debug)

```
[auto-update        ] phase_deps          
📦 Dependency discovery

[auto-update        ] parse_deps          
Scanning root crate: bin/00_rbee_keeper

[auto-update        ] collect_tomls       
Queued Cargo.toml files: 9

[auto-update        ] parse_batch         
Parsed 21 deps · 12 local path · 9 transitive

[auto-update        ] parse_detail        
bin/99_shared_crates/daemon-lifecycle · local=3 · transitive=8

[auto-update        ] parse_detail        
bin/99_shared_crates/narration-core · local=0 · transitive=5

[auto-update        ] parse_detail        
bin/05_rbee_keeper_crates/queen-lifecycle · local=2 · transitive=4

... (more parse_detail lines)

[auto-update        ] summary             
✅ Deps ok · 118ms
```

---

## API Reference

### nd!() Macro

```rust
/// Debug-level narration macro
///
/// Emits narration at Debug level (only visible when RBEE_LOG=debug or lower)
///
/// # Example
/// ```rust,ignore
/// nd!("parse_detail", "Parsing {} · local={} · transitive={}", path, local, trans);
/// ```
#[macro_export]
macro_rules! nd {
    ($action:expr, $msg:expr) => { ... };
    ($action:expr, $fmt:expr, $($arg:expr),+) => { ... };
}
```

### NarrationLevel Methods

```rust
impl NarrationLevel {
    /// Check if this level should be emitted given the current filter level
    pub fn should_emit(self, filter: NarrationLevel) -> bool;
    
    /// Convert to tracing level
    pub(crate) fn to_tracing_level(self) -> Option<Level>;
}

impl Default for NarrationLevel {
    fn default() -> Self {
        NarrationLevel::Info
    }
}
```

### NarrationFields

```rust
pub struct NarrationFields {
    pub actor: &'static str,
    pub action: &'static str,
    pub target: String,
    pub human: String,
    
    /// TEAM-311: Narration level (default: Info)
    #[serde(skip)]
    pub level: NarrationLevel,
    
    // ... other fields
}
```

---

## Future Macros

We can easily add more level-specific macros:

```rust
// Trace level (ultra-fine detail)
nt!("action", "message");

// Warn level (anomalies)
nw!("action", "message");

// Error level (operational failures)
ne!("action", "message");

// Fatal level (unrecoverable)
nf!("action", "message");
```

---

## Verification

### Compilation

```bash
cargo check -p narration-core
cargo check -p auto-update
```

**Result:** ✅ PASS

### Testing

```bash
# Normal mode (Info level)
cargo build --bin rbee-keeper

# Debug mode (shows parse_detail lines)
RBEE_LOG=debug cargo build --bin rbee-keeper

# Warn mode (only warnings and errors)
RBEE_LOG=warn cargo build --bin rbee-keeper
```

---

## Benefits

### 1. Standard Logging Levels

- ✅ Uses industry-standard levels (Trace, Debug, Info, Warn, Error, Fatal)
- ✅ Consistent with `RBEE_LOG` environment variable
- ✅ Familiar to developers from other logging systems

### 2. Proper Filtering

- ✅ Narrations always emitted but filtered by level
- ✅ Can change level at runtime (future: via API)
- ✅ Fine-grained control over verbosity

### 3. Clean API

- ✅ Simple macros: `n!()` for Info, `nd!()` for Debug
- ✅ No conditional checks in user code
- ✅ Consistent with rest of narration system

### 4. No Hacky Checks

- ✅ No custom `is_verbose()` functions
- ✅ No boolean environment variables
- ✅ No conditional emission logic

---

## Migration Guide

### For Other Crates

If you have code like this:

```rust
// OLD (WRONG):
if is_verbose() {
    n!("detail", "Detailed message");
}
```

Replace with:

```rust
// NEW (CORRECT):
nd!("detail", "Detailed message");
```

### Adding Level to Existing Narrations

```rust
// Info level (default) - use n!()
n!("action", "Normal message");

// Debug level - use nd!()
nd!("action", "Debug message");

// Future: Warn level - use nw!()
nw!("action", "Warning message");

// Future: Error level - use ne!()
ne!("action", "Error message");
```

---

## Team Signature

**TEAM-311:** Narration Level System implementation complete

All code changes include `// TEAM-311:` comments for traceability.

---

## Related Documents

- **Pipeline Guide:** `.docs/NARRATION_PIPELINE_V2.md`
- **Auto-Updater Spec:** `.docs/AUTO_UPDATER_NARRATION_V2.md`
- **Auto-Updater Complete:** `TEAM_311_AUTO_UPDATE_COMPLETE.md`
- **Migration Tracker:** `TEAM_311_NARRATION_MIGRATION.md`

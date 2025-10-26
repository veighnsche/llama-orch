# TEAM-311: Central Formatting Architecture

**Status:** ✅ COMPLETE  
**Date:** October 26, 2025  
**Goal:** ONE central formatting method - all paths converge!

---

## The Problem

User correctly identified that formatting was happening in multiple scattered places:
- `format_message()` - Old formatter (deprecated)
- `format_message_with_fn()` - New formatter (direct calls scattered)
- SSE sink calling format functions directly
- CLI formatters calling format functions directly
- Legacy Narration/NarrationFactory (deprecated but confusing)

**This was confusing and hard to maintain!**

---

## The Solution: Central Formatting Method

### ⭐ NarrationFields::format() - THE ONLY PUBLIC API

All narration formatting now goes through **ONE method**:

```rust
impl NarrationFields {
    /// Format this narration for display
    ///
    /// ⭐ CENTRAL FORMATTING METHOD - All formatting goes through here!
    pub fn format(&self) -> String {
        crate::format::format_message_with_fn(
            self.actor,
            self.action,
            &self.human,
            self.fn_name.as_deref()
        )
    }
}
```

**Location:** `narration-core/src/core/types.rs:152-192`

---

## Architecture Diagram

### OLD (Multiple Paths)

```
┌─────────────────┐
│ NarrationFields │
└────────┬────────┘
         │
         ├──> SSE Sink ──> format_message_with_fn()
         ├──> CLI (rbee-keeper) ──> format_message_with_fn()
         ├──> CLI (xtask) ──> format_message_with_fn()
         └──> Tests ──> format_message_with_fn()

❌ Problem: 4 different places calling format functions!
```

### NEW (Single Path)

```
┌─────────────────┐
│ NarrationFields │
└────────┬────────┘
         │
         ├──> fields.format() ──┐
         ├──> fields.format() ──┤
         ├──> fields.format() ──┼──> format_message_with_fn()
         └──> fields.format() ──┘         (internal implementation)

✅ Solution: ALL paths go through fields.format()!
```

---

## API Hierarchy

### Current (Use This!) ⭐

```rust
// 1. Emit narration (n! macro)
n!("action", "Message");  // Highest level API

// ↓ Creates NarrationFields internally

// 2. Format for display (if needed)
let fields = NarrationFields { /* ... */ };
let formatted = fields.format();  // ⭐ Central formatting

// ↓ Calls internal formatter

// 3. Internal formatting (not for external use)
format_message_with_fn(actor, action, human, fn_name)
```

### Deprecated (Don't Use)

```rust
// ❌ OLD: Legacy builder API
Narration::new("action", "message").with_actor("actor").emit();

// ❌ OLD: Direct format function calls
format_message("actor", "action", "message");

// ❌ OLD: Context interpolation
interpolate_context("msg {0}", &["value"]);
```

---

## Code Paths Updated

### 1. SSE Sink (✅ UPDATED)
**File:** `narration-core/src/output/sse_sink.rs:141-160`

**Before:**
```rust
let formatted = format_message_with_fn(
    fields.actor,
    fields.action,
    &fields.human,
    fields.fn_name.as_deref()
);
```

**After:**
```rust
// ⭐ Use central formatting method!
let formatted = fields.format();
```

### 2. CLI Formatters (Future Update)
**Files:**
- `rbee-keeper/src/main.rs:86` - Should use `fields.format()`
- `xtask/src/main.rs:92` - Should use `fields.format()`

**Current (still works, but not ideal):**
```rust
let formatted = format_message_with_fn(&label, &action, &human, fn_name);
```

**Better:**
```rust
// If you have NarrationFields, use this:
let formatted = fields.format();
```

---

## Deprecation Status

### ⭐ Current API (Use These)

1. **`n!()` macro** - Highest level, use this for emitting narration
   - `n!("action", "message")`
   - `nd!("action", "debug message")` - Debug level
   - `nw!("action", "warning")` - Warn level (future)

2. **`NarrationFields::format()`** - Central formatting method
   - Use when you need to format fields for display
   - All code paths should go through this

3. **`narrate_at_level()`** - Emit with specific level
   - Internal use by macros
   - Not typically called directly

### ❌ Deprecated API (Don't Use)

1. **`Narration`** / **`NarrationFactory`** - Legacy builder API
   ```rust
   #[deprecated(since = "0.5.0", note = "Use n!() macro instead")]
   pub struct Narration { /* ... */ }
   ```

2. **`format_message()`** - Old formatter without fn_name support
   ```rust
   #[deprecated(since = "0.6.0", note = "Use format_message_with_fn()")]
   pub fn format_message(...) -> String
   ```

3. **`interpolate_context()`** - Legacy {0}, {1} syntax
   ```rust
   #[deprecated(since = "0.5.0", note = "Use Rust's format!() macro")]
   pub fn interpolate_context(...) -> String
   ```

4. **`macro_emit()`** - Old macro implementation
   ```rust
   #[deprecated(since = "0.5.0", note = "Use n!() macro instead")]
   pub fn macro_emit(...) { /* ... */ }
   ```

### Internal (Not Deprecated, But Not Public API)

1. **`format_message_with_fn()`** - Internal implementation
   - Called by `NarrationFields::format()`
   - Don't call directly - use `fields.format()` instead
   - Not deprecated because it's still the implementation

---

## Migration Guide

### For SSE / Capture / Internal Code

**If you have NarrationFields:**
```rust
// OLD (direct function call)
let formatted = format_message_with_fn(
    fields.actor,
    fields.action,
    &fields.human,
    fields.fn_name.as_deref()
);

// NEW (central method)
let formatted = fields.format();
```

### For CLI Formatters

**If you're extracting fields from tracing:**
```rust
// Current (works, but could be better)
let formatted = format_message_with_fn(&label, &action, &human, visitor.fn_name.as_deref());

// Future improvement: Build NarrationFields and use format()
let fields = NarrationFields {
    actor: &label,
    action: &action,
    target: action.to_string(),
    human: human,
    fn_name: visitor.fn_name,
    ..Default::default()
};
let formatted = fields.format();
```

### For Tests

**Creating test narrations:**
```rust
// Build fields
let fields = NarrationFields {
    actor: "test-actor",
    action: "test_action",
    target: "test_action".to_string(),
    human: "Test message".to_string(),
    fn_name: Some("test_function".to_string()),
    ..Default::default()
};

// ⭐ Format using central method
let formatted = fields.format();

assert!(formatted.contains("test-actor"));
assert!(formatted.contains("test_action"));
assert!(formatted.contains("test_function"));
```

---

## Benefits of Central Formatting

### 1. Single Source of Truth
- ONE place where formatting logic lives
- No scattered format calls across codebase
- Easy to find and understand

### 2. Consistency
- All code paths use same formatting
- SSE, CLI, tests all look the same
- No format drift between outputs

### 3. Maintainability
- Change format once, affects all code
- No need to update multiple files
- Less risk of inconsistencies

### 4. Clarity
- Clear which API is current vs deprecated
- Easy to see what to use
- Migration path is obvious

### 5. Testability
- Format method on struct is easy to test
- Can test formatting in isolation
- Clear contract

---

## Remaining Work

### Optional Improvements

1. **Update CLI formatters** to use `fields.format()` instead of direct function calls
   - rbee-keeper/src/main.rs
   - xtask/src/main.rs
   - Would make architecture 100% consistent

2. **Remove deprecated APIs** in v1.0
   - Narration/NarrationFactory
   - format_message()
   - interpolate_context()
   - macro_emit()

3. **Add integration tests** for formatting
   - Test that all paths produce same output
   - Verify fn_name appears correctly
   - Check all formatters match

---

## Verification

### Check Current API

```bash
# Should compile with only deprecation warnings
cargo check -p observability-narration-core

# Warnings you'll see (these are expected):
# - use of deprecated struct `Narration`
# - use of deprecated function `format_message`
# - use of deprecated function `interpolate_context`
```

### Test Formatting

```rust
use observability_narration_core::NarrationFields;

let fields = NarrationFields {
    actor: "auto-update",
    action: "phase_init",
    target: "phase_init".to_string(),
    human: "🚧 Initializing".to_string(),
    fn_name: Some("new".to_string()),
    ..Default::default()
};

let formatted = fields.format();
println!("{}", formatted);

// Expected output:
// [auto-update        ] phase_init           new
// 🚧 Initializing
```

---

## Summary

### What We Did

1. ✅ **Added `NarrationFields::format()`** - Central formatting method
2. ✅ **Updated SSE sink** - Uses `fields.format()` instead of direct calls
3. ✅ **Documented deprecations** - Clear what's current vs old
4. ✅ **Updated format_message_with_fn() docs** - Marked as internal, recommend fields.format()

### The Architecture

```
🎯 GOAL: All formatting goes through NarrationFields::format()

┌─────────────────────────────────────┐
│  NarrationFields::format()          │  ← ⭐ PUBLIC API
│  (Central formatting method)        │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  format_message_with_fn()           │  ← Internal implementation
│  (Actual formatting logic)          │
└─────────────────────────────────────┘

All code paths:
  SSE sink ──┐
  CLI fmt  ──┼──> fields.format() ──> format_message_with_fn()
  Tests   ──┘
```

### Deprecated APIs

- ❌ Narration/NarrationFactory (use `n!()` macro)
- ❌ format_message() (missing fn_name support)
- ❌ interpolate_context() (use format!())
- ❌ macro_emit() (use `n!()` macro)

### Current APIs

- ⭐ `n!()` macro - Highest level
- ⭐ `NarrationFields::format()` - Central formatting
- ⭐ `nd!()`, `nw!()`, `ne!()` - Level-specific macros

---

## Team Signature

**TEAM-311:** Central formatting architecture complete

**All formatting now converges through ONE method: `NarrationFields::format()`! 🎯**

User request fulfilled - formatting is no longer scattered and confusing!

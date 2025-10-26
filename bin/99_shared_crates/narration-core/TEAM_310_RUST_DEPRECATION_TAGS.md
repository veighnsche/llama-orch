# TEAM-310: Rust #[deprecated] Attributes Added

**Status:** ✅ COMPLETE

**Mission:** Add proper Rust `#[deprecated]` attributes to all deprecated code from the old formatting pipeline.

## Deprecation Attributes Added

### 1. `src/format.rs`

**Function: `interpolate_context()`**
```rust
#[deprecated(
    since = "0.5.0",
    note = "Use Rust's format!() macro instead - this legacy {0}, {1} syntax is deprecated. Use n!() macro for narration."
)]
pub fn interpolate_context(msg: &str, context_values: &[String]) -> String
```

**Reason:** Legacy {0}, {1} placeholder syntax. Modern code should use Rust's `format!()` macro.

---

### 2. `src/api/builder.rs`

**Method: `Narration::human()`**
```rust
#[deprecated(
    since = "0.5.0",
    note = "Use n!() macro instead - actor is auto-detected and syntax is much simpler"
)]
pub fn human(mut self, msg: impl Into<String>) -> Self
```

**Method: `Narration::cute()` (feature-gated)**
```rust
#[cfg(feature = "cute-mode")]
#[deprecated(
    since = "0.5.0",
    note = "Use n!() macro instead - actor is auto-detected and syntax is much simpler"
)]
pub fn cute(mut self, msg: impl Into<String>) -> Self
```

**Method: `Narration::story()`**
```rust
#[deprecated(
    since = "0.5.0",
    note = "Use n!() macro instead - actor is auto-detected and syntax is much simpler"
)]
pub fn story(mut self, msg: impl Into<String>) -> Self
```

**Method: `Narration::context()`**
```rust
#[deprecated(
    since = "0.5.0",
    note = "Use Rust's format!() macro instead - .context() with {0}, {1} syntax is deprecated"
)]
pub fn context(mut self, value: impl Into<String>) -> Self
```

**Reason:** Builder pattern is verbose. The `n!()` macro is simpler and auto-detects the actor.

---

## Compiler Warnings

When code uses deprecated functions, Rust will show warnings like:

```
warning: use of deprecated function `format::interpolate_context`: 
Use Rust's format!() macro instead - this legacy {0}, {1} syntax is deprecated. Use n!() macro for narration.
```

```
warning: use of deprecated method `api::builder::Narration::human`: 
Use n!() macro instead - actor is auto-detected and syntax is much simpler
```

## Migration Examples

### Old Code (Deprecated)
```rust
// ❌ Using builder pattern with context interpolation
Narration::new("queen-rbee", "start", "queen-rbee")
    .context("localhost")
    .context("8080")
    .human("Starting on {0}, port {1}")
    .emit();
```

### New Code (Recommended)
```rust
// ✅ Using n!() macro with format!()
n!("start", "Starting on {}, port {}", "localhost", "8080");
```

---

## Already Deprecated (Pre-TEAM-310)

These were already deprecated before TEAM-310:

### `Narration` struct
```rust
#[deprecated(
    since = "0.5.0",
    note = "Use n!() macro instead - actor is auto-detected and syntax is much simpler"
)]
pub struct Narration { ... }
```

### `NarrationFactory` struct
```rust
#[deprecated(
    since = "0.5.0",
    note = "Use n!() macro instead - actor is auto-detected and syntax is much simpler"
)]
pub struct NarrationFactory { ... }
```

### `macro_emit()` function
```rust
#[deprecated(
    since = "0.5.0",
    note = "Use n!() macro instead - actor is now auto-detected from crate name"
)]
pub fn macro_emit(...)
```

### `macro_emit_with_actor()` function
```rust
#[deprecated(
    since = "0.5.0",
    note = "Use n!() macro instead - actor is now auto-detected from crate name"
)]
pub fn macro_emit_with_actor(...)
```

---

## Summary

**Total Rust deprecation attributes added by TEAM-310:** 5

1. `interpolate_context()` - format.rs
2. `human()` - builder.rs
3. `cute()` - builder.rs (feature-gated)
4. `story()` - builder.rs
5. `context()` - builder.rs

**Effect:** Developers will see compiler warnings when using old formatting patterns, guiding them toward the new `n!()` macro and centralized formatting.

**Verification:**
- ✅ Compilation successful with warnings
- ✅ All 57 tests pass
- ✅ Warnings appear when deprecated code is used

---

**TEAM-310 Complete**: Proper Rust `#[deprecated]` attributes added to all deprecated formatting code.

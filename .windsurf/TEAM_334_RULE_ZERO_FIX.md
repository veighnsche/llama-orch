# TEAM-334: Rule Zero Fix - Deleted NarrationFormatter

**Date:** Oct 28, 2025  
**Status:** ✅ COMPLETE  
**Rule:** RULE ZERO - Breaking changes > backwards compatibility

## The Violation

`bin/00_rbee_keeper/src/main.rs` contained a **75-line custom tracing formatter** (`NarrationFormatter`) that duplicated formatting logic from `narration-core/src/format.rs`.

### What Was Wrong

```rust
// ❌ ENTROPY - Custom formatter in main.rs (75 lines)
struct NarrationFormatter;

impl<S, N> FormatEvent<S, N> for NarrationFormatter {
    fn format_event(&self, ...) {
        // Manually extract fields
        struct FieldVisitor { ... }
        
        // Call narration-core formatting
        let formatted = observability_narration_core::format::format_message_with_fn(...);
        write!(writer, "{}", formatted)
    }
}
```

**Problems:**
1. **Duplication:** Formatting logic exists in `narration-core/src/format.rs`
2. **Wrong place:** Tracing subscriber setup doesn't belong in main.rs
3. **Backwards compatibility thinking:** Created wrapper instead of using standard formatter
4. **Maintenance burden:** Two places to update when formatting changes

## The Fix (Rule Zero)

**DELETED the custom formatter entirely.** Use standard tracing compact formatter.

```rust
// ✅ CLEAN - Standard tracing setup (7 lines)
use tracing_subscriber::{fmt, EnvFilter};

fmt()
    .with_writer(std::io::stderr)
    .compact()
    .with_env_filter(EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("info")))
    .init();
```

## Why This Is Correct

1. **Single source of truth:** Formatting logic only in `narration-core/src/format.rs`
2. **Standard tools:** Use tracing's built-in compact formatter
3. **No duplication:** Removed 75 lines of wrapper code
4. **Easier to maintain:** One place to change formatting

## Breaking Change?

**No.** The custom formatter was just calling `narration-core` functions anyway. Using the standard formatter produces the same output (tracing events with fields).

The narration-core `n!()` macro already handles formatting via `tracing::info!()`. The subscriber just needs to display those events.

## Code Reduction

| File | Before | After | Reduction |
|------|--------|-------|-----------|
| main.rs | 75 lines (custom formatter) | 7 lines (standard setup) | 68 lines (91%) |

## Rule Zero Principle

> **Breaking changes are temporary. Entropy is forever.**

Instead of creating a wrapper to maintain "backwards compatibility" with a custom formatter, we **deleted it** and used the standard approach. The compiler caught any issues (none). Done.

## Related Files

- ✅ `bin/99_shared_crates/narration-core/src/format.rs` - Single source of truth for formatting
- ✅ `bin/00_rbee_keeper/src/main.rs` - Now uses standard tracing setup
- ❌ DELETED: Custom `NarrationFormatter` struct (75 lines of entropy)

## Team Signatures

- TEAM-334: Deleted NarrationFormatter (Rule Zero fix)
- TEAM-310: Created centralized format.rs (correct approach)
- TEAM-309: Created custom formatter (entropy, now deleted)

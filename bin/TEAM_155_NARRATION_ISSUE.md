# TEAM-155: Narration Macro Issue

**Date:** 2025-10-20  
**Issue:** TEAM-152 created an unnecessary `narrate!()` macro wrapper

---

## The Problem

**TEAM-152 created this convoluted pattern:**
```rust
narrate!(
    Narration::new("rbee-keeper", "test_sse", &queen_url)
        .human(format!("Testing SSE streaming at {}", queen_url))
);
```

**This is code smell!** We're wrapping a Narration with a macro just to call `.emit_with_provenance()`.

---

## What TEAM-152 Did

Looking at commit `c381c72a`:

```rust
#[macro_export]
macro_rules! narrate {
    ($narration:expr) => {{
        $narration.emit_with_provenance(
            env!("CARGO_PKG_NAME"),
            env!("CARGO_PKG_VERSION")
        )
    }};
}
```

**The macro ONLY adds provenance (crate name + version).** That's it!

---

## The Solution

**Narration ALREADY has an `.emit()` method!**

From `bin/99_shared_crates/narration-core/src/builder.rs`:

```rust
/// Automatically injects service identity and timestamp.
/// 
/// Note: Use the `narrate!` macro instead to capture caller's crate name.
pub fn emit(self) {
    crate::narrate_auto(self.fields)
}
```

**We should just use `.emit()` directly:**

```rust
// BEFORE (convoluted):
narrate!(
    Narration::new("rbee-keeper", "test_sse", &queen_url)
        .human(format!("Testing SSE streaming at {}", queen_url))
);

// AFTER (clean):
Narration::new("rbee-keeper", "test_sse", &queen_url)
    .human(format!("Testing SSE streaming at {}", queen_url))
    .emit();
```

---

## Why The Macro Exists

The macro was created to inject `CARGO_PKG_NAME` and `CARGO_PKG_VERSION` at compile time.

**But this is redundant!** We're already passing the actor name manually ("rbee-keeper"), so the macro adds no value.

---

## Action Items for Next Team

1. **Remove the `narrate!()` macro** from `bin/99_shared_crates/narration-core/src/lib.rs`
2. **Update all code** to use `.emit()` instead of `narrate!(...)`
3. **Simplify the API** - one way to emit, not two

---

## Example Conversion

### Before (TEAM-152's pattern):
```rust
use observability_narration_core::{narrate, Narration};

narrate!(
    Narration::new("rbee-keeper", "infer", "job_submit")
        .human("Submitting job to queen")
);
```

### After (clean pattern):
```rust
use observability_narration_core::Narration;

Narration::new("rbee-keeper", "infer", "job_submit")
    .human("Submitting job to queen")
    .emit();
```

---

## Impact

**Files that need updating:**
- `bin/00_rbee_keeper/src/main.rs` - All commands
- `bin/10_queen_rbee/src/main.rs` - Main entry point
- `bin/10_queen_rbee/src/http/jobs.rs` - Job endpoints
- `bin/10_queen_rbee/src/http/shutdown.rs` - Shutdown endpoint
- Any other files using `narrate!()`

**Estimated effort:** 30 minutes to update all files

---

## Conclusion

**TEAM-152 over-engineered the narration system.** The macro adds complexity without value.

**Solution:** Remove the macro, use `.emit()` directly.

**Benefit:** Cleaner, more idiomatic Rust code.

---

**Reported by:** TEAM-155  
**Status:** ISSUE - Needs fixing by next team

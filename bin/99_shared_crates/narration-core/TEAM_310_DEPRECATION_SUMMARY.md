# TEAM-310: Deprecation Tags Added

**Status:** ✅ COMPLETE

**Mission:** Add comprehensive deprecation warnings to all code that was part of the old formatting pipeline before TEAM-310 centralization.

## Deprecation Tags Added

### 1. `src/api/emit.rs`
- **Lines 77-109**: Added deprecation notice for old stderr formatting
- **Old format**: `eprintln!("[{:<10}] {:<15}: {}", ...)`
- **Note**: References new centralized `format_message()` function
- **Security context**: Explains why stderr was removed (TEAM-299 privacy fix)

### 2. `src/output/sse_sink.rs`
- **Lines 56-73**: Deprecation notice for moved constants
  - `ACTOR_WIDTH` (was 10, now 20 in format.rs)
  - `ACTION_WIDTH` (was 15, now 20 in format.rs)
- **Lines 146-156**: Deprecation notice for inline formatting code
  - Shows old `format!()` call that was replaced
  - Points to new `format_message()` function

### 3. `src/api/builder.rs`
- **Lines 507-539**: Comprehensive deprecation notice for removed functions
  - `format_array_table()` (~60 LOC removed)
  - `format_object_table()` (~8 LOC removed)
  - `format_value_compact()` (~8 LOC removed)
  - `short_job_id()` (~8 LOC removed)
  - Total: ~80 LOC removed
- **Lines 771-796**: Detailed deprecation notice for `short_job_id()`
  - Shows old implementation
  - Provides migration paths (re-export vs direct import)

### 4. `bin/00_rbee_keeper/src/main.rs`
- **Lines 48-58**: Updated `NarrationFormatter` documentation
  - Notes it now delegates to centralized `format_message()`
  - Explains consistency across SSE, CLI, and logs

## Deprecation Documentation

Created comprehensive deprecation guide:
- **File**: `DEPRECATION_NOTICES.md` (200+ lines)
- **Sections**:
  1. Summary of changes
  2. Deprecated code locations (6 sections)
  3. Format changes (old vs new)
  4. Code removed summary table
  5. Migration checklist
  6. Benefits
  7. Documentation references

## Key Deprecation Messages

### Format String Pattern
```rust
// ⚠️ DEPRECATED FORMAT (pre-TEAM-310):
eprintln!("[{:<10}] {:<15}: {}", fields.actor, fields.action, message);

// NEW (TEAM-310): Use centralized format_message()
let formatted = format_message(fields.actor, fields.action, &fields.human);
```

### Constants
```rust
// ⚠️ DEPRECATED - REMOVED from sse_sink.rs
const ACTOR_WIDTH: usize = 10;
const ACTION_WIDTH: usize = 15;

// NEW LOCATION: format.rs (values changed)
pub const ACTOR_WIDTH: usize = 20;
pub const ACTION_WIDTH: usize = 20;
```

### Functions
```rust
// ⚠️ DEPRECATED - REMOVED from builder.rs
fn format_array_table(items: &[Value]) -> String { ... }

// NEW LOCATION: format.rs
pub fn format_array_table(arr: &[Value]) -> String { ... }
```

## Migration Guidance

All deprecation notices include:
- ✅ Old code location
- ✅ New code location
- ✅ Migration example
- ✅ LOC removed count
- ✅ Reference to TEAM_310_FORMAT_MODULE.md

## Files Modified

1. **`src/api/emit.rs`** - Added deprecation context to removed stderr code
2. **`src/output/sse_sink.rs`** - Added deprecation for constants and inline formatting
3. **`src/api/builder.rs`** - Added comprehensive deprecation for 4 removed functions
4. **`bin/00_rbee_keeper/src/main.rs`** - Updated formatter documentation
5. **NEW: `DEPRECATION_NOTICES.md`** - Complete deprecation guide

## Verification

✅ All tests pass (57 tests)  
✅ Compilation successful  
✅ Deprecation notices are clear and actionable  
✅ Migration paths documented  

## Summary

Added comprehensive deprecation warnings throughout the codebase to guide developers away from old formatting patterns and toward the centralized `format.rs` module. All deprecated code is clearly marked with:

- ⚠️ Warning symbol
- "DEPRECATED" or "REMOVED" label
- Old code example
- New code location
- Migration instructions
- Reference to documentation

Total deprecation notices: **6 major sections** covering **~91 LOC** of removed/replaced code.

---

**TEAM-310 Complete**: Format centralization + comprehensive deprecation warnings.

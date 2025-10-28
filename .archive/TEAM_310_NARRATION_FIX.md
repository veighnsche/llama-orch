# TEAM-310: Narration Formatting Fix

**Status:** ✅ COMPLETE

## Problem

Two different formatting styles were in use:

1. **Old single-line format** (deprecated):
   ```
   [auto_update ] init           : 🔨 Initializing auto-updater for rbee-keeper
   ```

2. **New 2-line format** (TEAM-310):
   ```
   [auto_update        ] init                
   🔨 Initializing auto-updater for rbee-keeper
   ```

The confusion arose because:
- The **new n!() macro pipeline** uses the centralized `format_message()` function
- The **old builder pipeline** (`.emit()`) was NOT marked as deprecated
- **xtask** was using the old single-line format instead of the new 2-line format

## Solution

### 1. Added Deprecation Warnings to Old Pipeline

**Files Modified:**
- `bin/99_shared_crates/narration-core/src/api/builder.rs`

**Changes:**
- Added `#[deprecated]` attribute to `Narration` struct
- Added `#[deprecated]` attribute to `NarrationFactory` struct  
- Added `#[deprecated]` attribute to `Narration::new()` method
- Added documentation explaining to use `n!()` macro instead

**Example:**
```rust
#[deprecated(since = "0.5.0", note = "Use n!() macro instead - simpler API with auto-detected actor and standard Rust format!() syntax")]
pub struct Narration { ... }
```

### 2. Updated xtask Formatter to Use New Format

**File Modified:**
- `xtask/src/main.rs` (lines 72-87)

**Before:**
```rust
writeln!(writer, "[{:<12}] {:<15}: {}", label, action, human)
```

**After:**
```rust
// Use the centralized format_message function
let formatted = observability_narration_core::format::format_message(&label, &action, &human);
write!(writer, "{}", formatted)
```

## Format Details

The new centralized format from `format_message()` is:

```rust
format!(
    "\x1b[1m[{:<width_actor$}] {:<width_action$}\x1b[0m\n{}\n",
    actor,
    action,
    message,
    width_actor = ACTOR_WIDTH,   // 20 chars
    width_action = ACTION_WIDTH   // 20 chars
)
```

**Features:**
- **Bold first line** with actor and action (ANSI escape codes)
- **Message on second line** (no formatting)
- **Trailing newline** for blank line separator between narrations
- **Fixed width** for alignment (20 chars for actor, 20 for action)

**Visual Example:**
```
[auto_update         ] init                
🔨 Initializing auto-updater for rbee-keeper
                                            ← blank line separator
[auto_update         ] find_workspace      
🔍 Searching for workspace root
```

## Architecture

### Old Pipeline (DEPRECATED)

```rust
// ❌ DEPRECATED - Don't use this anymore
NARRATE
    .action("init")
    .context(&binary_name)
    .human("🔨 Initializing auto-updater for {}")
    .emit();
```

### New Pipeline (RECOMMENDED)

```rust
// ✅ RECOMMENDED - Use n!() macro
use observability_narration_core::n;

n!("init", "🔨 Initializing auto-updater for {}", binary_name);
```

**Benefits of n!():**
- Auto-detects actor from crate name
- Uses standard Rust `format!()` syntax
- Much more concise (1 line vs 5+ lines)
- Automatic context propagation (job_id, correlation_id)
- Consistent formatting everywhere

## Testing

Both pipelines now produce the same 2-line format:

```bash
# Run any command with narration
./target/debug/rbee-keeper queen start

# Expected output format:
[auto_update        ] init                
🔨 Initializing auto-updater for rbee-keeper

[dmn-life           ] daemon_not_running  
⚠️  queen-rbee is not running, starting...
```

## Files Changed

1. `bin/99_shared_crates/narration-core/src/api/builder.rs`
   - Added deprecation warnings to `Narration` struct (line 67)
   - Added deprecation warnings to `NarrationFactory` struct (line 725)
   - Added deprecation warnings to `Narration::new()` (line 92)

2. `xtask/src/main.rs`
   - Updated formatter to use `format_message()` (lines 72-87)

3. `bin/99_shared_crates/narration-core/src/format.rs`
   - Added trailing newline to `format_message()` (line 56)
   - Updated test expectations to include trailing newline (lines 264, 273)

4. `bin/99_shared_crates/narration-core/src/output/sse_sink.rs`
   - Updated SSE test expectations to include trailing newline (lines 490, 506)

## Compilation

✅ All builds successful:
```bash
cargo check -p observability-narration-core  # ✅ Pass (with expected deprecation warnings)
cargo check -p xtask                          # ✅ Pass
cargo build --bin rbee-keeper                 # ✅ Pass
```

## Next Steps

**For developers:**
- Migrate from `NARRATE.action().human().emit()` to `n!()` macro
- Compiler will warn about deprecated usage with helpful migration messages
- See examples in this document for migration patterns

**Deprecation timeline:**
- Current: Deprecated with warnings
- v0.6.0: Consider removing (after migration period)

## Historical Context

- **TEAM-297:** Introduced `n!()` macro for concise narration
- **TEAM-299:** Removed global stderr for privacy (SSE-only in multi-tenant)
- **TEAM-309:** Fixed CLI narration visibility by adding tracing subscriber
- **TEAM-310:** Centralized formatting in `format.rs` module
- **TEAM-310 (this fix):** Deprecated old pipeline, unified formatting

## Key Insight

The confusion was that the old builder pipeline (`Narration::new().emit()`) was still active and NOT deprecated, while the new `n!()` macro was recommended but not enforced. Now both pipelines are clearly marked:

- Old pipeline: **DEPRECATED** (compiler warnings)
- New pipeline: **RECOMMENDED** (clean, concise)

Both now use the same formatting function for consistency.

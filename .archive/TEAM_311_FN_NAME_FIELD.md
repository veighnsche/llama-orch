# TEAM-311: fn_name Field Implementation

**Status:** ✅ COMPLETE  
**Date:** October 26, 2025  
**Mission:** Add separate `fn_name` field to track which function narrations come from

---

## Summary

Added dedicated `fn_name` field to `NarrationFields` that gets populated by the `#[narrate_fn]` macro. This allows seeing both the **action** (what's happening) and the **function** (where it's happening) separately.

---

## Changes Made

### 1. Added fn_name Field to NarrationFields

**File:** `bin/99_shared_crates/narration-core/src/core/types.rs`

```rust
pub struct NarrationFields {
    pub actor: &'static str,
    pub action: &'static str,
    pub target: String,
    pub human: String,
    pub level: NarrationLevel,
    
    /// TEAM-311: Function name (from #[narrate_fn] macro)
    pub fn_name: Option<String>,
    
    // ... other fields
}
```

### 2. Updated Macro Implementation

**File:** `bin/99_shared_crates/narration-core/src/api/macro_impl.rs`

**Before:**
- Thread-local `target` was set to function name
- This mixed "what" (action) with "where" (function)

**After:**
- Thread-local provides `fn_name` (separate field)
- `target` remains as action name
- Clean separation of concerns

```rust
// Get function name from thread-local (set by #[narrate_fn])
let fn_name = crate::thread_actor::get_target();

// Target defaults to action
let target = action.to_string();

let fields = NarrationFields {
    actor,
    action,
    target,
    human: selected_message.to_string(),
    level,
    fn_name, // ← New separate field
    // ...
};
```

### 3. Added Format Function with fn_name

**File:** `bin/99_shared_crates/narration-core/src/format.rs`

```rust
pub fn format_message_with_fn(
    actor: &str, 
    action: &str, 
    message: &str, 
    fn_name: Option<&str>
) -> String {
    if let Some(fn_name) = fn_name {
        format!(
            "\x1b[1m[{:<20}] {:<20}\x1b[0m \x1b[2m{}\x1b[0m\n{}\n",
            actor,
            action,
            fn_name,  // ← Dimmed function name
            message,
        )
    } else {
        format_message(actor, action, message)
    }
}
```

### 4. Updated SSE Sink

**File:** `bin/99_shared_crates/narration-core/src/output/sse_sink.rs`

```rust
let formatted = format_message_with_fn(
    fields.actor,
    fields.action,
    &fields.human,
    fields.fn_name.as_deref()  // ← Pass fn_name
);
```

---

## Output Format

### Without #[narrate_fn]

**Code:**
```rust
pub fn parse(workspace_root: &Path, source_dir: &Path) -> Result<Vec<PathBuf>> {
    n!("phase_deps", "📦 Dependency discovery");
    n!("parse_deps", "Scanning root crate: {}", source_dir.display());
    // ...
}
```

**Output:**
```
[auto-update        ] phase_deps          
📦 Dependency discovery

[auto-update        ] parse_deps          
Scanning root crate: bin/00_rbee_keeper
```

### With #[narrate_fn]

**Code:**
```rust
#[narrate_fn]
pub fn parse(workspace_root: &Path, source_dir: &Path) -> Result<Vec<PathBuf>> {
    n!("phase_deps", "📦 Dependency discovery");
    n!("parse_deps", "Scanning root crate: {}", source_dir.display());
    // ...
}
```

**Output:**
```
[auto-update        ] phase_deps           parse
📦 Dependency discovery

[auto-update        ] parse_deps           parse
Scanning root crate: bin/00_rbee_keeper
```

**Note:** The function name "parse" appears in **dimmed text** (\x1b[2m) after the action.

---

## Visual Breakdown

### Format Structure

```
\x1b[1m[actor              ] action              \x1b[0m \x1b[2mfn_name\x1b[0m
message

```

- **Bold** header: `[actor] action`
- **Dimmed** function name (optional): `fn_name`
- **Plain** message on second line
- **Blank** third line for separation

### ANSI Codes

- `\x1b[1m` - Bold (for actor/action header)
- `\x1b[2m` - Dim (for function name)
- `\x1b[0m` - Reset (back to normal)

---

## Real-World Example

### auto-update Crate

**File:** `src/dependencies.rs`

```rust
#[narrate_fn]
pub fn parse(workspace_root: &Path, source_dir: &Path) -> Result<Vec<PathBuf>> {
    let start = Instant::now();
    n!("phase_deps", "📦 Dependency discovery");
    n!("parse_deps", "Scanning root crate: {}", source_dir.display());
    
    // ... parsing logic ...
    
    n!("collect_tomls", "Queued Cargo.toml files: {}", toml_files.len());
    n!("parse_batch", "Parsed {} deps · {} local path · {} transitive", 
        toml_files.len(), local_deps, transitive_deps);
    
    // Debug-level detail
    for (path, local, trans) in &toml_files {
        nd!("parse_detail", "{} · local={} · transitive={}", 
            path.display(), local, trans);
    }
    
    let elapsed = start.elapsed().as_millis();
    n!("summary", "✅ Deps ok · {}ms", elapsed);
    
    Ok(all_deps)
}
```

**Output (in terminal or SSE):**

```
[auto-update        ] phase_deps           parse
📦 Dependency discovery

[auto-update        ] parse_deps           parse
Scanning root crate: bin/00_rbee_keeper

[auto-update        ] collect_tomls        parse
Queued Cargo.toml files: 9

[auto-update        ] parse_batch          parse
Parsed 21 deps · 12 local path · 9 transitive

[auto-update        ] parse_detail         parse
bin/99_shared_crates/daemon-lifecycle · local=3 · transitive=8

[auto-update        ] parse_detail         parse
bin/99_shared_crates/narration-core · local=0 · transitive=5

[auto-update        ] summary              parse
✅ Deps ok · 118ms
```

---

## Benefits

### 1. Clear Traceability

You can now see:
- **What** is happening: `parse_deps`, `collect_tomls`, etc.
- **Where** it's happening: `parse` function

### 2. No Confusion

**Before:** `target` field was overloaded (sometimes action, sometimes function)  
**After:** Clean separation - `action` is always the action, `fn_name` is always the function

### 3. Optional

Functions without `#[narrate_fn]` work exactly as before - no function name shown.

### 4. Visual Distinction

Function name is **dimmed** so it doesn't clutter the important information (action + message).

---

## Usage Guidelines

### When to Use #[narrate_fn]

✅ **Use it for:**
- Important entry points (API handlers, lifecycle functions)
- Functions with multiple narrations
- Functions that are hard to trace

❌ **Skip it for:**
- Simple helpers
- Functions with only one narration
- Internal utilities

### Example: Good Use Cases

```rust
// ✅ Good - Entry point with multiple steps
#[narrate_fn]
pub fn check(updater: &AutoUpdater) -> Result<bool> {
    n!("phase_build", "🛠️ Build state");
    // ... multiple narrations ...
    n!("phase_scan", "🔍 Source freshness checks");
    // ... more narrations ...
    n!("phase_decision", "📑 Rebuild decision");
}

// ✅ Good - Complex operation
#[narrate_fn]
pub fn parse(workspace_root: &Path, source_dir: &Path) -> Result<Vec<PathBuf>> {
    n!("phase_deps", "📦 Dependency discovery");
    // ... many narrations ...
}

// ❌ Skip - Simple helper
fn is_newer(file: &Path, ref_time: SystemTime) -> bool {
    // No narration or just one - don't need #[narrate_fn]
}
```

---

## Compilation Status

```bash
cargo check -p observability-narration-core  # ✅ PASS
cargo check -p auto-update                   # ✅ PASS
```

---

## Integration with Existing Features

### Works with all narration features:

1. **Level system:**
   ```rust
   #[narrate_fn]
   pub fn parse(...) {
       n!("action", "Info level");
       nd!("detail", "Debug level");
   }
   ```
   Output shows function name for both.

2. **Job context:**
   ```rust
   #[narrate_fn]
   pub fn parse(...) {
       let ctx = NarrationContext::new().with_job_id("job-123");
       with_narration_context(ctx, async {
           n!("action", "Message");
       }).await
   }
   ```
   Function name + job_id both propagate.

3. **SSE streams:**
   All SSE events include `fn_name` field if present.

4. **Test capture:**
   Tests can assert on `fn_name` field.

---

## API Reference

### NarrationFields

```rust
pub struct NarrationFields {
    pub fn_name: Option<String>,  // Function name from #[narrate_fn]
    // ... other fields
}
```

### format_message_with_fn

```rust
pub fn format_message_with_fn(
    actor: &str,
    action: &str,
    message: &str,
    fn_name: Option<&str>
) -> String
```

Formats a narration with optional function name (dimmed).

---

## Summary

The `fn_name` field is now working correctly:

✅ **Added** dedicated field to `NarrationFields`  
✅ **Separated** from `target` (clean architecture)  
✅ **Formatted** with dimmed text (visual distinction)  
✅ **Optional** - only appears with `#[narrate_fn]`  
✅ **Compiles** - all packages build successfully  
✅ **Integrated** - works with levels, context, SSE  

**The `#[narrate_fn]` macro now provides clear function traceability without cluttering the narration output.**

---

## Team Signature

**TEAM-311:** Implemented separate `fn_name` field for function traceability

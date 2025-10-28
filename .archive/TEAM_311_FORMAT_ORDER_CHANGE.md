# TEAM-311: Format Order Change - fn_name Before action

**Status:** ✅ COMPLETE  
**Date:** October 26, 2025  
**Requested by:** User  

---

## Summary

Changed the narration format order so that function name (`fn_name`) appears **before** action, and updated the styling so that actor and fn_name are **bold** while action is **light** (not bold).

---

## Format Change

### Before (Old Format)
```
[actor              ] action               fn_name (dimmed)
message
```

**Styling:**
- `[actor] action` - **BOLD**
- `fn_name` - *dimmed* (lighter gray)

**Example:**
```
[auto-update        ] phase_init           new
🚧 Initializing auto-updater
```

### After (New Format)
```
[actor              ] fn_name              action
message
```

**Styling:**
- `[actor] fn_name` - **BOLD**
- `action` - light (not bold)

**Example:**
```
[auto-update        ] new                  phase_init
🚧 Initializing auto-updater
```

---

## Rationale

User requested this change for better visual hierarchy:
- **Bold** elements (`actor` and `fn_name`) are the **WHERE** (which component, which function)
- **Light** element (`action`) is the **WHAT** (what's happening)

This makes it easier to scan for which function is executing.

---

## Files Changed

### 1. Core Formatting Function
**File:** `narration-core/src/format.rs:106-121`

**Changed:**
```rust
// OLD: Order was actor, action, fn_name
format!(
    "\x1b[1m[{:<width_actor$}] {:<width_action$}\x1b[0m \x1b[2m{}\x1b[0m\n{}\n",
    actor,
    action,
    fn_name,  // dimmed
    message,
    //...
)

// NEW: Order is actor, fn_name, action  
format!(
    "\x1b[1m[{:<width_actor$}] {:<width_action$}\x1b[0m {}\n{}\n",
    actor,
    fn_name,  // bold
    action,   // light
    message,
    //...
)
```

**Key Changes:**
- Swapped `action` and `fn_name` positions
- Changed fn_name from `\x1b[2m` (dimmed) to part of `\x1b[1m` (bold) section
- Action moved outside bold section (light)

### 2. Documentation Updates

**File:** `narration-core/src/format.rs:68-105`
- Updated doc comments to reflect new order
- Updated example output

**File:** `narration-core/src/core/types.rs:152-193`
- Updated `NarrationFields::format()` doc comments
- Updated example showing new order

### 3. Test Fixes

**File:** `narration-core/tests/format_consistency.rs`

**Changes:**
- Line 33-40: Updated assertions for new format (without fn_name case)
- Line 67-73: Updated padding test assertions
- Changed from `starts_with` to `contains` (ANSI codes at start)

**Tests Status:** ✅ All passing

---

## Visual Comparison

### Without fn_name (Same as Before)
```
[rbee-keeper        ] queen_status        
✅ Queen is running
```

No change - when there's no fn_name, format is unchanged.

### With fn_name (NEW ORDER)

**Before:**
```
[auto-update        ] phase_init           new
```
- Bold: `[auto-update] phase_init`
- Dimmed: `new`

**After:**
```
[auto-update        ] new                  phase_init
```
- Bold: `[auto-update] new`
- Light: `phase_init`

---

## Width Constants

Both fields use the same width constants:
- `ACTOR_WIDTH = 20` chars
- `ACTION_WIDTH = 20` chars (used for fn_name when present)

So the format is:
```
[actor (20 chars)] fn_name (20 chars) action (no padding)
message
```

---

## Verification

### Compilation
```bash
cargo check -p observability-narration-core
```
**Result:** ✅ PASS

### Tests
```bash
cargo test -p observability-narration-core --test format_consistency
```
**Result:** ✅ 2 passed

### Integration
```bash
cargo build --bin rbee-keeper
```
**Result:** ✅ PASS (with expected deprecation warnings)

---

## Expected Output Examples

### Auto-Update (with #[narrate_fn])
```
[auto-update        ] new                  phase_init
🚧 Initializing auto-updater for rbee-keeper

[auto-update        ] parse                phase_deps
📦 Dependency discovery

[auto-update        ] check                phase_build
🛠️ Build state
```

### Queen-Lifecycle (after migration)
```
[rbee-keeper        ] handle_self_check    narrate_test_start
Starting rbee-keeper narration test
```

### Without fn_name
```
[rbee-keeper        ] queen_status        
✅ Queen is running on http://localhost:8500
```

---

## Benefits

1. **Better Visual Hierarchy**
   - Bold = WHERE (actor, function)
   - Light = WHAT (action)

2. **Easier Scanning**
   - Function names stand out in bold
   - Easy to see which function is executing

3. **Consistent**
   - Both "where" indicators (actor, fn_name) use same styling

4. **Logical Order**
   - [component] function action
   - Reads naturally: "auto-update's new does phase_init"

---

## Migration Notes

- **No breaking changes** - This is a display format change only
- **API unchanged** - `NarrationFields::format()` still works the same
- **Backward compatible** - Without fn_name, format is unchanged
- **Tests updated** - All assertions now match new format

---

## Team Signature

**TEAM-311:** Format order changed to [actor] fn_name action with bold styling

User request implemented - function names now appear before actions in bold! 🎯

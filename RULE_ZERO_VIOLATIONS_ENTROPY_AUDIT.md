# Rule Zero Violations - Entropy Audit

**Date:** 2025-10-28  
**Auditor:** TEAM-337

---

## What is Entropy?

**Entropy** = Functions/variables with names like `_v2`, `_new`, `_with_X`, `_and_Y` that indicate:
- Someone was too scared to update the original function
- Created a new function to avoid "breaking changes"
- Left both functions in the codebase
- Now we have 2+ ways to do the same thing

**Rule Zero:** Update existing functions. Let the compiler find call sites. Fix them. Delete old code.

---

## Entropy Found in Codebase

### 1. `narrate_at_level()` + 6 wrapper functions 🔴 CRITICAL ENTROPY

**File:** `bin/99_shared_crates/narration-core/src/api/emit.rs:60`

```rust
pub fn narrate_at_level(fields: NarrationFields, level: NarrationLevel)  // ← Should just be narrate()
pub fn narrate(fields: NarrationFields)          // ← Wrapper #1 (calls narrate_at_level with Info)
pub fn narrate_warn(fields: NarrationFields)     // ← Wrapper #2 (calls narrate_at_level with Warn)
pub fn narrate_error(fields: NarrationFields)    // ← Wrapper #3 (calls narrate_at_level with Error)
pub fn narrate_fatal(fields: NarrationFields)    // ← Wrapper #4 (calls narrate_at_level with Fatal)
pub fn narrate_debug(fields: NarrationFields)    // ← Wrapper #5 (calls narrate_at_level with Debug)
pub fn narrate_trace(fields: NarrationFields)    // ← Wrapper #6 (calls narrate_at_level with Trace)
```

**Evidence of Entropy:**
- Created `narrate_at_level()` instead of adding parameter to `narrate()`
- The `_at_level` suffix is pure entropy
- Then created 6 wrapper functions that all call `narrate_at_level()`
- Now have 7 functions doing the same thing

**What Should Have Happened (Rule Zero):**
```rust
// BEFORE:
pub fn narrate(fields: NarrationFields)

// AFTER (Rule Zero):
pub fn narrate(fields: NarrationFields, level: NarrationLevel)
//                                      ^^^^^^^^^^^^^^^^^^^^^^ Just add parameter

// Update all call sites:
narrate(fields, NarrationLevel::Info)   // Default
narrate(fields, NarrationLevel::Warn)   // Warning
narrate(fields, NarrationLevel::Error)  // Error

// Compiler finds all call sites
// Fix them
// Done. Clean API.
```

**What Actually Happened (Entropy):**
```rust
// Created NEW function with ugly name
pub fn narrate_at_level(fields: NarrationFields, level: NarrationLevel)
//            ^^^^^^^^^ ENTROPY SUFFIX

// Then created 6 wrapper functions!
pub fn narrate(fields: NarrationFields) {
    narrate_at_level(fields, NarrationLevel::Info)  // ← Wrapper
}

pub fn narrate_warn(fields: NarrationFields) {
    narrate_at_level(fields, NarrationLevel::Warn)  // ← Wrapper
}

// ... 4 more wrappers
```

**Impact:**
- 7 functions instead of 1
- Every wrapper adds maintenance burden
- Confusing API: "Should I use narrate() or narrate_at_level()?"
- Permanent technical debt
- **This is the WORST entropy violation in the codebase**

**Fix:**
```rust
// 1. Rename narrate_at_level → narrate
// 2. Delete all 6 wrapper functions
// 3. Update all call sites to pass level parameter
// 4. Compiler finds them all
```

---

### 2. ~~`format_message_with_fn()`~~ ✅ FIXED BY USER

**File:** `bin/99_shared_crates/narration-core/src/format.rs:75`

**Status:** ✅ User renamed it to `format_message()` - ENTROPY REMOVED!

---

### 3. ~~`macro_emit_auto_with_fn()`~~ ✅ FIXED BY USER

**File:** `bin/99_shared_crates/narration-core/src/api/macro_impl.rs:62`

**Status:** ✅ User renamed it to `macro_emit()` - ENTROPY REMOVED!

---

### 4. `emit_with_provenance()` ⚠️ DEPRECATED (Skip)

**File:** `bin/99_shared_crates/narration-core/src/api/builder.rs:466`

```rust
pub fn emit_with_provenance(mut self, crate_name: &str, crate_version: &str)
```

**Evidence of Entropy:**
- Name implies there's an `emit_without_provenance`
- The `_with_X` suffix is a code smell

**What Should Have Happened:**
```rust
// Just make emit() take provenance parameters
pub fn emit(self, crate_name: &str, crate_version: &str)
```

**Current State:**
- Marked `#[doc(hidden)]` - internal use only
- Used by macros
- Still entropy in the API

---

### 3. `macro_emit_auto_with_fn()` ❌ ENTROPY

**File:** `bin/99_shared_crates/narration-core/src/api/macro_impl.rs:62`

```rust
pub fn macro_emit_auto_with_fn(
    action: &'static str,
    human: &str,
    cute: Option<&str>,
    ...
)
```

**Evidence of Entropy:**
- `macro_emit_auto_with_fn` - THREE suffixes!
- `macro_` prefix (okay, it's for macros)
- `_auto` suffix (what's non-auto?)
- `_with_fn` suffix (ENTROPY)

**What This Implies:**
- There's a `macro_emit_auto_without_fn`?
- There's a `macro_emit_manual_with_fn`?
- There's a `macro_emit_manual_without_fn`?

**Naming Explosion:**
```
macro_emit_auto_with_fn
macro_emit_auto_without_fn  ← Does this exist?
macro_emit_manual_with_fn   ← Does this exist?
macro_emit_manual_without_fn ← Does this exist?
```

**This is what happens when you avoid Rule Zero!**

---

### 4. Test Functions with `_with_` ⚠️ ACCEPTABLE

**Files:** Multiple test files

```rust
fn test_parse_with_skipped_tests()
fn test_parse_with_unicode()
fn test_hive_start_with_alias()
```

**Verdict:** ✅ ACCEPTABLE
- Test names are descriptive
- Not part of public API
- `_with_` describes test scenario, not API variation
- This is fine

---

## Summary of Violations

| Function | File | Severity | Status |
|----------|------|----------|--------|
| `narrate_at_level` + 6 wrappers | narration-core/src/api/emit.rs | 🔴 CRITICAL | ❌ TODO |
| ~~`format_message_with_fn`~~ | narration-core/src/format.rs | 🔴 HIGH | ✅ FIXED |
| ~~`macro_emit_auto_with_fn`~~ | narration-core/src/api/macro_impl.rs | 🟡 MEDIUM | ✅ FIXED |
| `emit_with_provenance` | narration-core/src/api/builder.rs | 🟡 MEDIUM | ⚠️ DEPRECATED (Skip) |

---

## The Pattern

### How Entropy Happens

1. Developer needs to add parameter to function
2. Developer thinks: "This might break things"
3. Developer creates `function_with_new_param()`
4. Old function stays (or gets deleted later)
5. **Entropy created:** Ugly name is now permanent

### What Rule Zero Says

1. Developer needs to add parameter to function
2. Developer adds parameter to existing function
3. Compiler finds all call sites (30 seconds)
4. Developer fixes call sites (5 minutes)
5. **No entropy:** Clean name preserved

### Time Comparison

**Entropy approach:**
- Create new function: 2 minutes
- Update some call sites: 10 minutes
- Leave others broken: Forever
- **Result:** Technical debt forever

**Rule Zero approach:**
- Update function signature: 30 seconds
- Compiler finds call sites: Instant
- Fix all call sites: 5 minutes
- **Result:** Clean codebase

---

## Recommended Actions

### Critical Priority (Remaining)

1. **Fix `narrate_at_level()` entropy** 🔴
   
   **Step 1:** Rename `narrate_at_level` → `narrate`
   ```rust
   // In narration-core/src/api/emit.rs
   // Change line 60:
   pub fn narrate(fields: NarrationFields, level: NarrationLevel)
   //     ^^^^^^^ Remove _at_level suffix
   ```
   
   **Step 2:** Delete all 6 wrapper functions
   ```rust
   // DELETE these functions (lines 138-167):
   // pub fn narrate(fields: NarrationFields)
   // pub fn narrate_warn(fields: NarrationFields)
   // pub fn narrate_error(fields: NarrationFields)
   // pub fn narrate_fatal(fields: NarrationFields)
   // pub fn narrate_debug(fields: NarrationFields)
   // pub fn narrate_trace(fields: NarrationFields)
   ```
   
   **Step 3:** Update all call sites
   ```bash
   # Find all uses:
   rg "narrate\(" --type rust
   rg "narrate_warn\(" --type rust
   rg "narrate_error\(" --type rust
   # etc.
   
   # Update each to:
   narrate(fields, NarrationLevel::Info)
   narrate(fields, NarrationLevel::Warn)
   narrate(fields, NarrationLevel::Error)
   ```
   
   **Step 4:** Compiler finds any missed call sites
   ```bash
   cargo check
   # Fix any compilation errors
   ```

### Completed ✅

2. ~~**Rename `format_message_with_fn` → `format_message`**~~ ✅ DONE BY USER
3. ~~**Rename `macro_emit_auto_with_fn` → `macro_emit`**~~ ✅ DONE BY USER

---

## Lessons Learned

### For Future Teams

1. **When you see `_with_`, `_and_`, `_v2`, `_new` in function names:**
   - This is entropy
   - Someone avoided Rule Zero
   - Fix it now before it spreads

2. **When you need to add a parameter:**
   - Update the existing function
   - Let compiler find call sites
   - Fix them
   - Don't create `function_with_param()`

3. **When you see "deprecated" comments:**
   - Check if new function has entropy name
   - If yes, rename it to the old name
   - Update call sites
   - Delete old function

---

## Cost of Entropy

### `format_message_with_fn` Example

**Every time someone uses this function:**
```rust
// What they have to type:
format_message_with_fn(action, msg, fn_name)
//            ^^^^^^^^ 8 extra characters

// What they should type:
format_message(action, msg, fn_name)
//            ^ Clean, simple
```

**Across codebase:**
- Used in: 10+ files
- Each use: 8 extra characters
- Mental overhead: "Why _with_fn? Is there one without?"
- **Total cost:** Permanent confusion and verbosity

---

## Conclusion

**Found:** 3 major entropy violations  
**Impact:** Ugly function names, confused developers, technical debt  
**Fix:** Rename functions, update call sites (compiler helps)  
**Time:** 30 minutes to fix all 3  
**Benefit:** Clean codebase forever

**Rule Zero exists for a reason. Use it.**

---

**TEAM-337** - Entropy Audit Complete

# TEAM-312: Automatic Function Name Capture Proposal

**Status:** 📋 PROPOSAL  
**Date:** Oct 26, 2025  
**Question:** Can we automatically capture function names without `#[narrate_fn]`?

---

## Answer: YES ✅

There are **3 approaches** to automatically capture function names:

---

## Approach 1: Enhanced n!() Macro with `function!()` (RECOMMENDED)

Add a new macro that uses Rust's built-in `function!()` macro to capture the function name automatically.

### Implementation

```rust
/// Narration with automatic function name capture
///
/// TEAM-312: Uses Rust's function!() macro to capture function name automatically.
/// No #[narrate_fn] attribute needed!
///
/// # Example
/// ```rust,ignore
/// fn my_function() {
///     nf!("action", "message");  // Automatically includes "my_function"
/// }
/// ```
#[macro_export]
macro_rules! nf {
    // Simple: nf!("action", "message")
    ($action:expr, $msg:expr) => {{
        $crate::macro_emit_auto_with_fn($action, $msg, None, None, env!("CARGO_CRATE_NAME"), function!());
    }};
    
    // With format: nf!("action", "msg {}", arg)
    ($action:expr, $fmt:expr, $($arg:expr),+ $(,)?) => {{
        $crate::macro_emit_auto_with_fn($action, &format!($fmt, $($arg),+), None, None, env!("CARGO_CRATE_NAME"), function!());
    }};
}

// Add to macro_impl.rs:
#[doc(hidden)]
pub fn macro_emit_auto_with_fn(
    action: &'static str,
    human: &str,
    cute: Option<&str>,
    story: Option<&str>,
    crate_name: &'static str,
    fn_name: &'static str,
) {
    // ... same as macro_emit_auto but with fn_name parameter
    let fields = NarrationFields {
        actor: crate_name,
        action,
        target: action.to_string(),
        human: human.to_string(),
        fn_name: Some(fn_name.to_string()), // ← Automatically captured!
        // ... rest of fields
    };
    
    crate::narrate_at_level(fields, NarrationLevel::Info);
}
```

### Usage

```rust
fn execute_hive_start() -> Result<()> {
    nf!("start", "Starting hive");  // Automatically includes "execute_hive_start"
    // ...
}
```

### Pros
- ✅ No proc macro needed
- ✅ No #[narrate_fn] attribute needed
- ✅ Works everywhere (sync/async)
- ✅ Zero runtime overhead
- ✅ Simple implementation

### Cons
- ❌ Function path includes module (e.g., "my_crate::module::function")
- ❌ Need to use `nf!()` instead of `n!()` to opt-in

---

## Approach 2: Make n!() ALWAYS Capture Function Name (BREAKING CHANGE)

Modify the existing `n!()` macro to always capture function names.

### Implementation

```rust
#[macro_export]
macro_rules! n {
    ($action:expr, $msg:expr) => {{
        $crate::macro_emit_auto_with_fn($action, $msg, None, None, env!("CARGO_CRATE_NAME"), function!());
    }};
    // ... all other variants updated similarly
}
```

### Pros
- ✅ No new macro needed
- ✅ Works everywhere automatically
- ✅ Consistent behavior

### Cons
- ❌ BREAKING CHANGE - all existing n!() calls now include function names
- ❌ May be too verbose for simple cases
- ❌ Can't opt-out if you don't want function names

---

## Approach 3: Global Configuration Flag (FLEXIBLE)

Add a feature flag or environment variable to control automatic function name capture.

### Implementation

```rust
// In Cargo.toml:
[features]
auto-fn-names = []

// In macro:
#[macro_export]
macro_rules! n {
    ($action:expr, $msg:expr) => {{
        #[cfg(feature = "auto-fn-names")]
        {
            $crate::macro_emit_auto_with_fn($action, $msg, None, None, env!("CARGO_CRATE_NAME"), function!());
        }
        #[cfg(not(feature = "auto-fn-names"))]
        {
            $crate::macro_emit_auto($action, $msg, None, None, env!("CARGO_CRATE_NAME"));
        }
    }};
}
```

### Pros
- ✅ Opt-in per crate
- ✅ No breaking changes
- ✅ Flexible

### Cons
- ❌ More complex implementation
- ❌ Feature flag must be consistent across workspace
- ❌ Can't mix behaviors in same crate

---

## Comparison with Current #[narrate_fn]

### Current System (TEAM-311)
```rust
#[narrate_fn]
fn execute_hive_start() -> Result<()> {
    n!("start", "Starting hive");  // fn_name = "execute_hive_start"
}
```

**How it works:**
- Proc macro wraps function body
- Sets thread-local `fn_name` on entry
- Clears thread-local on exit
- All `n!()` calls inside automatically get `fn_name`

**Pros:**
- ✅ Explicit opt-in (clear intent)
- ✅ Works with nested functions
- ✅ Clean function name (no module path)

**Cons:**
- ❌ Requires attribute on every function
- ❌ Easy to forget
- ❌ Proc macro overhead (compile time)

---

## Recommendation: Approach 1 (nf! macro)

**Why:**
1. **No breaking changes** - `n!()` stays the same
2. **Opt-in** - Use `nf!()` when you want function names
3. **Simple** - No proc macro, just a macro variant
4. **Fast** - Zero runtime overhead
5. **Flexible** - Can mix `n!()` and `nf!()` in same crate

**Migration Path:**
```rust
// Before (requires #[narrate_fn]):
#[narrate_fn]
fn my_function() {
    n!("action", "message");
}

// After (no attribute needed):
fn my_function() {
    nf!("action", "message");  // Automatically captures "my_function"
}
```

**Function Name Format:**
- `function!()` returns: `"my_crate::module::my_function"`
- We can strip the module path if desired:
  ```rust
  let fn_name = function!().rsplit("::").next().unwrap_or(function!());
  ```

---

## Implementation Plan

### Phase 1: Add nf!() macro (2 hours)
1. Add `macro_emit_auto_with_fn()` to `macro_impl.rs`
2. Add `nf!()` macro to `lib.rs`
3. Add tests
4. Update documentation

### Phase 2: Migrate timeout-enforcer (1 hour)
```rust
// Before:
n!("start", "⏱️  {} (timeout: {}s)", label, total_secs);

// After:
nf!("start", "⏱️  {} (timeout: {}s)", label, total_secs);
```

### Phase 3: Document pattern (30 min)
- Update README.md
- Add migration guide
- Add examples

**Total effort:** ~3.5 hours

---

## Alternative: Keep #[narrate_fn] (CURRENT SYSTEM)

If we want to keep the current system, we can make it easier to use:

### Option A: Workspace-wide proc macro
Create a workspace-level proc macro that automatically adds `#[narrate_fn]` to all functions:

```rust
#[auto_narrate]  // ← One attribute at module level
mod my_module {
    fn function1() { ... }  // ← Automatically gets #[narrate_fn]
    fn function2() { ... }  // ← Automatically gets #[narrate_fn]
}
```

### Option B: Compiler plugin (UNSTABLE)
Use unstable Rust features to inject function names at compile time.

**Not recommended:** Requires nightly Rust, unstable API.

---

## Decision Matrix

| Approach | Ease of Use | Performance | Breaking Change | Implementation |
|----------|-------------|-------------|-----------------|----------------|
| **nf!() macro** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ❌ No | ⭐⭐⭐⭐⭐ Easy |
| n!() always captures | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⚠️ Yes | ⭐⭐⭐⭐ Easy |
| Feature flag | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ❌ No | ⭐⭐⭐ Medium |
| #[narrate_fn] (current) | ⭐⭐ | ⭐⭐⭐⭐ | ❌ No | ✅ Done |
| #[auto_narrate] module | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ❌ No | ⭐⭐ Hard |

---

## Recommendation

**Implement Approach 1: `nf!()` macro**

This gives us:
- ✅ Automatic function name capture
- ✅ No breaking changes
- ✅ Simple implementation
- ✅ Zero runtime overhead
- ✅ Opt-in (use when needed)

**Next Steps:**
1. Implement `nf!()` macro
2. Test with timeout-enforcer
3. Document pattern
4. Gradually migrate codebase

---

**Made with 💝 by TEAM-312**

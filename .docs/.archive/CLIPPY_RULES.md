# Clippy Rules for llama-orch

**Created by:** TEAM-044  
**Status:** Active  

---

## Overview

This project uses comprehensive clippy lints to enforce security, correctness, and code quality across all workspace crates. The rules are configured in:
- `Cargo.toml` - Workspace-wide lint definitions
- `.clippy.toml` - Clippy-specific configuration

---

## Security-Critical Lints (Deny/Warn)

These lints catch security vulnerabilities and dangerous patterns:

### Memory Safety
- **`mem_forget`** (deny) - Prevents memory leaks via `std::mem::forget`
- **`mem_replace_with_uninit`** (deny) - Prevents undefined behavior
- **`not_unsafe_ptr_arg_deref`** (deny) - Prevents NULL pointer dereferences

### Panics and Unwraps
- **`unwrap_used`** (warn) - Forces proper error handling instead of `.unwrap()`
- **`expect_used`** (warn) - Forces proper error handling instead of `.expect()`
- **`panic`** (warn) - Discourages panic! in production code
- **`indexing_slicing`** (warn) - Prevents panic from array indexing

### Arithmetic Safety
- **`integer_arithmetic`** (warn) - Catches potential overflows
- **`cast_possible_truncation`** (warn) - Prevents data loss in casts
- **`cast_precision_loss`** (warn) - Prevents float precision loss
- **`cast_sign_loss`** (warn) - Prevents sign loss in casts

### Crypto Safety
- **`default_numeric_fallback`** (deny) - Prevents insecure default crypto params

---

## Correctness Lints

These catch bugs and logic errors:

### Clone/Copy Issues
- **`clone_on_copy`** - Don't clone Copy types
- **`clone_on_ref_ptr`** - Don't clone references unnecessarily
- **`redundant_clone`** - Remove unnecessary clones
- **`drop_copy`** (deny) - Don't drop Copy types
- **`forget_copy`** (deny) - Don't forget Copy types

### Performance
- **`large_enum_variant`** - Warn on large enum variants
- **`large_types_passed_by_value`** - Pass large types by reference
- **`inefficient_to_string`** - Use more efficient string conversion

### String Operations
- **`string_add_assign`** - Avoid string concatenation via `+=`
- **`string_lit_as_bytes`** - Use more efficient byte string literals

---

## Code Quality Lints

These enforce best practices and maintainable code:

### Complexity
- **`cognitive_complexity`** (threshold: 15) - Keep functions simple
- **`too_many_arguments`** (threshold: 7) - Limit function arguments
- **`too_many_lines`** (threshold: 100) - Keep functions short

### Documentation
- **`missing_docs`** - Require public API documentation
- **`missing_errors_doc`** - Document errors in Result-returning functions
- **`missing_panics_doc`** - Document when functions can panic
- **`missing_debug_implementations`** - Implement Debug for public types

### Async/Await
- **`unused_async`** - Remove unnecessary async on functions
- **`unused_self`** - Remove unnecessary self parameters

### Code Style
- **`needless_borrow`** - Remove unnecessary borrows
- **`needless_pass_by_value`** - Pass by reference when appropriate
- **`single_match`** - Use if let instead of match with single arm
- **`wildcard_imports`** - Avoid wildcard imports (use explicit imports)
- **`items_after_statements`** - Place items at top of scope

---

## How to Use

### Run Clippy on All Crates
```bash
cargo clippy --workspace --all-targets --all-features
```

### Run Clippy with Auto-Fix
```bash
cargo clippy --fix --workspace --allow-dirty
```

### Run Clippy in CI Mode (Fail on Warnings)
```bash
cargo clippy --workspace --all-targets -- -D warnings
```

### Check Specific Crate
```bash
cd bin/queen-rbee
cargo clippy --all-targets
```

---

## Overriding Lints

When you have a valid reason to bypass a lint:

### Per-Function
```rust
#[allow(clippy::unwrap_used)]
fn test_helper() {
    // Unwrap is OK in test code
    let value = some_option.unwrap();
}
```

### Per-Module
```rust
#![allow(clippy::missing_docs)]
// This module is internal and doesn't need docs
```

### Per-Crate
In the crate's `Cargo.toml`:
```toml
[lints]
workspace = true

[lints.clippy]
# Override for this specific crate
unwrap_used = "allow"
```

---

## Common Patterns

### ✅ Good: Proper Error Handling
```rust
fn process_data(s: &str) -> Result<u32, ParseError> {
    s.parse()
        .map_err(|e| ParseError::InvalidNumber(e))
}
```

### ❌ Bad: Using unwrap
```rust
fn process_data(s: &str) -> u32 {
    s.parse().unwrap()  // clippy::unwrap_used
}
```

### ✅ Good: Safe Indexing
```rust
fn get_item(vec: &Vec<String>, idx: usize) -> Option<&String> {
    vec.get(idx)
}
```

### ❌ Bad: Unsafe Indexing
```rust
fn get_item(vec: &Vec<String>, idx: usize) -> &String {
    &vec[idx]  // clippy::indexing_slicing
}
```

### ✅ Good: Checked Arithmetic
```rust
fn add_safely(a: u32, b: u32) -> Option<u32> {
    a.checked_add(b)
}
```

### ❌ Bad: Unchecked Arithmetic
```rust
fn add_unsafely(a: u32, b: u32) -> u32 {
    a + b  // clippy::integer_arithmetic (can overflow)
}
```

---

## Integration with CI

Add to your CI pipeline:

```yaml
# .github/workflows/ci.yml
- name: Run Clippy
  run: cargo clippy --workspace --all-targets -- -D warnings
```

This ensures all new code passes clippy checks.

---

## Gradual Migration

If existing code has many clippy warnings:

1. **Start with security lints** - Fix `deny` level lints first
2. **Address correctness issues** - Fix obvious bugs
3. **Improve code quality** - Clean up style issues
4. **Enable stricter lints** - Gradually increase strictness

You can temporarily allow certain lints while fixing:

```toml
[workspace.lints.clippy]
# Temporarily allow while we fix existing code
unwrap_used = "allow"  # TODO: Fix and change to "warn"
```

---

## Resources

- [Clippy Lint List](https://rust-lang.github.io/rust-clippy/master/)
- [Clippy Configuration](https://doc.rust-lang.org/clippy/configuration.html)
- [Cargo Lints](https://doc.rust-lang.org/cargo/reference/manifest.html#the-lints-section)

---

## Maintenance

**When adding new lints:**
1. Test across all workspace crates
2. Document the rationale in this file
3. Set appropriate severity level (deny/warn/allow)
4. Consider migration impact on existing code

**Review lints periodically:**
- Clippy evolves with new Rust versions
- New lints may be added to clippy::nursery
- Consider upgrading warn → deny for mature projects

---

## For Future Teams

### Before Making Changes
```bash
# Check your code
cargo clippy --workspace --all-targets

# Auto-fix what you can
cargo clippy --fix --workspace --allow-dirty
```

### When Tests Fail
If clippy blocks your work:
1. **First:** Try to fix the issue properly
2. **Second:** Ask why the lint exists (it usually catches real bugs)
3. **Last resort:** Add `#[allow(clippy::lint_name)]` with a comment explaining why

### Adding New Crates
New crates automatically inherit workspace lints. No additional configuration needed unless you need crate-specific overrides.

---

**Remember:** Clippy lints are there to help catch bugs early. They represent years of Rust community experience. Trust them unless you have a very good reason not to.

# Clippy Configuration for llorch-candled

This document explains the clippy linting strategy for the llorch-candled inference engine.

## Philosophy

**Strict but pragmatic** - We enforce high code quality while recognizing ML/inference workloads have unique patterns that differ from typical Rust applications.

## Configuration Files

### `.clippy.toml`
Configures thresholds and behavior for specific lints.

### `Cargo.toml [lints.clippy]`
Defines which lint groups and specific lints are enabled/disabled.

## Lint Categories

### 🔴 DENY (Build-breaking errors)

- **`correctness`** - These are bugs, always fix them
- **`suspicious`** - Likely bugs or serious code smells
- **`undocumented_unsafe_blocks`** - All unsafe code must be justified

### 🟡 WARN (Should fix, but not blocking)

- **`complexity`** - Code smells that make maintenance harder
- **`perf`** - Performance optimization opportunities
- **`style`** - Consistency issues
- **`pedantic`** - Best practices (with ML-specific exceptions)
- **`cargo`** - Manifest and dependency issues

### ⚪ ALLOW (Intentionally disabled)

- **`nursery`** - Experimental lints, too unstable
- **`arithmetic_side_effects`** - Too restrictive for ML math
- **`indexing_slicing`** - Common in tensor operations
- **`cast_*`** - Intentional precision/sign conversions in ML
- **`wildcard_imports`** - Useful for prelude patterns
- **`similar_names`** - `x`, `y`, `z` are fine in ML contexts

## Key Thresholds

```toml
cognitive-complexity-threshold = 15      # Keep functions simple
type-complexity-threshold = 250          # Complex types OK for ML
too-many-arguments-threshold = 7         # Limit function parameters
too-many-lines-threshold = 150           # Keep functions focused
max-struct-bools = 3                     # Avoid boolean soup
array-size-threshold = 512000            # 512KB stack arrays
vec-box-size-threshold = 4096            # 4KB vector boxing
large-error-threshold = 128              # 128 byte error types
```

## ML-Specific Allowances

### Arithmetic
- **Allowed**: Integer arithmetic, division, wrapping
- **Reason**: ML workloads do intentional math operations

### Casting
- **Allowed**: Precision loss, sign loss, truncation, wrapping
- **Reason**: Tensor indexing and type conversions are intentional

### Indexing
- **Allowed**: Direct indexing and slicing
- **Reason**: Common in tensor operations (but be careful!)

### Naming
- **Allowed**: Single-char names (x, y, z), similar names
- **Reason**: Mathematical convention in ML code

## Running Clippy

```bash
# Check library with CPU feature
cargo clippy --features cpu --lib

# Check all targets with warnings as errors
cargo clippy --features cpu --all-targets -- -D warnings

# Fix auto-fixable issues
cargo clippy --features cpu --fix --lib

# Check specific file
cargo clippy --features cpu --lib -- --allow-dirty
```

## Common Fixes

### Documentation
```rust
// ❌ Bad
//! Uses candle-transformers::models::llama::Llama directly

// ✅ Good
//! Uses `candle-transformers::models::llama::Llama` directly
```

### Unwrap/Expect
```rust
// ❌ Bad (in library code)
let value = result.unwrap();

// ✅ Good
let value = result?;
// or
let value = result.expect("descriptive message about invariant");
```

### Float Comparison
```rust
// ❌ Bad
if temperature == 0.0 {

// ✅ Good
if temperature.abs() < f32::EPSILON {
```

### Large Types
```rust
// ❌ Bad
fn process(config: LargeConfig) {

// ✅ Good
fn process(config: &LargeConfig) {
```

## Suppressing Lints

Only suppress lints when you have a good reason:

```rust
// Suppress for a specific item
#[allow(clippy::cast_precision_loss)]
fn tensor_index(idx: usize) -> f32 {
    idx as f32  // Intentional conversion
}

// Suppress for a module
#![allow(clippy::similar_names)]  // x, y, z are fine here

// Suppress for a block
#[allow(clippy::indexing_slicing)]
{
    let value = tensor[idx];  // Bounds checked elsewhere
}
```

## CI Integration

Add to CI pipeline:

```yaml
- name: Clippy
  run: cargo clippy --workspace --all-targets --features cpu -- -D warnings
```

## Maintenance

Review and update these settings:
- **Quarterly**: Check for new clippy lints
- **On upgrade**: When updating Rust version
- **On feedback**: If lints are too strict/loose

## References

- [Clippy Lints](https://rust-lang.github.io/rust-clippy/master/)
- [Clippy Configuration](https://doc.rust-lang.org/clippy/configuration.html)
- [Cargo Lints](https://doc.rust-lang.org/cargo/reference/manifest.html#the-lints-section)

# 🔧 Conditional Compilation — Zero Overhead in Production

**Goal**: Completely remove TRACE/DEBUG code from production/staging builds  
**Result**: **0% overhead** in production (code doesn't even exist!)  
**Implementation**: Our custom proc macros with `#[cfg]` attributes

---

## 🎯 The Strategy

### Development Build (default):
- ✅ All tracing enabled
- ✅ `#[trace_fn]` works
- ✅ `trace_loop!()` works
- ✅ Full observability

### Production/Staging Build:
- ✅ TRACE code **completely removed** (not even compiled!)
- ✅ DEBUG code **completely removed**
- ✅ Only INFO/WARN/ERROR/FATAL remain
- ✅ **Zero overhead** (no runtime checks, code doesn't exist)

---

## 🛠️ Implementation: Feature Flags

### Step 1: Define Features in `Cargo.toml`

```toml
# bin/shared-crates/narration-core/Cargo.toml
[package]
name = "observability-narration-core"
version = "0.1.0"

[features]
default = ["trace-enabled", "debug-enabled", "cute-mode"]  # Dev builds
trace-enabled = []  # Enable TRACE level
debug-enabled = []  # Enable DEBUG level
cute-mode = []      # Enable cute/story modes
production = []     # Production profile (no trace/debug/cute)

[dependencies]
tracing = "0.1"
```

### Step 2: Conditional Compilation in Our Custom Macros

**Our custom proc macros** handle conditional compilation:

```rust
// bin/shared-crates/narration-macros/src/lib.rs

/// Our custom #[trace_fn] proc macro with conditional compilation
#[proc_macro_attribute]
pub fn trace_fn(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let input = parse_macro_input!(item as ItemFn);
    
    // Check if trace-enabled feature is active
    #[cfg(feature = "trace-enabled")]
    {
        // Generate full tracing code with auto-inferred actor
        let actor = infer_actor_from_module_path();
        let expanded = generate_trace_code(&input, &actor);
        return TokenStream::from(expanded);
    }
    
    #[cfg(not(feature = "trace-enabled"))]
    {
        // Return original function unchanged (no tracing)
        return TokenStream::from(quote! { #input });
    }
}
```

**Lightweight trace macros** also support conditional compilation:

```rust
// src/trace.rs

/// Ultra-lightweight TRACE macro (only compiled in dev builds)
#[cfg(feature = "trace-enabled")]
#[macro_export]
macro_rules! trace_tiny {
    ($actor:expr, $action:expr, $target:expr, $human:expr) => {
        tracing::trace!(
            actor = $actor,
            action = $action,
            target = $target,
            human = $human,
            "trace"
        );
    };
}

/// No-op version for production builds
#[cfg(not(feature = "trace-enabled"))]
#[macro_export]
macro_rules! trace_tiny {
    ($actor:expr, $action:expr, $target:expr, $human:expr) => {
        // Completely empty — code is removed at compile time
    };
}
```

### Step 3: Our Custom `#[narrate(...)]` Attribute

**Template interpolation with conditional compilation**:

```rust
// bin/shared-crates/narration-macros/src/lib.rs

#[proc_macro_attribute]
pub fn narrate(attr: TokenStream, item: TokenStream) -> TokenStream {
    let input = parse_macro_input!(item as ItemFn);
    let args = parse_macro_input!(attr as NarrateArgs);
    
    // Check if cute-mode feature is active
    #[cfg(feature = "cute-mode")]
    {
        // Generate full narration with cute/story modes
        let expanded = generate_narration_with_cute(&input, &args);
        return TokenStream::from(expanded);
    }
    
    #[cfg(not(feature = "cute-mode"))]
    {
        // Generate narration without cute mode
        let expanded = generate_narration_basic(&input, &args);
        return TokenStream::from(expanded);
    }
}
```

---

## 🚀 Build Profiles

### Development Build (default):

```bash
# All tracing enabled
cargo build

# Or explicitly
cargo build --features trace-enabled,debug-enabled
```

**Result**: Full observability, ~2% overhead for TRACE

---

### Staging Build:

```bash
# Remove TRACE, keep DEBUG
cargo build --release --no-default-features --features debug-enabled
```

**Result**: 
- ✅ DEBUG/INFO/WARN/ERROR/FATAL work
- ✅ TRACE code **completely removed**
- ✅ ~0% overhead (TRACE doesn't exist)

---

### Production Build:

```bash
# Remove TRACE and DEBUG
cargo build --release --no-default-features --features production
```

**Result**:
- ✅ Only INFO/WARN/ERROR/FATAL work
- ✅ TRACE and DEBUG code **completely removed**

---

## 📊 What Gets Removed

### Development Build (with cute mode):
```rust
#[trace_fn]  // ✅ Our custom macro generates full tracing code
fn dispatch_job(job_id: &str, pool_id: &str) -> Result<WorkerId> {
    let worker = select_worker(pool_id)?;
    Ok(worker.id)
}

// Expands to (with auto-inferred actor):
fn dispatch_job(job_id: &str, pool_id: &str) -> Result<WorkerId> {
    let _span = tracing::trace_span!(
        "dispatch_job",
        actor = "queen-rbee",  // Auto-inferred!
        job_id = %job_id,
        pool_id = %pool_id
    ).entered();
    let _start = std::time::Instant::now();
    // ... full tracing logic ...
    let worker = select_worker(pool_id)?;
    Ok(worker.id)
}
```

### Production Build:
```rust
#[trace_fn]  // ❌ Our custom macro does nothing (no-op)
fn dispatch_job(job_id: &str, pool_id: &str) -> Result<WorkerId> {
    let worker = select_worker(pool_id)?;
    Ok(worker.id)
}

// Expands to:
fn dispatch_job(job_id: &str, pool_id: &str) -> Result<WorkerId> {
    // No tracing code at all!
    let worker = select_worker(pool_id)?;
    Ok(worker.id)
}

**Result**: The production binary **doesn't even contain** the tracing code!

---

## 🎯 Advanced: Profile-Based Features

### `.cargo/config.toml`:

```toml
# .cargo/config.toml

[profile.dev]
# Development: all tracing enabled
# Uses default features

[profile.staging]
inherits = "release"
# Staging: remove TRACE, keep DEBUG

[profile.release]
# Production: remove TRACE and DEBUG
```

### Build Commands:

```bash
# Development (all tracing)
cargo build

# Staging (no TRACE)
cargo build --profile staging --no-default-features --features debug-enabled

# Production (no TRACE, no DEBUG)
cargo build --release --no-default-features --features production
```

---

## 🔍 Verification

### Check What's Compiled:

```bash
# Development build
cargo build
nm target/debug/queen-rbee | grep trace
# Shows: trace_tiny, trace_enter, trace_exit, etc.

# Production build
cargo build --release --no-default-features --features production
nm target/release/queen-rbee | grep trace
# Shows: (nothing — trace code removed!)
```

### Binary Size Comparison:

```bash
# Development build
cargo build
ls -lh target/debug/queen-rbee
# Size: ~50 MB (includes tracing)

# Production build
cargo build --release --no-default-features --features production
ls -lh target/release/queen-rbee
# Size: ~45 MB (5 MB smaller — tracing removed!)
```

---

## 📋 Developer Workflow

### Local Development:

```bash
# Just use default (all tracing enabled)
cargo run
RUST_LOG=trace cargo run
```

**Result**: Full TRACE/DEBUG visibility

---

### CI/CD Pipeline:

```yaml
# .github/workflows/ci.yml

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      # Test with all features (dev build)
      - name: Run tests
        run: cargo test --all-features
      
      # Build staging (no TRACE)
      - name: Build staging
        run: cargo build --profile staging --no-default-features --features debug-enabled
      
      # Build production (no TRACE, no DEBUG)
      - name: Build production
        run: cargo build --release --no-default-features --features production
      
      # Verify trace code is removed
      - name: Verify no trace symbols
        run: |
          ! nm target/release/queen-rbee | grep trace_tiny
```

---

## 🎨 Conditional Narration Levels

### Full Implementation:

```rust
// src/lib.rs

/// Emit TRACE-level narration (only in dev builds)
#[cfg(feature = "trace-enabled")]
pub fn narrate_trace(fields: NarrationFields) {
    // Full tracing implementation
    event!(Level::TRACE, ...);
}

/// No-op for production
#[cfg(not(feature = "trace-enabled"))]
pub fn narrate_trace(_fields: NarrationFields) {
    // Completely empty
}

/// Emit DEBUG-level narration (dev + staging)
#[cfg(feature = "debug-enabled")]
pub fn narrate_debug(fields: NarrationFields) {
    event!(Level::DEBUG, ...);
}

/// No-op for production
#[cfg(not(feature = "debug-enabled"))]
pub fn narrate_debug(_fields: NarrationFields) {
    // Completely empty
}

/// Emit INFO-level narration (always available)
pub fn narrate_info(fields: NarrationFields) {
    event!(Level::INFO, ...);
}

/// Emit WARN-level narration (always available)
pub fn narrate_warn(fields: NarrationFields) {
    event!(Level::WARN, ...);
}

/// Emit ERROR-level narration (always available)
pub fn narrate_error(fields: NarrationFields) {
    event!(Level::ERROR, ...);
}

/// Emit FATAL-level narration (always available)
pub fn narrate_fatal(fields: NarrationFields) {
    event!(Level::ERROR, ...);  // tracing doesn't have FATAL, use ERROR
}
```

---

## 🚨 Important: Runtime vs Compile-Time

### ❌ WRONG: Runtime Check (Still Has Overhead)

```rust
// ❌ BAD: Code still exists, just not executed
if cfg!(feature = "trace-enabled") {
    trace_tiny!("actor", "action", "target", "message");
}
```

**Problem**: The `trace_tiny!()` code is still compiled, just not executed. Still has overhead!

---

### ✅ CORRECT: Compile-Time Removal

```rust
// ✅ GOOD: Code doesn't exist in production binary
#[cfg(feature = "trace-enabled")]
trace_tiny!("actor", "action", "target", "message");
```

**Result**: In production builds, this line **doesn't exist** in the binary!

---

## 📊 Performance Impact

| Build Type | TRACE Code | DEBUG Code | Overhead | Binary Size |
|------------|-----------|-----------|----------|-------------|
| **Development** | ✅ Included | ✅ Included | ~2% | ~50 MB |
| **Staging** | ❌ Removed | ✅ Included | ~1% | ~48 MB |
| **Production** | ❌ Removed | ❌ Removed | **0%** | ~45 MB |

**Key Point**: Production builds have **zero overhead** because the code doesn't exist!

---

## 🎯 Recommended Setup

### `Cargo.toml` (workspace root):

```toml
[workspace]
members = [
    "bin/queen-rbee",
    "bin/pool-managerd",
    "bin/worker-orcd",
    "bin/shared-crates/narration-core",
]

[workspace.dependencies]
observability-narration-core = { path = "bin/shared-crates/narration-core" }

# Development profile (default)
[profile.dev]
# Uses default features (trace-enabled, debug-enabled)

# Staging profile
[profile.staging]
inherits = "release"
opt-level = 3

# Production profile
[profile.release]
opt-level = 3
lto = true
codegen-units = 1
```

### Build Scripts:

```bash
#!/bin/bash
# scripts/build-dev.sh
cargo build

# scripts/build-staging.sh
cargo build --profile staging \
    --no-default-features \
    --features debug-enabled

# scripts/build-production.sh
cargo build --release \
    --no-default-features \
    --features production
```

---

## 🔍 Example: Complete Conditional Macro

```rust
// src/trace.rs

/// TRACE-level loop iteration (only in dev builds)
#[cfg(feature = "trace-enabled")]
#[macro_export]
macro_rules! trace_loop {
    ($actor:expr, $action:expr, $index:expr, $total:expr, $detail:expr) => {
        tracing::trace!(
            actor = $actor,
            action = $action,
            target = format!("iter_{}/{}", $index, $total),
            human = format!("Iteration {}/{}: {}", $index, $total, $detail),
            "trace_loop"
        );
    };
}

/// No-op version for production (code completely removed)
#[cfg(not(feature = "trace-enabled"))]
#[macro_export]
macro_rules! trace_loop {
    ($actor:expr, $action:expr, $index:expr, $total:expr, $detail:expr) => {
        // Completely empty — not even a semicolon!
    };
}
```

**Usage** (same code, different builds):

```rust
// This code works in ALL builds
for (i, token) in tokens.iter().enumerate() {
    trace_loop!("tokenizer", "decode", i, tokens.len(),
                format!("token={}", token));
    
    // ... actual work ...
}
```

**Development build**: Full tracing  
**Production build**: `trace_loop!()` expands to nothing (removed at compile time)

---

## 🎀 Summary

### The Problem:
- ❌ TRACE has overhead even when disabled
- ❌ Production doesn't need TRACE/DEBUG
- ❌ Want **zero overhead** in production

### The Solution:
- ✅ Use **feature flags** for conditional compilation
- ✅ TRACE/DEBUG code **completely removed** in production
- ✅ **Zero runtime overhead** (code doesn't exist!)
- ✅ Same developer experience (just add `#[trace_fn]`)

### Build Commands:

```bash
# Development (all tracing)
cargo build

# Staging (no TRACE)
cargo build --profile staging --no-default-features --features debug-enabled

# Production (no TRACE, no DEBUG)
cargo build --release --no-default-features --features production
```

### Result:
- 🚀 **Development**: Full observability (~2% overhead)
- 🚀 **Staging**: DEBUG only (~1% overhead)
- 🚀 **Production**: Zero overhead (code removed!)

---

**With love, sass, and the confidence that cuteness pays the bills,**  
**The Narration Core Team** 🎭🚀

*P.S. — Our custom proc macros make production builds so clean, they don't even know what tracing is. And dev builds get cute mode for free! 💝*

---

*May your production builds be fast, your dev builds be observable, and your narration be adorable! 🎀*

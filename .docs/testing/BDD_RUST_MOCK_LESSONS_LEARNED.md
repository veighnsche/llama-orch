# BDD Testing with Cucumber + Rust + Mock FFI: Lessons Learned

**Date**: 2025-10-02  
**Source**: `vram-residency` crate BDD test debugging session  
**Status**: Production-ready patterns

---

## Executive Summary

This document captures critical lessons learned from implementing and debugging BDD tests in Rust using Cucumber with mock C FFI code. These patterns are essential for any crate that:
- Uses Cucumber for BDD testing
- Interfaces with C/C++ via FFI
- Needs to mock external dependencies (GPU, hardware, etc.)
- Runs integration tests outside of `cfg(test)` mode

**Key Insight**: BDD binaries are NOT compiled with `cfg(test)`, which breaks many common Rust testing assumptions.

---

## Problem 1: BDD Binaries Don't Get `cfg(test)`

### The Issue

```rust
// This ONLY works in unit tests, NOT in BDD binaries!
#[cfg(test)]
{
    // Mock initialization
}

#[cfg(not(test))]
{
    // Production code that validates real hardware
}
```

**Why it fails**: Cucumber BDD tests run as separate binaries (`[[bin]]` in `Cargo.toml`), not as `#[test]` functions. The `cfg(test)` flag is **never set** for these binaries.

### The Solution

Use environment variables for test mode detection:

```rust
pub fn new(device: u32) -> Result<Self> {
    // Check for BDD test mode (integration tests)
    #[cfg(not(test))]
    {
        if std::env::var("LLORCH_BDD_MODE").is_ok() {
            tracing::warn!(
                device = %device,
                "Context initialized in BDD MODE (no hardware validation)"
            );
            return Ok(Self { device });
        }
    }
    
    // Unit test mode
    #[cfg(test)]
    {
        tracing::warn!(device = %device, "TEST MODE");
        return Ok(Self { device });
    }
    
    // Production mode: validate real hardware
    #[cfg(not(test))]
    {
        let hardware = detect_hardware()?;
        // ... validation logic
    }
}
```

**Set the flag in your BDD main.rs**:

```rust
#[tokio::main(flavor = "multi_thread")]
async fn main() {
    // Enable BDD test mode FIRST
    std::env::set_var("LLORCH_BDD_MODE", "1");
    
    BddWorld::cucumber().run_and_exit(features).await;
}
```

---

## Problem 2: Static State in Mock C Code Persists Across Scenarios

### The Issue

Mock C/C++ code often uses static variables for tracking:

```c
static size_t mock_allocated_bytes = 0;
static AllocationEntry allocations[MAX_ALLOCATIONS];
```

**Problem**: Cucumber creates a new `World` for each scenario, but C static variables persist across ALL scenarios in the entire test run.

**Symptom**: Tests pass individually but fail when run together. Later scenarios see "leaked" state from earlier scenarios.

### The Solution

Implement a reset function in your mock C code:

```c
// Reset mock state (for testing only)
// WARNING: This leaks memory but is safe for BDD tests
// We reset tracking counters but don't free memory because
// Rust's Drop will handle that
void mock_reset_state(void) {
    allocation_count = 0;
    mock_allocated_bytes = 0;
}
```

**Critical**: Don't free the memory in C! Rust's `Drop` implementations still hold pointers to that memory and will free it later. Freeing in C causes double-free crashes.

Expose to Rust:

```rust
extern "C" {
    fn mock_reset_state();
}
```

Call it when creating fresh test state:

```rust
#[given(expr = "a Manager with {int}MB capacity")]
pub async fn given_manager_with_capacity(world: &mut BddWorld, capacity_mb: usize) {
    // Clear old state
    world.items.clear();
    if let Some(old_manager) = world.manager.take() {
        std::mem::drop(old_manager);
    }
    
    // Reset mock C state
    extern "C" { fn mock_reset_state(); }
    unsafe { mock_reset_state(); }
    
    // Set environment variables for mock configuration
    std::env::set_var("MOCK_CAPACITY_MB", capacity_mb.to_string());
    
    // Create fresh manager
    world.manager = Some(Manager::new());
}
```

---

## Problem 3: Rust Drop Deferral in Async Contexts

### The Issue

When you assign a new value to an `Option<T>`, Rust should drop the old value immediately. But in async contexts or when panics occur, Drop can be deferred:

```rust
world.manager = Some(new_manager); // Old manager SHOULD drop here
// But Drop might not run until much later!
```

### The Solution

Explicitly drop before reassigning:

```rust
// Force immediate drop
if let Some(old_manager) = world.manager.take() {
    std::mem::drop(old_manager);
}

// Now create new one
world.manager = Some(Manager::new());
```

Also implement `Drop` for your `World` to ensure cleanup:

```rust
impl Drop for BddWorld {
    fn drop(&mut self) {
        // Clear collections first
        self.items.clear();
        
        // Force drop of manager
        if let Some(manager) = self.manager.take() {
            std::mem::drop(manager);
        }
    }
}
```

---

## Problem 4: Environment Variables and Mock Configuration

### The Issue

Mock C code reads environment variables at runtime:

```c
const char* capacity_env = getenv("MOCK_CAPACITY_MB");
```

But if you set the variable AFTER the C code has already read it, the mock won't see the new value.

### The Solution

**Always set environment variables BEFORE creating objects that read them**:

```rust
#[given(expr = "a Manager with {int}MB capacity")]
pub async fn given_manager(world: &mut BddWorld, capacity_mb: usize) {
    // 1. Set environment variable FIRST
    std::env::set_var("MOCK_CAPACITY_MB", capacity_mb.to_string());
    
    // 2. THEN create the manager (which will read the env var)
    world.manager = Some(Manager::new());
}
```

**Pattern for dynamic mock configuration**:

```c
int mock_get_info(size_t* free, size_t* total) {
    // Read env var EVERY TIME (don't cache)
    const char* capacity_mb = getenv("MOCK_CAPACITY_MB");
    
    if (capacity_mb) {
        *total = atoi(capacity_mb) * 1024ULL * 1024ULL;
    } else {
        *total = DEFAULT_CAPACITY;
    }
    
    *free = *total - mock_allocated;
    return SUCCESS;
}
```

---

## Problem 5: Step Definition Visibility

### The Issue

Cucumber step definitions in different files can't call each other if they're not `pub`:

```rust
// In file_a.rs
#[given("something")]
async fn given_something(world: &mut World) { }

// In file_b.rs
#[given("something else")]
async fn given_something_else(world: &mut World) {
    given_something(world).await; // ERROR: private function
}
```

### The Solution

Make reusable step definitions `pub`:

```rust
// In file_a.rs
#[given("something")]
pub async fn given_something(world: &mut World) { }

// In file_b.rs
use super::file_a::given_something;

#[given("something else")]
pub async fn given_something_else(world: &mut World) {
    given_something(world).await; // OK
}
```

**Pattern**: Create "composite" steps that reuse atomic steps:

```rust
// atomic.rs
#[given("a manager")]
pub async fn given_manager(world: &mut World) { }

#[given("a model with {int}MB")]
pub async fn given_model(world: &mut World, size_mb: usize) { }

// composite.rs
#[given("a sealed model {string} with {int}MB")]
pub async fn given_sealed_model(
    world: &mut World, 
    id: String, 
    size_mb: usize
) {
    given_manager(world).await;
    given_model(world, size_mb).await;
    when_seal_model(world, id, 0).await;
    
    if !world.last_succeeded() {
        panic!("Failed to create sealed model");
    }
}
```

---

## Problem 6: Auto-Generated IDs vs Test-Specified IDs

### The Issue

Your production code might auto-generate IDs:

```rust
pub fn create_item(&mut self, data: &[u8]) -> Result<Item> {
    let id = format!("item-{}-{:x}", self.counter, hash(data));
    // ...
}
```

But BDD scenarios want to specify IDs:

```gherkin
When I create item "my-specific-id"
```

### The Solution

**Option A**: Store the ACTUAL generated ID, not the requested one:

```rust
#[when("I create item {string}")]
async fn when_create_item(world: &mut World, requested_id: String) {
    let result = world.manager.create_item(&world.data);
    
    match result {
        Ok(item) => {
            let actual_id = item.id.clone(); // Auto-generated!
            world.items.insert(actual_id.clone(), item);
            world.current_id = actual_id; // Store actual, not requested
        }
        Err(e) => world.store_error(e),
    }
}
```

**Option B**: Remove scenarios that test specific IDs if your API doesn't support them:

```gherkin
# Remove this if IDs are auto-generated:
Scenario: Reject invalid ID with path traversal
  When I create item "../etc/passwd"
  Then it should fail with "InvalidInput"
```

---

## Problem 7: Debugging BDD Failures

### Patterns for Effective Debugging

**1. Add debug output to step definitions**:

```rust
#[given("a manager with {int}MB capacity")]
pub async fn given_manager(world: &mut World, capacity_mb: usize) {
    println!("✓ Manager created with {}MB capacity", capacity_mb);
}

#[when("I create item")]
async fn when_create(world: &mut World) {
    match result {
        Ok(item) => println!("✓ Item created: {}", item.id),
        Err(e) => println!("✗ Creation failed: {}", e),
    }
}
```

**2. Add debug output to mock C code** (remove before committing):

```c
int mock_operation(void) {
    fprintf(stderr, "[mock] operation called, state=%d\n", state);
    return SUCCESS;
}
```

**3. Run specific scenarios**:

```bash
# Set environment variable to run specific feature file
LLORCH_BDD_FEATURE_PATH=tests/features/specific.feature cargo run --bin bdd-runner
```

**4. Check for undefined steps**:

```bash
cargo run --bin bdd-runner 2>&1 | grep "Undefined step"
```

---

## Complete BDD Setup Checklist

### 1. Cargo.toml Structure

```toml
# Main crate
[package]
name = "my-crate"

# BDD test crate (separate package)
# In my-crate/bdd/Cargo.toml:
[package]
name = "my-crate-bdd"

[dependencies]
cucumber = { version = "0.20", features = ["macros"] }
tokio = { version = "1", features = ["macros", "rt-multi-thread"] }
my-crate = { path = ".." }

[[bin]]
name = "bdd-runner"
path = "src/main.rs"
```

### 2. BDD Main Entry Point

```rust
// bdd/src/main.rs
mod steps;

use cucumber::World as _;
use steps::world::BddWorld;

#[tokio::main(flavor = "multi_thread")]
async fn main() {
    // CRITICAL: Set BDD mode before anything else
    std::env::set_var("LLORCH_BDD_MODE", "1");
    
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let features = root.join("tests/features");
    
    BddWorld::cucumber().run_and_exit(features).await;
}
```

### 3. World Structure

```rust
// bdd/src/steps/world.rs
use cucumber::World;
use std::collections::HashMap;

#[derive(World)]
#[world(init = Self::new)]
pub struct BddWorld {
    pub manager: Option<Manager>,
    pub items: HashMap<String, Item>,
    pub last_result: Option<Result<(), Error>>,
    pub current_data: Vec<u8>,
}

impl BddWorld {
    pub fn new() -> Self {
        Self {
            manager: None,
            items: HashMap::new(),
            last_result: None,
            current_data: Vec::new(),
        }
    }
    
    pub fn store_result(&mut self, result: Result<(), Error>) {
        self.last_result = Some(result);
    }
    
    pub fn last_succeeded(&self) -> bool {
        matches!(self.last_result, Some(Ok(())))
    }
}

impl Drop for BddWorld {
    fn drop(&mut self) {
        self.items.clear();
        if let Some(manager) = self.manager.take() {
            std::mem::drop(manager);
        }
    }
}
```

### 4. Step Definitions Module Structure

```
bdd/src/steps/
├── mod.rs          # Exports all step modules
├── world.rs        # BddWorld definition
├── setup.rs        # Given steps (setup)
├── actions.rs      # When steps (actions)
├── assertions.rs   # Then steps (assertions)
└── helpers.rs      # Shared utilities
```

```rust
// bdd/src/steps/mod.rs
pub mod world;
mod setup;
mod actions;
mod assertions;
mod helpers;
```

### 5. Mock C Code Pattern

```c
// src/ffi/mock_impl.c
#include <stdlib.h>
#include <stdio.h>

static size_t mock_state = 0;

int mock_operation(void* ptr, size_t size) {
    // Read config from environment EVERY TIME
    const char* config = getenv("MOCK_CONFIG");
    
    // Update state
    mock_state += size;
    
    return SUCCESS;
}

// Reset function for BDD tests
void mock_reset_state(void) {
    mock_state = 0;
    // Don't free memory - Rust will do that
}
```

### 6. Build Script

```rust
// build.rs
fn main() {
    let build_mock = env::var("MY_CRATE_BUILD_REAL")
        .unwrap_or_else(|_| "0".to_string()) != "1";
    
    if build_mock {
        println!("cargo:warning=Building with MOCK implementation");
        build_mock_impl();
    } else {
        println!("cargo:warning=Building with REAL implementation");
        build_real_impl();
    }
}

fn build_mock_impl() {
    cc::Build::new()
        .file("src/ffi/mock_impl.c")
        .compile("mock_impl");
}
```

---

## Common Pitfalls

### ❌ Don't Do This

```rust
// DON'T: Rely on cfg(test) in production code used by BDD
#[cfg(test)]
fn init_mock() { }

// DON'T: Cache environment variables in C
static size_t cached_config = 0;
void init() {
    if (cached_config == 0) {
        cached_config = atoi(getenv("CONFIG"));
    }
}

// DON'T: Free memory in mock reset
void mock_reset() {
    for (int i = 0; i < count; i++) {
        free(allocations[i]); // CRASH: Rust will free this later!
    }
}

// DON'T: Assume Drop runs immediately
world.manager = Some(new_manager); // Old one might not drop yet!
```

### ✅ Do This Instead

```rust
// DO: Use environment variables for test mode
if std::env::var("BDD_MODE").is_ok() { }

// DO: Read env vars dynamically in C
const char* config = getenv("CONFIG"); // Every time!

// DO: Reset counters only, not memory
void mock_reset() {
    allocation_count = 0;
    // Memory will be freed by Rust
}

// DO: Explicitly drop before reassigning
if let Some(old) = world.manager.take() {
    std::mem::drop(old);
}
world.manager = Some(new_manager);
```

---

## Testing the Tests

Verify your BDD setup works:

```bash
# Should pass with mock
cargo run --bin bdd-runner

# Should show BDD mode warning
cargo run --bin bdd-runner 2>&1 | grep "BDD MODE"

# Should reset between scenarios
cargo run --bin bdd-runner 2>&1 | grep "mock_reset_state"

# Check for memory leaks (acceptable in tests)
valgrind --leak-check=full cargo run --bin bdd-runner
```

---

## Summary: The Golden Rules

1. **BDD binaries are NOT `cfg(test)`** - use environment variables
2. **Static C state persists** - implement reset functions
3. **Explicitly drop before reassigning** - don't rely on automatic Drop
4. **Set env vars BEFORE creating objects** - timing matters
5. **Make reusable steps `pub`** - enable composition
6. **Store actual IDs, not requested ones** - if auto-generated
7. **Don't free in C reset** - let Rust's Drop handle it
8. **Add debug output liberally** - remove before committing
9. **Test scenarios independently** - ensure no cross-contamination
10. **Document your mock behavior** - future you will thank you

---

## References

- Cucumber Rust: https://github.com/cucumber-rs/cucumber
- This pattern used in: `bin/worker-orcd-crates/vram-residency/bdd/`
- Related: `.docs/testing/BDD_WIRING.md`
- Related: `.docs/testing/TESTING_POLICY.md`

---

**Last Updated**: 2025-10-02  
**Maintainer**: llama-orch core team  
**Status**: Production-ready, battle-tested

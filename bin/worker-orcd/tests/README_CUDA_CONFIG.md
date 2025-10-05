# CUDA Test Configuration System

## Overview

The worker-orcd test suite reads CUDA configuration from `.llorch.toml` and **LOUDLY announces** whether tests are running with real CUDA or stub implementations.

## Configuration File

**Location**: `/home/vince/Projects/llama-orch/.llorch.toml` (gitignored)

**Template**: Copy from `.llorch.toml.example`

```toml
[build]
# CUDA support (default: true)
cuda = false  # Set to true to enable CUDA

# Auto-detect CUDA toolkit (default: false)
auto_detect_cuda = false

[development]
# Skip CUDA tests even when CUDA is available (default: false)
skip_cuda_tests = false
```

## Test Behavior

### When `build.cuda = false` (STUB MODE)

Tests will display a **LOUD announcement**:

```
╔═══════════════════════════════════════════════════════════════╗
║  ⛔ CUDA DISABLED - Running tests in STUB mode               ║
║                                                               ║
║  Configuration: .llorch.toml                                  ║
║  • build.cuda = false                                         ║
║                                                               ║
║  ⚠️  All CUDA-dependent tests will use STUB implementations   ║
║  ⚠️  No actual GPU operations will be performed               ║
║                                                               ║
║  To enable CUDA:                                              ║
║  1. Edit .llorch.toml                                         ║
║  2. Set build.cuda = true                                     ║
║  3. Rebuild: cargo clean && cargo test                        ║
╚═══════════════════════════════════════════════════════════════╝
```

Each test will also announce:
```
🔧 [STUB MODE] test_name: Using stub CUDA implementation
```

### When `build.cuda = true` (CUDA ENABLED)

Tests will display:

```
╔═══════════════════════════════════════════════════════════════╗
║  🚀 CUDA ENABLED - Running tests with GPU support            ║
║                                                               ║
║  Configuration: .llorch.toml                                  ║
║  • build.cuda = true                                          ║
║  • development.skip_cuda_tests = false                       ║
║                                                               ║
║  ✅ CUDA tests will RUN                                       ║
╚═══════════════════════════════════════════════════════════════╝
```

### When `skip_cuda_tests = true`

Even with CUDA enabled, tests can be skipped:

```
╔═══════════════════════════════════════════════════════════════╗
║  🚀 CUDA ENABLED - Running tests with GPU support            ║
║                                                               ║
║  Configuration: .llorch.toml                                  ║
║  • build.cuda = true                                          ║
║  • development.skip_cuda_tests = true                        ║
║                                                               ║
║  ⚠️  CUDA tests will be SKIPPED (skip_cuda_tests=true)       ║
╚═══════════════════════════════════════════════════════════════╝
```

## Using in Tests

### Initialize Test Environment

Every test file should include:

```rust
mod common;

#[test]
fn my_test() {
    common::init_test_env();  // Shows LOUD announcement once
    announce_stub_mode!("my_test");  // Shows stub mode per test
    
    // Test code...
}
```

### Require CUDA

For tests that MUST have CUDA:

```rust
#[test]
fn cuda_required_test() {
    require_cuda!();  // Skips test if CUDA disabled
    
    // CUDA-specific test code...
}
```

### Conditional Code

```rust
// Run only with CUDA
cuda_only! {
    // Real CUDA operations
}

// Run only in stub mode
stub_only! {
    // Stub implementations
}
```

## Macros Available

- `init_test_env()` - Initialize and show LOUD announcement (call once per test)
- `announce_stub_mode!(test_name)` - Announce stub mode for specific test
- `require_cuda!()` - Skip test if CUDA not available
- `is_cuda_enabled()` - Check if CUDA is enabled
- `cuda_only! { }` - Code that runs only with CUDA
- `stub_only! { }` - Code that runs only in stub mode

## Setup Instructions

### 1. Create Configuration

```bash
cd /home/vince/Projects/llama-orch
cp .llorch.toml.example .llorch.toml
```

### 2. Edit Configuration

For **CPU-only development**:
```toml
[build]
cuda = false
auto_detect_cuda = false
```

For **GPU development**:
```toml
[build]
cuda = true
auto_detect_cuda = false
```

For **Laptop (auto-detect)**:
```toml
[build]
cuda = true
auto_detect_cuda = true
```

### 3. Rebuild

```bash
cd bin/worker-orcd
cargo clean
cargo test
```

## Verification

Run a single test to see the announcement:

```bash
cargo test test_adapter_unified_interface_qwen -- --nocapture
```

You should see the **LOUD** CUDA status announcement in the output.

## Implementation Details

- Configuration is read from `../../.llorch.toml` relative to test binary
- Announcement is shown **once** per test run via `Once` synchronization
- Each test can optionally announce stub mode individually
- Build script (`build.rs`) also reads this config to enable/disable CUDA compilation
- Tests use `#[cfg(feature = "cuda")]` for conditional compilation

## Troubleshooting

**Q: Tests still use CUDA even though I set `cuda = false`**
- Run `cargo clean` first, then rebuild

**Q: No announcement shown**
- Make sure you call `common::init_test_env()` in your test
- Use `--nocapture` flag: `cargo test -- --nocapture`

**Q: Configuration file not found**
- Copy `.llorch.toml.example` to `.llorch.toml`
- File must be at repo root: `/home/vince/Projects/llama-orch/.llorch.toml`

**Q: Want to skip CUDA tests temporarily**
- Set `development.skip_cuda_tests = true` in `.llorch.toml`
- No rebuild needed, just rerun tests

# Feature Flags Guide for llm-worker-rbee

## What Are Feature Flags?

Feature flags in Rust let you enable/disable code at compile time. Think of them like compiler switches that turn parts of your code on or off.

## The Problem You Had

Your code was showing as "inactive" (grayed out) because **no backend feature was enabled by default**. The code was there, but the compiler wasn't including it in the build.

## The Fix

Changed this:
```toml
[features]
default = []  # ❌ Nothing enabled!
```

To this:
```toml
[features]
default = ["cpu"]  # ✅ CPU enabled by default
```

## Available Backends

### 🖥️ CPU (Default)
```bash
# These all use CPU:
cargo build
cargo test
cargo run
```

**When to use**: Development, testing, or when you don't have a GPU.

### 🎮 CUDA (NVIDIA GPUs)
```bash
# Explicitly enable CUDA:
cargo build --features cuda --no-default-features
cargo test --features cuda --no-default-features
cargo run --bin llorch-cuda-candled --features cuda
```

**When to use**: Production inference on NVIDIA GPUs (RTX 3060, 3090, etc.)

### 🍎 Metal (Apple Silicon GPU)
```bash
# Explicitly enable Metal:
cargo build --features metal --no-default-features
cargo test --features metal --no-default-features
cargo run --bin llorch-metal-candled --features metal
```

**When to use**: GPU inference on Mac with M1/M2/M3/M4 chips (pre-release).

## How Feature Flags Work

### In Code
```rust
// This code only compiles when 'cpu' feature is enabled
#[cfg(feature = "cpu")]
pub fn init_cpu_device() -> Result<Device> {
    Ok(Device::Cpu)
}

// This code only compiles when 'cuda' feature is enabled
#[cfg(feature = "cuda")]
pub fn init_cuda_device(gpu_id: usize) -> Result<Device> {
    Device::new_cuda(gpu_id)
}
```

### Why It's Grayed Out
When you see grayed out code in your IDE, it means:
- The code exists in the file
- But it won't be compiled because the feature isn't enabled
- The IDE is showing you "this code is inactive"

## IDE Configuration

### VS Code / Windsurf
Add to `.vscode/settings.json`:
```json
{
    "rust-analyzer.cargo.features": ["cpu"]
}
```

Or for CUDA:
```json
{
    "rust-analyzer.cargo.features": ["cuda"]
}
```

### RustRover / IntelliJ
Go to: **Settings → Languages & Frameworks → Rust → Cargo**
- Check "Use all features"
- Or manually add: `cpu` to the features list

## Common Commands

### Development (CPU)
```bash
# Just use defaults - CPU is now enabled
cargo check
cargo build
cargo test
```

### Testing Specific Backend
```bash
# Test CPU code
cargo test --features cpu

# Test CUDA code (requires NVIDIA GPU)
cargo test --features cuda --no-default-features

# Test Metal code (requires Apple Silicon)
cargo test --features metal --no-default-features
```

### Building Specific Binary
```bash
# CPU binary (default)
cargo build --bin llm-worker-rbee

# CUDA binary
cargo build --bin llorch-cuda-candled --features cuda

# Metal binary
cargo build --bin llorch-metal-candled --features metal
```

## Why Multiple Backends?

Different hardware needs different code:
- **CPU**: Pure Rust, works everywhere, slower
- **CUDA**: NVIDIA GPU kernels, 10-50x faster, needs NVIDIA GPU
- **Metal**: Apple's GPU API, fast on M-series Macs

We can't enable all at once because they have conflicting dependencies.

## Troubleshooting

### "Code is still grayed out"
1. Restart your IDE (rust-analyzer needs to reload)
2. Run: `cargo clean && cargo check`
3. Check IDE settings (see above)

### "Feature 'cpu' not found"
- Make sure you're in the right directory: `cd bin/llm-worker-rbee`
- Check `Cargo.toml` has the features section

### "CUDA errors"
- CUDA requires NVIDIA GPU and drivers
- Use `--features cpu` instead for development

### "Multiple backends conflict"
- This is intentional! Use `--no-default-features` when switching:
  ```bash
  cargo build --features cuda --no-default-features
  ```

## Quick Reference

| What You Want | Command |
|---------------|---------|
| Normal development | `cargo build` (uses CPU) |
| Test everything | `cargo test` (uses CPU) |
| Run on NVIDIA GPU | `cargo run --features cuda --no-default-features` |
| Run on Apple Silicon GPU | `cargo run --features metal --no-default-features` |
| Check all features | `cargo check --all-features` |

## Summary

✅ **Fixed**: Changed `default = []` to `default = ["cpu"]`  
✅ **Result**: Code is now active by default  
✅ **Benefit**: No more grayed out code in your IDE  
✅ **Flexibility**: Can still switch backends when needed  

Now your code should show as active (not grayed out) because the `cpu` feature is enabled by default! 🎉

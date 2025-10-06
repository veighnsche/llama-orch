# Quick Build Reference

**TL;DR**: You can now edit `.cu`/`.cpp` files and just run `cargo build` — no more `cargo clean`! 🎉

---

## Daily Workflow

### Editing CUDA Kernels
```bash
# 1. Edit kernel
vim cuda/kernels/gqa_attention.cu

# 2. Build (3-5 seconds)
cargo build --features cuda

# 3. Test
cargo test --test haiku_generation_anti_cheat --features cuda -- --nocapture --ignored
```

### Editing C++ Implementation
```bash
# 1. Edit implementation
vim cuda/src/transformer/qwen_transformer.cpp

# 2. Build (3-5 seconds)
cargo build --features cuda

# 3. Run
cargo run --features cuda --bin worker-orcd
```

### Editing Rust Code
```bash
# 1. Edit Rust
vim src/inference/cuda_backend.rs

# 2. Build (1-2 seconds)
cargo build --features cuda

# 3. Test
cargo test --features cuda
```

---

## When to Clean Build

### ✅ Normal Changes (No Clean Needed)
- Editing `.cu` kernel files
- Editing `.cpp` implementation files
- Editing `.h` or `.cuh` headers
- Editing `.rs` Rust files
- Adding new source files

### ⚠️ Rare Cases (Clean Required)
- Modifying `CMakeLists.txt`
- Changing build flags in `build.rs`
- Linking errors (very rare)
- Switching debug ↔ release

```bash
# Clean only worker-orcd (fast)
cargo clean -p worker-orcd

# Full clean (slow, rarely needed)
cargo clean
```

---

## Build Times Reference

| Change | Time | Command |
|--------|------|---------|
| Single `.cu` file | 3-5s | `cargo build --features cuda` |
| Single `.cpp` file | 3-5s | `cargo build --features cuda` |
| Single `.rs` file | 1-2s | `cargo build --features cuda` |
| Header file | 5-15s | `cargo build --features cuda` |
| Full rebuild | 90-120s | `cargo clean -p worker-orcd && cargo build --features cuda` |

---

## Troubleshooting

### Build doesn't pick up my change
```bash
# Force rebuild
touch cuda/kernels/your_file.cu
cargo build --features cuda
```

### CMake errors
```bash
# Clear CMake cache
rm -rf cuda/build
cargo clean -p worker-orcd
cargo build --features cuda
```

### Linking errors
```bash
# Full clean rebuild
cargo clean -p worker-orcd
cargo build --features cuda
```

---

## Verification

### Check if file is tracked
```bash
cargo build --features cuda -vv 2>&1 | grep "rerun-if-changed.*your_file"
```

### See all tracked files
```bash
cargo clean -p worker-orcd
cargo build --features cuda -vv 2>&1 | grep "rerun-if-changed.*\.cu"
```

---

**Remember**: Just `cargo build --features cuda` — that's it! 🚀

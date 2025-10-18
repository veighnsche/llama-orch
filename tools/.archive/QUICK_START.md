# 🚀 Quick Start Guide

Get from fresh Ubuntu to production-ready inference workstation in 20 minutes!

---

## One-Line Install

```bash
./tools/setup-dev-workstation.sh
```

That's it! The script will:
1. ✅ Install all development tools
2. ✅ Install Rust toolchain
3. ✅ Install NVIDIA drivers + CUDA
4. ✅ Build and test everything
5. ✅ Verify system is ready

---

## What You Get

After running the script:

- **Development Environment**
  - Rust 1.90.0 + Cargo
  - CMake, GCC, G++, Make
  - Google Test for C++

- **GPU Infrastructure**
  - NVIDIA drivers (latest)
  - CUDA toolkit 12.0+
  - Multi-GPU support

- **Validated System**
  - 479 Rust tests passing
  - 426 CUDA tests built
  - Ready for inference

---

## Installation Options

### Full Install (Recommended)
```bash
./tools/setup-dev-workstation.sh
```

### CPU-Only (No GPU)
```bash
./tools/setup-dev-workstation.sh --skip-nvidia
```

### Quick Install (Skip Tests)
```bash
./tools/setup-dev-workstation.sh --skip-tests
```

### Custom CUDA Version
```bash
./tools/setup-dev-workstation.sh --cuda-version 12-0
```

---

## After Installation

### If Reboot Required

The script will tell you if reboot is needed (for NVIDIA drivers):

```bash
# 1. Reboot
sudo reboot

# 2. Verify GPUs
nvidia-smi

# 3. Run CUDA tests
cd bin/worker-orcd/cuda/build
./cuda_tests
```

### Build Worker

```bash
cd bin/worker-orcd
cargo build --release
```

### Download Test Model

```bash
mkdir -p ~/models && cd ~/models
wget https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf
```

### Start Inference

```bash
cd ~/llama-orch/bin/worker-orcd
./target/release/worker-orcd \
  --model ~/models/qwen2.5-0.5b-instruct-q4_k_m.gguf \
  --gpu 0 \
  --port 8080
```

### Test Inference

```bash
# Health check
curl http://localhost:8080/health

# Generate text
curl -X POST http://localhost:8080/execute \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Write a haiku about testing",
    "max_tokens": 50,
    "temperature": 0.7
  }'
```

---

## Troubleshooting

### Script Permission Denied
```bash
chmod +x tools/setup-dev-workstation.sh
```

### No NVIDIA GPU
```bash
./tools/setup-dev-workstation.sh --skip-nvidia
```

### Rust Already Installed
```bash
./tools/setup-dev-workstation.sh --skip-rust
```

### Need Help
```bash
./tools/setup-dev-workstation.sh --help
```

---

## System Requirements

- **OS**: Ubuntu 24.04 LTS (or compatible)
- **RAM**: 8GB minimum, 16GB+ recommended
- **Disk**: 20GB free space
- **GPU**: NVIDIA GPU (optional, for CUDA)
- **Network**: Internet connection for downloads

---

## Expected Results

### Test Summary
- ✅ 266 Rust library tests
- ✅ 213 Rust integration tests  
- ✅ 426 CUDA C++ tests
- ✅ **905 total tests passing**

### Installation Time
- Build tools: ~2 minutes
- Rust: ~2 minutes
- NVIDIA + CUDA: ~5 minutes
- Building tests: ~5 minutes
- Running tests: ~10 seconds
- **Total: ~15 minutes**

### Disk Usage
- Rust toolchain: ~2GB
- CUDA toolkit: ~3GB
- Build artifacts: ~2GB
- **Total: ~7GB**

---

## What's Tested

### Foundation Team ✅
- HTTP server and middleware
- FFI boundary (Rust ↔ C++)
- CUDA context management
- Sampling kernels
- Error handling

### Llama Team ✅
- GGUF parser
- BPE tokenizer
- RoPE, RMSNorm, GQA kernels
- KV cache management
- Model adapters

### GPT Team ✅
- GPT model support
- Advanced sampling
- Model integration
- All shared components

---

## Next Steps

1. ✅ Run the setup script
2. ✅ Reboot if needed
3. ✅ Verify tests pass
4. ✅ Build worker binary
5. ✅ Download a model
6. 🚀 Start serving inference!

---

**Questions?** Check the full [Tools README](README.md) or [Worker-orcd docs](../bin/worker-orcd/README.md)

**Ready to go?** Run: `./tools/setup-dev-workstation.sh`

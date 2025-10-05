# llama-orch Tools

This directory contains utilities and scripts for working with the llama-orch project.

---

## 🚀 Quick Start Scripts

### `setup-dev-workstation.sh`

Automated setup script that transforms a fresh Ubuntu system into a complete development and inference workstation.

**What it installs**:
- ✅ Rust toolchain (rustup, cargo, rustc)
- ✅ Build tools (cmake, gcc, g++, make)
- ✅ NVIDIA drivers (latest stable)
- ✅ CUDA toolkit (12.0+)
- ✅ Google Test (C++ testing)
- ✅ All development dependencies

**Usage**:
```bash
# Full installation (recommended for fresh Ubuntu systems)
./setup-dev-workstation.sh

# Skip NVIDIA/CUDA (for CPU-only development)
./setup-dev-workstation.sh --skip-nvidia

# Skip Rust (if already installed)
./setup-dev-workstation.sh --skip-rust

# Skip running tests after setup
./setup-dev-workstation.sh --skip-tests

# Show all options
./setup-dev-workstation.sh --help
```

**Requirements**:
- Ubuntu 24.04 LTS (or compatible)
- Sudo access
- Internet connection

**Duration**: ~15-20 minutes

**Result**:
- 479 Rust tests passing
- 426 CUDA tests built and ready
- System ready for inference and development

---

## 🛠️ Development Tools

### `openapi-client/`

OpenAPI client generator for llama-orch APIs.

### `readme-index/`

README index generator for documentation.

### `spec-extract/`

Specification extraction and validation tools.

### `worker-crates-migration/`

Tools for migrating worker crates and dependencies.

---

## 📝 Examples

### Complete Fresh Install

Starting with a fresh Ubuntu 24.04 server:

```bash
# Clone repository
git clone https://github.com/your-org/llama-orch.git
cd llama-orch

# Run setup script
./tools/setup-dev-workstation.sh

# After reboot (if NVIDIA drivers were installed)
sudo reboot

# Verify GPUs
nvidia-smi

# Run CUDA tests
cd bin/worker-orcd/cuda/build
./cuda_tests

# Build worker
cd ../..
cargo build --release

# Download test model
mkdir -p ~/models
cd ~/models
wget https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf

# Run worker
cd ~/llama-orch/bin/worker-orcd
./target/release/worker-orcd --model ~/models/qwen2.5-0.5b-instruct-q4_k_m.gguf --gpu 0
```

### CPU-Only Development

For development without GPU:

```bash
./tools/setup-dev-workstation.sh --skip-nvidia

# All Rust tests will still pass
cd bin/worker-orcd
cargo test
```

### CI/CD Integration

For automated testing in CI/CD pipelines:

```bash
# Install only what's needed for testing
./tools/setup-dev-workstation.sh --skip-nvidia --skip-tests

# Run tests separately
cd bin/worker-orcd
cargo test --lib
cargo test --test '*'
```

---

## 🔧 Troubleshooting

### Script fails with "command not found"

Make sure the script is executable:
```bash
chmod +x tools/setup-dev-workstation.sh
```

### NVIDIA driver installation fails

Check if GPU is detected:
```bash
lspci | grep -i nvidia
```

If no GPU is found, use `--skip-nvidia` flag.

### Rust installation fails

Manually install Rust:
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env
```

Then run script with `--skip-rust`:
```bash
./tools/setup-dev-workstation.sh --skip-rust
```

### Tests fail after installation

Check system requirements:
```bash
# Verify Rust
rustc --version
cargo --version

# Verify CUDA (if installed)
nvcc --version
nvidia-smi

# Verify build tools
cmake --version
gcc --version
```

---

## 📚 Additional Resources

- [Worker-orcd README](../bin/worker-orcd/README.md)
- [CUDA Setup Guide](../bin/worker-orcd/CUDA_TEST_SETUP.md)
- [Test Reports](../bin/worker-orcd/FINAL_TEST_REPORT.md)
- [Contributing Guide](../CONTRIBUTING.md)

---

## 🤝 Contributing

To add new tools to this directory:

1. Create a new subdirectory or script
2. Add documentation to this README
3. Include usage examples
4. Add to CI/CD if applicable

---

**Last Updated**: 2025-10-05  
**Maintainer**: llama-orch team

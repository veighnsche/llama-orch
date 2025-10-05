# CUDA Test Setup Guide

This guide explains how to install CUDA and run the complete test suite including CUDA C++ tests.

---

## Prerequisites

- Ubuntu 24.04 LTS
- NVIDIA GPU (RTX 3090 / RTX 3060 detected)
- Sudo access

---

## Step 1: Install NVIDIA CUDA Toolkit

### Option A: Ubuntu Package (Recommended for Testing)

```bash
# Install CUDA toolkit from Ubuntu repositories
sudo apt update
sudo apt install -y nvidia-cuda-toolkit

# Verify installation
nvcc --version
```

**Expected Output**: CUDA compilation tools, release 12.x

### Option B: NVIDIA Official Repository (Latest Version)

```bash
# Download and install CUDA repository package
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb

# Update package list
sudo apt update

# Install CUDA toolkit
sudo apt install -y cuda-toolkit-13-0

# Add CUDA to PATH
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# Verify installation
nvcc --version
```

---

## Step 2: Install Additional Dependencies

```bash
# Install Google Test for C++ testing
sudo apt install -y libgtest-dev

# Build and install Google Test
cd /usr/src/gtest
sudo cmake .
sudo make
sudo cp lib/*.a /usr/lib
```

---

## Step 3: Build CUDA Tests

```bash
# Navigate to worker-orcd directory
cd /home/vince/Projects/llama-orch/bin/worker-orcd

# Create build directory
mkdir -p cuda/build
cd cuda/build

# Configure with CMake
cmake ..

# Build tests
make -j$(nproc)

# Verify build
ls -lh cuda_tests
```

**Expected Output**: `cuda_tests` executable created

---

## Step 4: Run CUDA Tests

### Run All CUDA Tests

```bash
cd /home/vince/Projects/llama-orch/bin/worker-orcd/cuda/build

# Run all tests
./cuda_tests

# Or use ctest
ctest --verbose
```

**Expected**: 254 tests passing

### Run Specific Test Suites

```bash
# Foundation Team Sprint 3 tests
./cuda_tests --gtest_filter="*VRAM*:*FFI*:*Device*:*Health*:*Embedding*:*cuBLAS*"

# Foundation Team Sprint 4 tests (Advanced Sampling)
./cuda_tests --gtest_filter="*TopK*:*TopP*:*Repetition*:*StopSequence*:*MinP*:*RNG*"

# Run with detailed output
./cuda_tests --gtest_color=yes
```

---

## Step 5: Run Complete Test Suite

### All Tests (Rust + CUDA)

```bash
cd /home/vince/Projects/llama-orch/bin/worker-orcd

# Run Rust library tests
cargo test --lib --no-fail-fast

# Run Rust integration tests
cargo test --test '*' --no-fail-fast

# Run CUDA tests
cd cuda/build && ./cuda_tests && cd ../..

# Summary
echo "✅ All tests complete!"
```

### Using Test Scripts

```bash
# Foundation Team Sprint 1 tests (HTTP/FFI)
cd cuda && ./run_sprint1_tests.sh

# Llama Team Sprint 2 tests (Tokenizer)
cd .. && ./run_sprint2_tests.sh

# Llama Team Sprint 3 tests (Kernels)
./run_sprint3_tests.sh
```

---

## Expected Test Results

### CUDA C++ Tests (254 tests)

| Component | Tests | Duration |
|-----------|-------|----------|
| FFI Interface | 9 | ~0.5s |
| Error Handling | 39 | ~1.0s |
| Context Lifecycle | 18 | ~1.5s |
| Health Verification | 13 | ~0.8s |
| VRAM Tracker | 13 | ~0.7s |
| Device Memory RAII | 33 | ~1.2s |
| Embedding Kernel | 10 | ~2.0s |
| cuBLAS Wrapper | 15 | ~3.0s |
| Sampling Kernels | 54 | ~2.5s |
| Seeded RNG | 14 | ~0.5s |
| Integration | 3 | ~1.0s |
| **TOTAL** | **254** | **~15s** |

### Combined Test Results

- **Rust Library**: 266 tests (~0.2s)
- **Rust Integration**: 213 tests (~0.5s)
- **CUDA C++**: 254 tests (~15s)
- **TOTAL**: **733 tests** (~16s)

---

## Troubleshooting

### Issue: `nvcc: command not found`

**Solution**: CUDA not in PATH
```bash
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

### Issue: `CUDA driver version is insufficient`

**Solution**: Update NVIDIA driver
```bash
sudo apt install -y nvidia-driver-550
sudo reboot
```

### Issue: `libcudart.so: cannot open shared object file`

**Solution**: Add CUDA libraries to LD_LIBRARY_PATH
```bash
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
sudo ldconfig
```

### Issue: CMake can't find CUDA

**Solution**: Set CUDA_PATH
```bash
export CUDA_PATH=/usr/local/cuda
cmake -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda ..
```

### Issue: Tests fail with "no CUDA-capable device"

**Solution**: Check GPU detection
```bash
nvidia-smi
lspci | grep -i nvidia
```

---

## Verification Checklist

- [ ] CUDA toolkit installed (`nvcc --version` works)
- [ ] NVIDIA driver installed (`nvidia-smi` works)
- [ ] Google Test installed
- [ ] CMake build succeeds
- [ ] `cuda_tests` executable created
- [ ] All 254 CUDA tests pass
- [ ] All 479 Rust tests pass
- [ ] Total: 733 tests passing

---

## Performance Notes

### GPU Utilization
- Tests use minimal GPU memory (~2GB)
- Can run on RTX 3060 (12GB) or RTX 3090 (24GB)
- Tests run sequentially to avoid context conflicts

### Build Time
- First build: ~2 minutes
- Incremental builds: ~10 seconds
- Test execution: ~16 seconds total

---

## Next Steps After CUDA Tests Pass

1. **Run with Real Models**
   ```bash
   # Download test model
   wget https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf
   
   # Run ignored tests
   cargo test -- --ignored
   ```

2. **Performance Benchmarks**
   ```bash
   cargo bench
   ```

3. **Integration with Pool Manager**
   ```bash
   # Start worker
   ./target/release/worker-orcd --gpu 0
   
   # Test inference
   curl -X POST http://localhost:8080/execute \
     -H "Content-Type: application/json" \
     -d '{"prompt": "Hello, world!", "max_tokens": 50}'
   ```

---

## Additional Resources

- [CUDA Installation Guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/)
- [Worker-orcd README](./README.md)
- [Test Report](./TEST_RUN_REPORT.md)
- [CUDA Feature Documentation](./CUDA_FEATURE.md)

---

**Last Updated**: 2025-10-05  
**Tested On**: Ubuntu 24.04.3 LTS, CUDA 13.0, RTX 3090

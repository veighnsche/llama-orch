# TEAM-005: Task 3.10 - Build and Verify
**Part of:** Phase 3 - Implementation  
**Duration:** 15 minutes  
**Status:** ⏳ PENDING  
**Depends on:** Task 3.9 (Checkpoint 6)

---

## Objective

Build llama.cpp with checkpoint support and verify all instrumentation works.

**Goal:** Confirm clean build, test checkpoint extraction, validate output files.

---

## Prerequisites

- All tasks 3.1-3.9 completed
- All 6 checkpoints instrumented
- CMakeLists.txt updated
- llama-checkpoint.h created

---

## Build Steps

### 1. Clean Build

```bash
cd /home/vince/Projects/llama-orch/reference/llama.cpp

# Remove old build
rm -rf build-validate

# Create fresh build directory
mkdir build-validate
cd build-validate

# Configure with checkpoint support
cmake .. -DLLORCH_VALIDATE=ON

# Verify configuration
grep "TEAM-005" ../CMakeLists.txt
grep "LLORCH_VALIDATE" CMakeCache.txt
```

**Expected output:**
```
-- TEAM-005: Checkpoint extraction enabled
-- TEAM-005: Checkpoints will be saved when LLORCH_VALIDATE=1 env var is set
LLORCH_VALIDATE:BOOL=ON
```

### 2. Compile

```bash
# Build with all cores
make -j$(nproc) 2>&1 | tee build.log

# Check for errors
echo "=== Checking for errors ==="
grep -i "error" build.log | grep -v "error.h" | grep -v "std::error"

# Check for warnings (optional)
echo "=== Checking for warnings ==="
grep -i "warning" build.log | head -20
```

**Expected:** No errors, binaries created successfully.

### 3. Verify Binaries

```bash
# Check that binaries exist
ls -lh bin/llama-cli
ls -lh bin/llama-server

# Verify checkpoint code is compiled in
strings bin/llama-cli | grep "TEAM-005"
# Should see: "TEAM-005: Checkpoint Extraction Enabled"
```

---

## Runtime Verification

### 1. Test Without Environment Variable

```bash
# Run without LLORCH_VALIDATE set
./bin/llama-cli --version

# Should NOT see checkpoint banner
# Should work normally
```

### 2. Test With Environment Variable

```bash
# Set environment variables
export LLORCH_VALIDATE=1
export LLORCH_CHECKPOINT_DIR=/tmp/llama_cpp_checkpoints
mkdir -p $LLORCH_CHECKPOINT_DIR

# Run with checkpoint extraction
./bin/llama-cli --version

# Should see banner:
# ╔══════════════════════════════════════════════════════════════╗
# ║  TEAM-005: Checkpoint Extraction Enabled                     ║
# ║  Directory: /tmp/llama_cpp_checkpoints                       ║
# ╚══════════════════════════════════════════════════════════════╝
```

### 3. Test Full Checkpoint Extraction

**Note:** Requires GPT-2 model file. If not available, skip to verification checklist.

```bash
# Download GPT-2 model (if needed)
# wget https://huggingface.co/ggml-org/models/resolve/main/gpt2-117M/ggml-model-f16.gguf

# Run inference with checkpoint extraction
export LLORCH_VALIDATE=1
export LLORCH_CHECKPOINT_DIR=/tmp/llama_cpp_checkpoints
rm -rf $LLORCH_CHECKPOINT_DIR/*
mkdir -p $LLORCH_CHECKPOINT_DIR

./bin/llama-cli \
  --model /path/to/gpt2.gguf \
  --prompt "Hello world" \
  --n-predict 1 \
  2>&1 | tee run.log

# Check for checkpoint messages
grep "TEAM-005" run.log
```

**Expected output:**
```
╔══════════════════════════════════════════════════════════════╗
║  TEAM-005: Checkpoint Extraction Enabled                     ║
║  Directory: /tmp/llama_cpp_checkpoints                       ║
╚══════════════════════════════════════════════════════════════╝

✅ TEAM-005: checkpoint_01_ln1_output [2 × 768] → /tmp/llama_cpp_checkpoints/checkpoint_01_ln1_output.bin
✅ TEAM-005: checkpoint_02_q [64 × 12 × 2] → /tmp/llama_cpp_checkpoints/checkpoint_02_q.bin
✅ TEAM-005: checkpoint_02_k [64 × 12 × 2] → /tmp/llama_cpp_checkpoints/checkpoint_02_k.bin
✅ TEAM-005: checkpoint_02_v [64 × 12 × 2] → /tmp/llama_cpp_checkpoints/checkpoint_02_v.bin
✅ TEAM-005: checkpoint_03_cache_k [64 × 12 × N × 1] → /tmp/llama_cpp_checkpoints/checkpoint_03_cache_k.bin
✅ TEAM-005: checkpoint_03_cache_v [64 × 12 × N × 1] → /tmp/llama_cpp_checkpoints/checkpoint_03_cache_v.bin
✅ TEAM-005: checkpoint_04_scores [N × 2 × 12 × 1] → /tmp/llama_cpp_checkpoints/checkpoint_04_scores.bin
✅ TEAM-005: checkpoint_05_output [768 × 2 × 1] → /tmp/llama_cpp_checkpoints/checkpoint_05_output.bin
✅ TEAM-005: checkpoint_06_ffn [2 × 768] → /tmp/llama_cpp_checkpoints/checkpoint_06_ffn.bin

╔══════════════════════════════════════════════════════════════╗
║  TEAM-005: Checkpoint Extraction Complete                    ║
║  Files saved to: /tmp/llama_cpp_checkpoints                  ║
╚══════════════════════════════════════════════════════════════╝
```

### 4. Verify Checkpoint Files

```bash
# List all checkpoint files
ls -lh $LLORCH_CHECKPOINT_DIR/

# Count files (should be 9: 1 + 3 + 2 + 1 + 1 + 1)
ls $LLORCH_CHECKPOINT_DIR/*.bin | wc -l

# Check file sizes
du -h $LLORCH_CHECKPOINT_DIR/*.bin
```

**Expected files:**
- `checkpoint_01_ln1_output.bin` (~6KB)
- `checkpoint_02_q.bin` (~6KB)
- `checkpoint_02_k.bin` (~6KB)
- `checkpoint_02_v.bin` (~6KB)
- `checkpoint_03_cache_k.bin` (varies with cache size)
- `checkpoint_03_cache_v.bin` (varies with cache size)
- `checkpoint_04_scores.bin` (varies with cache size)
- `checkpoint_05_output.bin` (~6KB)
- `checkpoint_06_ffn.bin` (~6KB)

### 5. Validate Binary Format

```python
#!/usr/bin/env python3
"""Validate checkpoint binary format."""
import struct
import numpy as np
import sys
from pathlib import Path

def validate_checkpoint(filepath):
    """Validate a single checkpoint file."""
    print(f"\n{'='*60}")
    print(f"Validating: {filepath.name}")
    print(f"{'='*60}")
    
    with open(filepath, 'rb') as f:
        # Read dimensions
        n_dims = struct.unpack('i', f.read(4))[0]
        print(f"Dimensions: {n_dims}")
        
        # Read shape
        shape = struct.unpack(f'{n_dims}q', f.read(8 * n_dims))
        print(f"Shape: {shape}")
        
        # Read data
        data = np.frombuffer(f.read(), dtype=np.float32)
        print(f"Data elements: {len(data)}")
        
        # Verify shape matches data
        expected_elements = np.prod(shape)
        if len(data) == expected_elements:
            print(f"✅ Shape matches data ({expected_elements} elements)")
        else:
            print(f"❌ Shape mismatch: expected {expected_elements}, got {len(data)}")
            return False
        
        # Check for NaN/Inf
        if np.isnan(data).any():
            print(f"⚠️  Contains NaN values")
        if np.isinf(data).any():
            print(f"⚠️  Contains Inf values")
        
        # Basic statistics
        print(f"Min: {data.min():.6f}")
        print(f"Max: {data.max():.6f}")
        print(f"Mean: {data.mean():.6f}")
        print(f"Std: {data.std():.6f}")
        
        return True

# Validate all checkpoints
checkpoint_dir = Path("/tmp/llama_cpp_checkpoints")
if not checkpoint_dir.exists():
    print(f"❌ Checkpoint directory not found: {checkpoint_dir}")
    sys.exit(1)

files = sorted(checkpoint_dir.glob("checkpoint_*.bin"))
if not files:
    print(f"❌ No checkpoint files found in {checkpoint_dir}")
    sys.exit(1)

print(f"Found {len(files)} checkpoint files")

all_valid = True
for filepath in files:
    if not validate_checkpoint(filepath):
        all_valid = False

print(f"\n{'='*60}")
if all_valid:
    print("✅ All checkpoints validated successfully")
else:
    print("❌ Some checkpoints failed validation")
    sys.exit(1)
```

Save as `validate_checkpoints.py` and run:
```bash
python3 validate_checkpoints.py
```

---

## Success Criteria

### Build
- [ ] Clean build completes without errors
- [ ] No compilation warnings related to TEAM-005 code
- [ ] TEAM-005 cmake messages appear
- [ ] Binaries created (llama-cli, llama-server)
- [ ] Checkpoint code present in binaries (strings check)

### Runtime (without LLORCH_VALIDATE)
- [ ] Runs normally without checkpoint extraction
- [ ] No checkpoint banner appears
- [ ] No performance impact

### Runtime (with LLORCH_VALIDATE=1)
- [ ] Checkpoint banner appears at startup
- [ ] All 9 checkpoint files created
- [ ] Completion banner appears at end
- [ ] No crashes or errors

### File Validation
- [ ] All checkpoint files exist
- [ ] File sizes are reasonable
- [ ] Binary format is correct (dims + shape + data)
- [ ] No NaN or Inf values
- [ ] Shapes match expectations

---

## Troubleshooting

### Build Issues

**Issue:** CMake doesn't find LLORCH_VALIDATE option
- **Solution:** Check CMakeLists.txt was modified correctly
- **Solution:** Try `cmake .. -DLLORCH_VALIDATE=ON -DCMAKE_BUILD_TYPE=Release`

**Issue:** Compilation errors in llama-checkpoint.h
- **Solution:** Verify header syntax (missing semicolons, brackets)
- **Solution:** Check includes are correct

**Issue:** Undefined reference to checkpoint functions
- **Solution:** Verify functions are `inline` in header
- **Solution:** Check `#include "llama-checkpoint.h"` is inside `#ifdef`

### Runtime Issues

**Issue:** Banner doesn't appear
- **Solution:** Verify LLORCH_VALIDATE=1 is set (not just cmake flag)
- **Solution:** Check binary was built with -DLLORCH_VALIDATE=ON

**Issue:** No checkpoint files created
- **Solution:** Check directory exists and is writable
- **Solution:** Verify `il == 0` filters are working
- **Solution:** Ensure GPT-2 model is being used

**Issue:** Wrong number of files
- **Solution:** Check that all 6 checkpoints are instrumented
- **Solution:** Verify each checkpoint has correct `il == 0` filter

**Issue:** Crash during extraction
- **Solution:** Check for null pointer dereferences
- **Solution:** Verify tensor variables are valid
- **Solution:** Add more error checking in save_tensor

---

## Performance Notes

**With LLORCH_VALIDATE=1:**
- Expect ~10-20% slowdown due to file I/O
- Disk space: ~50-100KB per inference
- Memory: Minimal overhead (temporary buffers)

**Without LLORCH_VALIDATE:**
- Zero overhead (code not compiled in)
- No disk usage
- No memory overhead

---

## Next Steps

After successful verification:
1. Proceed to Phase 4 (Testing)
2. Compare checkpoints with PyTorch reference
3. Validate numerical accuracy
4. Document any discrepancies

---

**Status:** ⏳ PENDING  
**Assigned to:** TEAM-005  
**Estimated time:** 15 minutes  
**Actual time:** [fill after completion]

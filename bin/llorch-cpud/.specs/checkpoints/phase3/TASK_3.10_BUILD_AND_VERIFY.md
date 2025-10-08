# TEAM-006: Task 3.10 - Build and Verify
**Part of:** Phase 3 - Implementation  
**Duration:** 15 minutes  
**Status:** ⏳ READY (REVISED BY TEAM-005)  
**Depends on:** Task 3.9 (Checkpoint 6)  
**Updated by:** TEAM-006

---

## ✅ APPROACH REVISED BY TEAM-005

**Old (OBSOLETE):** Build llama.cpp with conditional compilation  
**New (CORRECT):** Build wrapper tool and add 3 minimal callbacks to llama.cpp

See [COMPREHENSIVE_ANALYSIS.md](COMPREHENSIVE_ANALYSIS.md) and [HANDOFF_TO_TEAM_006.md](HANDOFF_TO_TEAM_006.md) for full details.

---

## Objective

Build wrapper tool, add minimal llama.cpp callbacks, and verify checkpoint extraction.

**Goal:** Confirm wrapper builds, 3 callbacks added, all 9 checkpoints extracted.

---

## Prerequisites

- Tasks 3.1-3.3 completed (wrapper tool created) ✅
- Tasks 3.4-3.9 completed (callbacks verified/added) ✅
- llama.cpp built ✅

---

## Build Steps

### 1. Rebuild llama.cpp (with 3 callbacks added)

```bash
cd /home/vince/Projects/llama-orch/reference/llama.cpp

# Rebuild llama.cpp
cd build
make -j$(nproc)

# Verify it built successfully
ls -lh lib/libllama.so  # or .a depending on build config
```

### 2. Build Wrapper Tool

```bash
cd /home/vince/Projects/llama-orch/bin/llorch-cpud/tools/checkpoint-extractor

# Create build directory
mkdir -p build
cd build

# Configure (point to llama.cpp)
cmake .. \
  -DCMAKE_PREFIX_PATH=/home/vince/Projects/llama-orch/reference/llama.cpp/build

# Build
make -j$(nproc)

# Verify binary exists
ls -lh llorch-checkpoint-extractor
```

**Expected:** Binary created successfully.

### 3. Verify Callbacks Added

```bash
cd /home/vince/Projects/llama-orch/reference/llama.cpp

# Check KV cache callbacks (2 lines)
grep -A1 'get_v(ctx0, il)' src/llama-graph.cpp | grep 'cb.*cache'

# Check attention output callback (1 line)
grep 'cb.*attn_out_proj' src/llama-graph.cpp
```

**Expected:**
```
cb(k, "cache_k", il);
cb(v, "cache_v", il);
cb(cur, "attn_out_proj", il);
```

---

## Runtime Verification

### 1. Test Wrapper Tool

```bash
cd /home/vince/Projects/llama-orch/bin/llorch-cpud/tools/checkpoint-extractor/build

# Run wrapper tool (requires GPT-2 model)
./llorch-checkpoint-extractor \
  /path/to/gpt2.gguf \
  "Hello world" \
  /tmp/checkpoints

# Should see extraction happening
```

**Expected output:**
```
╔══════════════════════════════════════════════════════════════╗
║  TEAM-006: Checkpoint Extraction Enabled                     ║
║  Output: /tmp/checkpoints                                    ║
╚══════════════════════════════════════════════════════════════╝

Tokenized prompt: 2 tokens
✅ TEAM-006: attn_norm [2 × 768] → /tmp/checkpoints/checkpoint_attn_norm.bin
✅ TEAM-006: Qcur [64 × 12 × 2] → /tmp/checkpoints/checkpoint_Qcur.bin
✅ TEAM-006: Kcur [64 × 12 × 2] → /tmp/checkpoints/checkpoint_Kcur.bin
✅ TEAM-006: Vcur [64 × 12 × 2] → /tmp/checkpoints/checkpoint_Vcur.bin
✅ TEAM-006: cache_k [64 × 12 × N × 1] → /tmp/checkpoints/checkpoint_cache_k.bin
✅ TEAM-006: cache_v [64 × 12 × N × 1] → /tmp/checkpoints/checkpoint_cache_v.bin
✅ TEAM-006: kq_soft_max [N × 2 × 12 × 1] → /tmp/checkpoints/checkpoint_kq_soft_max.bin
✅ TEAM-006: attn_out_proj [768 × 2 × 1] → /tmp/checkpoints/checkpoint_attn_out_proj.bin
✅ TEAM-006: ffn_out [2 × 768] → /tmp/checkpoints/checkpoint_ffn_out.bin

╔══════════════════════════════════════════════════════════════╗
║  TEAM-006: Extraction Complete                               ║
║  Extracted 9 checkpoints                                     ║
╚══════════════════════════════════════════════════════════════╝
```

### 2. Verify Checkpoint Files

```bash
# List all checkpoint files
ls -lh /tmp/checkpoints/

# Count files (should be 9)
ls /tmp/checkpoints/*.bin | wc -l
```

**Expected files:**
- `checkpoint_attn_norm.bin` (~6KB)
- `checkpoint_Qcur.bin` (~6KB)
- `checkpoint_Kcur.bin` (~6KB)
- `checkpoint_Vcur.bin` (~6KB)
- `checkpoint_cache_k.bin` (varies with cache size)
- `checkpoint_cache_v.bin` (varies with cache size)
- `checkpoint_kq_soft_max.bin` (varies with cache size)
- `checkpoint_attn_out_proj.bin` (~6KB)
- `checkpoint_ffn_out.bin` (~6KB)

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
- [ ] llama.cpp rebuilt successfully
- [ ] Wrapper tool builds without errors
- [ ] Binary `llorch-checkpoint-extractor` created
- [ ] 3 callbacks added to llama.cpp (verified)

### Callbacks Added
- [ ] `cb(k, "cache_k", il)` added after `get_k()` 
- [ ] `cb(v, "cache_v", il)` added after `get_v()`
- [ ] `cb(cur, "attn_out_proj", il)` added before return in `build_attn`

### Runtime
- [ ] Wrapper tool runs without crashes
- [ ] Banner appears at startup
- [ ] All 9 checkpoint files created
- [ ] Completion banner shows count
- [ ] No errors during extraction

### File Validation
- [ ] All 9 checkpoint files exist
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

## Summary of Changes

### What Was Added (3 callbacks total)

1. **KV Cache (2 callbacks)** - `src/llama-graph.cpp` line ~1553-1554:
   ```cpp
   cb(k, "cache_k", il);
   cb(v, "cache_v", il);
   ```

2. **Attention Output (1 callback)** - `src/llama-graph.cpp` line ~1574:
   ```cpp
   cb(cur, "attn_out_proj", il);
   ```

### What Already Existed (6 callbacks)

- `cb(cur, "attn_norm", il)` - LayerNorm ✅
- `cb(Qcur, "Qcur", il)` - Q projection ✅
- `cb(Kcur, "Kcur", il)` - K projection ✅
- `cb(Vcur, "Vcur", il)` - V projection ✅
- `cb(kq, "kq_soft_max", il)` - Attention scores ✅
- `cb(cur, "ffn_out", il)` - FFN output ✅

### Wrapper Tool Created

- `bin/llorch-cpud/tools/checkpoint-extractor/` ✅
- Registers eval callback ✅
- Extracts all 9 checkpoints ✅

---

## Next Steps

After successful verification:
1. **Phase 4:** Compare checkpoints with PyTorch reference
2. Validate numerical accuracy
3. Document any discrepancies
4. Create validation report

---

**Status:** ⏳ READY  
**Assigned to:** TEAM-006  
**Estimated time:** 15 minutes  
**Actual time:** [fill after completion]

**Updated by TEAM-006 based on TEAM-005 comprehensive analysis**

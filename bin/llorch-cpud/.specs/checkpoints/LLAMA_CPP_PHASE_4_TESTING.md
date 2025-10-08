# TEAM-004: Phase 4 - Build and Test
**Part of:** llama.cpp Instrumentation Master Plan  
**Duration:** 1 hour  
**Status:** ⏳ PENDING  
**Depends on:** Phase 3 (Implementation) must be complete

---

## Objective

Verify instrumentation works and produces correct checkpoints with expected shapes.

**Goal:** Generate checkpoint files from llama.cpp and verify they match our expectations.

---

## Step 4.1: Download GPT-2 Model (10 min)

### Check if Model Exists

```bash
cd /home/vince/Projects/llama-orch/reference/llama.cpp

# Check for existing GPT-2 model
ls -lh models/gpt2*.gguf 2>/dev/null || echo "Model not found"
```

### Download GPT-2 in GGUF Format

**Option A: Using llama.cpp scripts**
```bash
cd /home/vince/Projects/llama-orch/reference/llama.cpp

# Check if download script exists
if [ -f "scripts/download-gguf.sh" ]; then
    ./scripts/download-gguf.sh gpt2
else
    echo "Download script not found, use Option B"
fi
```

**Option B: Using huggingface-cli**
```bash
# Install huggingface-cli if needed
pip3 install --user huggingface-hub

# Download GPT-2 GGUF
cd /home/vince/Projects/llama-orch/reference/llama.cpp
mkdir -p models
cd models

huggingface-cli download ggml-org/gpt-2 gpt2-f32.gguf --local-dir . --local-dir-use-symlinks False
```

**Option C: Manual download**
```bash
cd /home/vince/Projects/llama-orch/reference/llama.cpp/models

# Download from HuggingFace
wget https://huggingface.co/ggml-org/gpt-2/resolve/main/gpt2-f32.gguf
```

### Verify Model Downloaded

```bash
cd /home/vince/Projects/llama-orch/reference/llama.cpp

# Check model file
ls -lh models/gpt2*.gguf

# Verify it's a valid GGUF file
file models/gpt2*.gguf
```

**Expected:** File exists, size ~500MB-1GB, type shows GGUF format

### Checklist

- [ ] GPT-2 model downloaded
- [ ] File size reasonable (~500MB-1GB)
- [ ] File format verified (GGUF)
- [ ] Model accessible in `models/` directory

---

## Step 4.2: Run with Checkpoint Extraction (10 min)

### Clean Checkpoint Directory

```bash
# TEAM-004: Clean previous checkpoints
rm -rf /tmp/llama_cpp_checkpoints
mkdir -p /tmp/llama_cpp_checkpoints

# Verify directory is empty
ls -la /tmp/llama_cpp_checkpoints/
```

### Run llama.cpp with Checkpoints

```bash
cd /home/vince/Projects/llama-orch/reference/llama.cpp

# TEAM-004: Run with checkpoint extraction enabled
LLORCH_VALIDATE=1 ./build/bin/llama-cli \
    -m models/gpt2-f32.gguf \
    -p "Hello world" \
    -n 1 \
    --no-display-prompt \
    2>&1 | tee checkpoint_run.log
```

### Expected Output

```
╔══════════════════════════════════════════════════════════╗
║  TEAM-004: Checkpoint Extraction Enabled                 ║
║  Directory: /tmp/llama_cpp_checkpoints                   ║
╚══════════════════════════════════════════════════════════╝

[... model loading ...]

✅ TEAM-004: Checkpoint checkpoint_01_ln1_output saved [2x768]
✅ TEAM-004: Checkpoint checkpoint_02_q saved [2x768]
✅ TEAM-004: Checkpoint checkpoint_02_k saved [2x768]
✅ TEAM-004: Checkpoint checkpoint_02_v saved [2x768]
✅ TEAM-004: Checkpoint checkpoint_04_scores saved [12x2x2]
✅ TEAM-004: Checkpoint checkpoint_05_output saved [2x768]
✅ TEAM-004: Checkpoint checkpoint_06_ffn saved [2x768]

[... generation output ...]

╔══════════════════════════════════════════════════════════╗
║  TEAM-004: Checkpoint Extraction Complete                ║
╚══════════════════════════════════════════════════════════╝
```

### Verify Checkpoints Created

```bash
# TEAM-004: List checkpoint files
ls -lh /tmp/llama_cpp_checkpoints/

# Count checkpoint files
ls /tmp/llama_cpp_checkpoints/*.bin | wc -l
```

**Expected:** 7 files (or 9 if checkpoint 3 has separate K and V files)

### Checklist

- [ ] Checkpoint directory cleaned
- [ ] llama-cli runs without errors
- [ ] TEAM-004 init message appears
- [ ] All checkpoint save messages appear
- [ ] TEAM-004 finalize message appears
- [ ] Checkpoint files created

---

## Step 4.3: Create Conversion Script (15 min)

### Create Python Converter

**File:** `bin/llorch-cpud/.test_helpers/convert_llama_cpp_checkpoints.py`

```python
#!/usr/bin/env python3
"""
TEAM-004: Convert llama.cpp binary checkpoints to NumPy format

Usage:
    python3 convert_llama_cpp_checkpoints.py

Input:  /tmp/llama_cpp_checkpoints/*.bin
Output: .test-models/gpt2/extracted_weights/*_llama_cpp.npy
"""

import numpy as np
import struct
import sys
from pathlib import Path

def load_llama_cpp_checkpoint(filepath):
    """
    Load checkpoint from llama.cpp binary format
    
    Format:
        int32: number of dimensions
        int64[n_dims]: shape
        float32[n_elements]: data
    """
    with open(filepath, 'rb') as f:
        # Read number of dimensions
        n_dims_bytes = f.read(4)
        if len(n_dims_bytes) != 4:
            raise ValueError(f"Failed to read n_dims from {filepath}")
        n_dims = struct.unpack('i', n_dims_bytes)[0]
        
        # Read shape
        shape = []
        for _ in range(n_dims):
            dim_bytes = f.read(8)
            if len(dim_bytes) != 8:
                raise ValueError(f"Failed to read dimension from {filepath}")
            dim = struct.unpack('q', dim_bytes)[0]
            shape.append(dim)
        
        # Read data
        n_elements = int(np.prod(shape))
        data = np.fromfile(f, dtype=np.float32, count=n_elements)
        
        if len(data) != n_elements:
            raise ValueError(f"Expected {n_elements} elements, got {len(data)}")
        
        # Reshape
        return data.reshape(shape)

def convert_all_checkpoints():
    """Convert all checkpoints from llama.cpp to NumPy"""
    input_dir = Path("/tmp/llama_cpp_checkpoints")
    output_dir = Path("/home/vince/Projects/llama-orch/.test-models/gpt2/extracted_weights")
    
    # TEAM-004: Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # TEAM-004: Checkpoint files to convert
    checkpoints = [
        "checkpoint_01_ln1_output",
        "checkpoint_02_q",
        "checkpoint_02_k",
        "checkpoint_02_v",
        "checkpoint_03_cache_k",  # May not exist
        "checkpoint_03_cache_v",  # May not exist
        "checkpoint_04_scores",
        "checkpoint_05_output",
        "checkpoint_06_ffn",
    ]
    
    print("TEAM-004: Converting llama.cpp checkpoints to NumPy\n")
    
    converted = 0
    missing = 0
    failed = 0
    
    for checkpoint in checkpoints:
        input_file = input_dir / f"{checkpoint}.bin"
        output_file = output_dir / f"{checkpoint}_llama_cpp.npy"
        
        if not input_file.exists():
            print(f"⚠️  Missing: {checkpoint}")
            missing += 1
            continue
        
        try:
            data = load_llama_cpp_checkpoint(input_file)
            np.save(output_file, data)
            print(f"✅ Converted: {checkpoint:30s} shape={data.shape}")
            converted += 1
        except Exception as e:
            print(f"❌ Failed: {checkpoint:30s} error={e}")
            failed += 1
    
    print(f"\nTEAM-004: Conversion complete")
    print(f"  Converted: {converted}")
    print(f"  Missing:   {missing}")
    print(f"  Failed:    {failed}")
    
    return failed == 0

if __name__ == "__main__":
    success = convert_all_checkpoints()
    sys.exit(0 if success else 1)
```

### Make Script Executable

```bash
chmod +x /home/vince/Projects/llama-orch/bin/llorch-cpud/.test_helpers/convert_llama_cpp_checkpoints.py
```

### Checklist

- [ ] Conversion script created
- [ ] Script is executable
- [ ] TEAM-004 signatures present
- [ ] Error handling included

---

## Step 4.4: Convert Checkpoints (5 min)

### Run Conversion

```bash
cd /home/vince/Projects/llama-orch/bin/llorch-cpud/.test_helpers

# TEAM-004: Convert checkpoints
python3 convert_llama_cpp_checkpoints.py
```

### Expected Output

```
TEAM-004: Converting llama.cpp checkpoints to NumPy

✅ Converted: checkpoint_01_ln1_output       shape=(2, 768)
✅ Converted: checkpoint_02_q                shape=(2, 768)
✅ Converted: checkpoint_02_k                shape=(2, 768)
✅ Converted: checkpoint_02_v                shape=(2, 768)
⚠️  Missing: checkpoint_03_cache_k
⚠️  Missing: checkpoint_03_cache_v
✅ Converted: checkpoint_04_scores           shape=(12, 2, 2)
✅ Converted: checkpoint_05_output           shape=(2, 768)
✅ Converted: checkpoint_06_ffn              shape=(2, 768)

TEAM-004: Conversion complete
  Converted: 7
  Missing:   2
  Failed:    0
```

### Verify NumPy Files

```bash
cd /home/vince/Projects/llama-orch

# List converted files
ls -lh .test-models/gpt2/extracted_weights/*_llama_cpp.npy
```

### Checklist

- [ ] Conversion script runs without errors
- [ ] All available checkpoints converted
- [ ] NumPy files created in correct directory
- [ ] No failed conversions

---

## Step 4.5: Verify Checkpoint Shapes (10 min)

### Create Verification Script

```bash
cd /home/vince/Projects/llama-orch

python3 << 'EOF'
import numpy as np
from pathlib import Path

# TEAM-004: Verify checkpoint shapes
weights_dir = Path(".test-models/gpt2/extracted_weights")

checkpoints = {
    "checkpoint_01_ln1_output": (2, 768),
    "checkpoint_02_q": (2, 768),
    "checkpoint_02_k": (2, 768),
    "checkpoint_02_v": (2, 768),
    "checkpoint_04_scores": None,  # Shape may vary
    "checkpoint_05_output": (2, 768),
    "checkpoint_06_ffn": (2, 768),
}

print("TEAM-004: Checkpoint Shape Verification\n")
print(f"{'Checkpoint':<35} {'Expected':<15} {'Actual':<15} {'Status'}")
print("=" * 80)

all_good = True

for name, expected_shape in checkpoints.items():
    pytorch_file = weights_dir / f"{name}.npy"
    llama_cpp_file = weights_dir / f"{name}_llama_cpp.npy"
    
    if not llama_cpp_file.exists():
        print(f"{name:<35} {'N/A':<15} {'MISSING':<15} ⚠️")
        continue
    
    llama_data = np.load(llama_cpp_file)
    actual_shape = llama_data.shape
    
    # Check shape
    if expected_shape is None:
        status = "✅ (varies)"
    elif actual_shape == expected_shape:
        status = "✅"
    else:
        status = "❌ MISMATCH"
        all_good = False
    
    print(f"{name:<35} {str(expected_shape):<15} {str(actual_shape):<15} {status}")
    
    # Compare with PyTorch if available
    if pytorch_file.exists():
        pytorch_data = np.load(pytorch_file)
        
        if pytorch_data.shape != actual_shape:
            print(f"  ⚠️  Shape mismatch with PyTorch: {pytorch_data.shape}")
            all_good = False
        else:
            diff = np.abs(pytorch_data - llama_data).max()
            if diff < 1e-3:
                print(f"  ✅ vs PyTorch: max diff = {diff:.6e} (GOOD)")
            else:
                print(f"  ⚠️  vs PyTorch: max diff = {diff:.6e} (HIGH)")

print("\n" + "=" * 80)
if all_good:
    print("TEAM-004: ✅ All checkpoints verified")
else:
    print("TEAM-004: ⚠️  Some checkpoints have issues")
EOF
```

### Expected Output

```
TEAM-004: Checkpoint Shape Verification

Checkpoint                          Expected        Actual          Status
================================================================================
checkpoint_01_ln1_output            (2, 768)        (2, 768)        ✅
  ✅ vs PyTorch: max diff = 1.234567e-05 (GOOD)
checkpoint_02_q                     (2, 768)        (2, 768)        ✅
  ✅ vs PyTorch: max diff = 2.345678e-05 (GOOD)
checkpoint_02_k                     (2, 768)        (2, 768)        ✅
  ✅ vs PyTorch: max diff = 3.456789e-05 (GOOD)
checkpoint_02_v                     (2, 768)        (2, 768)        ✅
  ✅ vs PyTorch: max diff = 4.567890e-05 (GOOD)
checkpoint_04_scores                None            (12, 2, 2)      ✅ (varies)
  ✅ vs PyTorch: max diff = 5.678901e-05 (GOOD)
checkpoint_05_output                (2, 768)        (2, 768)        ✅
  ✅ vs PyTorch: max diff = 6.789012e-05 (GOOD)
checkpoint_06_ffn                   (2, 768)        (2, 768)        ✅
  ✅ vs PyTorch: max diff = 7.890123e-05 (GOOD)

================================================================================
TEAM-004: ✅ All checkpoints verified
```

### Checklist

- [ ] Verification script runs
- [ ] All shapes match expected values
- [ ] Differences with PyTorch are small (< 1e-3)
- [ ] No critical mismatches

---

## Step 4.6: Test Edge Cases (10 min)

### Test Without LLORCH_VALIDATE

```bash
cd /home/vince/Projects/llama-orch/reference/llama.cpp

# Run without checkpoint extraction
./build/bin/llama-cli \
    -m models/gpt2-f32.gguf \
    -p "Hello world" \
    -n 1 \
    --no-display-prompt \
    2>&1 | grep "TEAM-004"
```

**Expected:** No TEAM-004 messages (checkpoints disabled)

### Test with Different Prompt

```bash
# Clean checkpoints
rm -rf /tmp/llama_cpp_checkpoints/*

# Run with different prompt
LLORCH_VALIDATE=1 ./build/bin/llama-cli \
    -m models/gpt2-f32.gguf \
    -p "The quick brown fox" \
    -n 1 \
    --no-display-prompt \
    2>&1 | grep "✅ TEAM-004"
```

**Expected:** All checkpoints extracted successfully

### Test with Custom Directory

```bash
# Set custom checkpoint directory
export LLORCH_CHECKPOINT_DIR="/tmp/my_checkpoints"
mkdir -p "$LLORCH_CHECKPOINT_DIR"

LLORCH_VALIDATE=1 ./build/bin/llama-cli \
    -m models/gpt2-f32.gguf \
    -p "Test" \
    -n 1 \
    --no-display-prompt

# Verify checkpoints in custom directory
ls -lh "$LLORCH_CHECKPOINT_DIR"
```

**Expected:** Checkpoints in custom directory

### Checklist

- [ ] Works without LLORCH_VALIDATE (no checkpoints)
- [ ] Works with different prompts
- [ ] Works with custom checkpoint directory
- [ ] No crashes or errors in any test

---

## Completion Checklist

### Model Setup
- [ ] GPT-2 model downloaded
- [ ] Model file verified (GGUF format)
- [ ] Model accessible to llama-cli

### Checkpoint Extraction
- [ ] llama-cli runs with LLORCH_VALIDATE=1
- [ ] All checkpoint save messages appear
- [ ] Checkpoint files created
- [ ] File count matches expectations

### Conversion
- [ ] Conversion script created and executable
- [ ] All checkpoints converted to NumPy
- [ ] No conversion failures
- [ ] Files in correct output directory

### Verification
- [ ] All shapes match expected values
- [ ] Differences with PyTorch < 1e-3
- [ ] No critical mismatches
- [ ] Edge cases tested

### Ready for Next Phase
- [ ] All tests passing
- [ ] Checkpoints verified
- [ ] Ready to proceed to Phase 5 (Integration)

---

## Troubleshooting

### Issue: No checkpoint files created

**Check:**
- Is LLORCH_VALIDATE=1 set?
- Is llama.cpp built with -DLLORCH_VALIDATE=ON?
- Check stderr output for TEAM-004 messages

### Issue: Shape mismatches

**Check:**
- Is this the first transformer block?
- Are we extracting at the right point?
- Compare with Phase 2 mapping

### Issue: High differences with PyTorch

**Possible causes:**
- Different numerical precision (F16 vs F32)
- Different computation order
- Different random initialization

**Action:** Document differences, investigate if > 1e-3

---

## Notes and Issues

**TEAM-004 Notes:**
[Document any issues encountered during testing]

**Unexpected Findings:**
[Note any surprises or deviations from expectations]

**Performance Observations:**
[Document any performance impacts of checkpoint extraction]

---

**Status:** ⏳ PENDING  
**Previous Phase:** Phase 3 - Implementation (must be complete)  
**Next Phase:** Phase 5 - Integration with Tests  
**Estimated Time:** 1 hour  
**Actual Time:** [fill in after completion]

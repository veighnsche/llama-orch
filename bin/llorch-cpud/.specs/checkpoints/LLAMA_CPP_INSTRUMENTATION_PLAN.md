# TEAM-004: llama.cpp Instrumentation Master Plan
**Created by:** TEAM-004  
**Date:** 2025-10-08 17:00  
**Mission:** Instrument llama.cpp as our multi-reference validation dissection experiment  
**Status:** 🎯 READY FOR EXECUTION

---

## Mission Statement

**Goal:** Instrument llama.cpp on the `orch_log` branch to extract GPT-2 checkpoints 1-6 for multi-reference validation, achieving 75% confidence while maintaining our validation standards.

**Why llama.cpp:**
- ✅ Already in repo on `orch_log` branch
- ✅ C++ implementation (different from Python PyTorch)
- ✅ Most mature LLM inference engine
- ✅ Supports GPT-2 natively
- ✅ High confidence gain (+5%)

**Non-Negotiable:** We maintain our validation discipline. No shortcuts. No sacrificing assurance.

---

## Phase 1: Reconnaissance (1 hour)

### Objective
Locate exact instrumentation points in llama.cpp codebase for all 6 checkpoints.

### Tasks

#### Task 1.1: Find GPT-2 Model Architecture (15 min)
**Search for:**
- GPT-2 specific code in `src/llama-graph.cpp` or `src/llama-model.cpp`
- Model type enum for GPT-2
- Architecture-specific forward pass

**Expected locations:**
```cpp
// Likely in src/llama-graph.cpp or src/llama-model.cpp
enum llm_arch {
    LLM_ARCH_GPT2,
    LLM_ARCH_LLAMA,
    // ...
};
```

#### Task 1.2: Find Layer Implementations (20 min)
**Search for:**
- LayerNorm implementation
- Attention (QKV projection, scores, output)
- FFN implementation

**Expected patterns:**
```cpp
// LayerNorm
ggml_norm(ctx, input, eps);
ggml_mul(ctx, normalized, weight);
ggml_add(ctx, scaled, bias);

// Attention
ggml_mul_mat(ctx, w_qkv, input);  // QKV projection
ggml_soft_max(ctx, scores);        // Attention scores
ggml_mul_mat(ctx, attn, v);        // Attention output

// FFN
ggml_mul_mat(ctx, w_fc, input);    // FC layer
ggml_gelu(ctx, fc_output);         // GELU activation
ggml_mul_mat(ctx, w_proj, gelu);   // Projection
```

#### Task 1.3: Find Computation Graph Execution (15 min)
**Search for:**
- Where tensors are computed
- Where we can extract intermediate values
- Backend computation calls

**Expected:**
```cpp
ggml_backend_graph_compute(backend, gf);
// After this, tensors contain computed values
```

#### Task 1.4: Find Tensor Data Access (10 min)
**Search for:**
- How to read tensor data
- Conversion to float arrays
- Shape information access

**Expected:**
```cpp
float * data = (float *) tensor->data;
int64_t ne0 = tensor->ne[0];  // dimensions
int64_t ne1 = tensor->ne[1];
```

---

## Phase 2: Instrumentation Point Mapping (1 hour)

### Objective
Document exact file, function, and line numbers for each checkpoint.

### Checkpoint Mapping Template

For each checkpoint, document:
```markdown
### Checkpoint X: [Name]

**File:** `src/[filename].cpp`
**Function:** `[function_name]`
**Line:** ~[line_number]
**Tensor Name:** `[tensor_variable_name]`
**Shape:** [expected_shape]
**Data Type:** float32

**Instrumentation Point:**
```cpp
// TEAM-004: CHECKPOINT X - [Name]
// Location: After [operation], before [next_operation]
// Expected shape: [shape]
if (getenv("LLORCH_VALIDATE")) {
    // Extract and save checkpoint
}
```
```

### Expected Checkpoints

#### Checkpoint 1: LayerNorm Output
**Expected location:** After first LayerNorm in first transformer block
**Tensor:** Result of `ln_1` operation
**Shape:** `[n_tokens, hidden_size]` → `[2, 768]` for GPT-2

#### Checkpoint 2: QKV Projection
**Expected location:** After QKV linear projection, before split
**Tensors:** Q, K, V after projection
**Shape:** Each `[2, 768]` for GPT-2

#### Checkpoint 3: KV Cache State
**Expected location:** After KV cache update
**Tensor:** Cached K and V values
**Shape:** Depends on cache implementation

#### Checkpoint 4: Attention Scores
**Expected location:** After softmax, before attention output
**Tensor:** Attention weights
**Shape:** `[n_heads, n_tokens, n_tokens]` → `[12, 2, 2]` for GPT-2

#### Checkpoint 5: Attention Output
**Expected location:** After attention projection
**Tensor:** Output of attention block
**Shape:** `[2, 768]` for GPT-2

#### Checkpoint 6: FFN Output
**Expected location:** After FFN projection
**Tensor:** Output of feedforward network
**Shape:** `[2, 768]` for GPT-2

---

## Phase 3: Implement Checkpoint Extraction (2-3 hours)

### Objective
Add instrumentation code to extract and save checkpoints.

### Implementation Strategy

#### Step 3.1: Add Dependencies (15 min)

**File:** `CMakeLists.txt` or build system

**Add:**
- Include path for checkpoint extraction
- Link against any needed libraries (if using external serialization)

**OR use simple approach:**
- Write raw binary files
- Document format for Python loader

#### Step 3.2: Create Checkpoint Utilities (30 min)

**File:** `src/llama-checkpoint.h` (new file)

```cpp
// TEAM-004: Checkpoint extraction utilities for llorch-cpud validation
// Created: 2025-10-08

#pragma once

#include "ggml.h"
#include <cstdio>
#include <cstring>

namespace llama_checkpoint {

// Check if checkpoint extraction is enabled
inline bool is_enabled() {
    return getenv("LLORCH_VALIDATE") != nullptr;
}

// Save tensor to binary file
// Format: [shape_dims][shape_data][tensor_data]
inline void save_tensor(
    const char * checkpoint_name,
    const struct ggml_tensor * tensor
) {
    if (!is_enabled()) return;
    
    char filename[256];
    snprintf(filename, sizeof(filename), 
             "/tmp/llama_cpp_checkpoints/%s.bin", 
             checkpoint_name);
    
    FILE * f = fopen(filename, "wb");
    if (!f) {
        fprintf(stderr, "TEAM-004: Failed to open %s\n", filename);
        return;
    }
    
    // Write shape
    int32_t n_dims = ggml_n_dims(tensor);
    fwrite(&n_dims, sizeof(int32_t), 1, f);
    for (int i = 0; i < n_dims; i++) {
        int64_t dim = tensor->ne[i];
        fwrite(&dim, sizeof(int64_t), 1, f);
    }
    
    // Write data (convert to float32 if needed)
    size_t n_elements = ggml_nelements(tensor);
    if (tensor->type == GGML_TYPE_F32) {
        fwrite(tensor->data, sizeof(float), n_elements, f);
    } else {
        // Convert to float32
        float * data_f32 = new float[n_elements];
        ggml_backend_tensor_get(tensor, data_f32, 0, n_elements * sizeof(float));
        fwrite(data_f32, sizeof(float), n_elements, f);
        delete[] data_f32;
    }
    
    fclose(f);
    
    fprintf(stderr, "✅ TEAM-004: Checkpoint %s saved [", checkpoint_name);
    for (int i = 0; i < n_dims; i++) {
        fprintf(stderr, "%lld%s", (long long)tensor->ne[i], 
                i < n_dims-1 ? "x" : "");
    }
    fprintf(stderr, "]\n");
}

// Initialize checkpoint directory
inline void init() {
    if (!is_enabled()) return;
    
    system("mkdir -p /tmp/llama_cpp_checkpoints");
    fprintf(stderr, "TEAM-004: Checkpoint extraction enabled\n");
}

} // namespace llama_checkpoint
```

**File:** `src/llama-checkpoint.cpp` (new file)

```cpp
// TEAM-004: Checkpoint extraction implementation
#include "llama-checkpoint.h"

// Implementation details if needed
```

#### Step 3.3: Add Instrumentation to Each Checkpoint (90 min, 15 min each)

**Pattern for each checkpoint:**

```cpp
// TEAM-004: CHECKPOINT X - [Name]
// Extracts: [description]
// Expected shape: [shape]
// Validates against: PyTorch reference
#ifdef LLORCH_VALIDATE
    if (llama_checkpoint::is_enabled()) {
        llama_checkpoint::save_tensor("checkpoint_0X_[name]", [tensor_variable]);
    }
#endif
```

**Specific locations to instrument:**

1. **Checkpoint 1: LayerNorm**
   - After: `ggml_norm` + scale + bias
   - Before: Attention input

2. **Checkpoint 2: QKV**
   - After: QKV projection
   - Before: Split into Q, K, V

3. **Checkpoint 3: KV Cache**
   - After: Cache update
   - Before: Attention computation

4. **Checkpoint 4: Attention Scores**
   - After: Softmax
   - Before: Multiply with V

5. **Checkpoint 5: Attention Output**
   - After: Attention projection
   - Before: Residual add

6. **Checkpoint 6: FFN**
   - After: FFN projection
   - Before: Residual add

#### Step 3.4: Add Initialization Call (5 min)

**File:** `src/llama.cpp` or main entry point

```cpp
// TEAM-004: Initialize checkpoint extraction
#include "llama-checkpoint.h"

void llama_backend_init(void) {
    ggml_time_init();
    
    // TEAM-004: Initialize checkpoint directory
    llama_checkpoint::init();
    
    // ... rest of init
}
```

---

## Phase 4: Build and Test (1 hour)

### Objective
Verify instrumentation works and produces correct checkpoints.

### Step 4.1: Build llama.cpp (15 min)

```bash
cd /home/vince/Projects/llama-orch/reference/llama.cpp

# Clean build
rm -rf build
mkdir build
cd build

# Configure with checkpoint support
cmake .. -DLLORCH_VALIDATE=ON

# Build
make -j$(nproc)
```

### Step 4.2: Download GPT-2 Model (10 min)

```bash
# Download GPT-2 in GGUF format
cd /home/vince/Projects/llama-orch/reference/llama.cpp
./scripts/download-gguf.sh gpt2
# OR
huggingface-cli download ggml-org/gpt-2 gpt2.gguf
```

### Step 4.3: Run with Checkpoint Extraction (10 min)

```bash
cd /home/vince/Projects/llama-orch/reference/llama.cpp

# Clean checkpoint directory
rm -rf /tmp/llama_cpp_checkpoints
mkdir -p /tmp/llama_cpp_checkpoints

# Run with checkpoint extraction
LLORCH_VALIDATE=1 ./build/bin/llama-cli \
    -m models/gpt2.gguf \
    -p "Hello world" \
    -n 1 \
    --no-display-prompt

# Verify checkpoints created
ls -lh /tmp/llama_cpp_checkpoints/
```

**Expected output:**
```
✅ TEAM-004: Checkpoint extraction enabled
✅ TEAM-004: Checkpoint checkpoint_01_ln1_output saved [2x768]
✅ TEAM-004: Checkpoint checkpoint_02_q saved [2x768]
✅ TEAM-004: Checkpoint checkpoint_02_k saved [2x768]
✅ TEAM-004: Checkpoint checkpoint_02_v saved [2x768]
✅ TEAM-004: Checkpoint checkpoint_04_scores saved [12x2x2]
✅ TEAM-004: Checkpoint checkpoint_05_output saved [2x768]
✅ TEAM-004: Checkpoint checkpoint_06_ffn saved [2x768]
```

### Step 4.4: Convert to NumPy Format (15 min)

**File:** `bin/llorch-cpud/.test_helpers/convert_llama_cpp_checkpoints.py`

```python
#!/usr/bin/env python3
"""
TEAM-004: Convert llama.cpp binary checkpoints to NumPy format
"""

import numpy as np
import struct
import sys
from pathlib import Path

def load_llama_cpp_checkpoint(filepath):
    """Load checkpoint from llama.cpp binary format"""
    with open(filepath, 'rb') as f:
        # Read shape
        n_dims = struct.unpack('i', f.read(4))[0]
        shape = []
        for _ in range(n_dims):
            dim = struct.unpack('q', f.read(8))[0]
            shape.append(dim)
        
        # Read data
        n_elements = np.prod(shape)
        data = np.fromfile(f, dtype=np.float32, count=n_elements)
        
        # Reshape
        return data.reshape(shape)

def convert_all_checkpoints():
    """Convert all checkpoints from llama.cpp to NumPy"""
    input_dir = Path("/tmp/llama_cpp_checkpoints")
    output_dir = Path("/home/vince/Projects/llama-orch/.test-models/gpt2/extracted_weights")
    
    checkpoints = [
        "checkpoint_01_ln1_output",
        "checkpoint_02_q",
        "checkpoint_02_k",
        "checkpoint_02_v",
        "checkpoint_04_scores",
        "checkpoint_05_output",
        "checkpoint_06_ffn",
    ]
    
    for checkpoint in checkpoints:
        input_file = input_dir / f"{checkpoint}.bin"
        output_file = output_dir / f"{checkpoint}_llama_cpp.npy"
        
        if not input_file.exists():
            print(f"⚠️  Missing: {checkpoint}")
            continue
        
        try:
            data = load_llama_cpp_checkpoint(input_file)
            np.save(output_file, data)
            print(f"✅ Converted: {checkpoint} {data.shape}")
        except Exception as e:
            print(f"❌ Failed: {checkpoint} - {e}")

if __name__ == "__main__":
    convert_all_checkpoints()
```

**Run conversion:**
```bash
cd /home/vince/Projects/llama-orch/bin/llorch-cpud/.test_helpers
python3 convert_llama_cpp_checkpoints.py
```

### Step 4.5: Verify Checkpoint Shapes (10 min)

```bash
cd /home/vince/Projects/llama-orch

python3 << 'EOF'
import numpy as np
from pathlib import Path

weights_dir = Path(".test-models/gpt2/extracted_weights")

checkpoints = {
    "checkpoint_01_ln1_output": (2, 768),
    "checkpoint_02_q": (2, 768),
    "checkpoint_02_k": (2, 768),
    "checkpoint_02_v": (2, 768),
    "checkpoint_04_scores": (12, 2, 2),  # May vary
    "checkpoint_05_output": (2, 768),
    "checkpoint_06_ffn": (2, 768),
}

print("TEAM-004: Checkpoint Verification\n")

for name, expected_shape in checkpoints.items():
    pytorch_file = weights_dir / f"{name}.npy"
    llama_cpp_file = weights_dir / f"{name}_llama_cpp.npy"
    
    if not llama_cpp_file.exists():
        print(f"⚠️  {name}: llama.cpp checkpoint missing")
        continue
    
    llama_data = np.load(llama_cpp_file)
    
    if llama_data.shape != expected_shape:
        print(f"⚠️  {name}: shape mismatch")
        print(f"    Expected: {expected_shape}")
        print(f"    Got: {llama_data.shape}")
    else:
        print(f"✅ {name}: {llama_data.shape}")
    
    # Compare with PyTorch if available
    if pytorch_file.exists():
        pytorch_data = np.load(pytorch_file)
        diff = np.abs(pytorch_data - llama_data).max()
        print(f"    vs PyTorch: max diff = {diff:.6e}")
EOF
```

---

## Phase 5: Integration with Tests (30 min)

### Objective
Update llorch-cpud tests to use llama.cpp checkpoints.

### Step 5.1: Update Test Files (20 min)

**Pattern for each test file:**

```rust
// TEAM-004: Added llama.cpp reference validation
let llama_cpp_path = dir.join("checkpoint_01_ln1_output_llama_cpp.npy");
if llama_cpp_path.exists() {
    let mut llama_cpp_file = File::open(&llama_cpp_path)
        .expect("Failed to open llama.cpp reference");
    let llama_cpp_ref: Array2<f32> = Array2::read_npy(&mut llama_cpp_file)
        .expect("Failed to read llama.cpp reference");
    
    let mut llama_cpp_diff = 0.0f32;
    for (our, llama_cpp) in output.iter().zip(llama_cpp_ref.iter()) {
        llama_cpp_diff = llama_cpp_diff.max((our - llama_cpp).abs());
    }
    
    println!("\n📊 llama.cpp Comparison:");
    println!("  Max absolute difference: {:.6e}", llama_cpp_diff);
    
    if llama_cpp_diff < 1e-4 {
        println!("✅ LLAMA.CPP: Matches within tolerance");
    } else {
        println!("❌ LLAMA.CPP: Difference exceeds tolerance");
        panic!("llama.cpp max difference {} exceeds 1e-4", llama_cpp_diff);
    }
    
    // Cross-validate PyTorch vs llama.cpp
    let mut cross_diff = 0.0f32;
    for (pytorch, llama_cpp) in expected.iter().zip(llama_cpp_ref.iter()) {
        cross_diff = cross_diff.max((pytorch - llama_cpp).abs());
    }
    
    println!("\n📊 Cross-Validation (PyTorch vs llama.cpp):");
    println!("  Max difference: {:.6e}", cross_diff);
    
    if cross_diff < 1e-3 {
        println!("✅ CROSS-VALIDATION: References agree");
    } else {
        println!("⚠️  WARNING: References disagree by {:.6e}", cross_diff);
    }
    
    println!("\n🎉 MULTI-REFERENCE VALIDATION PASSED!");
    println!("   Our implementation matches BOTH PyTorch and llama.cpp");
} else {
    println!("\n⚠️  llama.cpp reference not available");
    println!("   Run: cd reference/llama.cpp && LLORCH_VALIDATE=1 ./build/bin/llama-cli ...");
    println!("   Single-reference validation only (PyTorch)");
}
```

### Step 5.2: Run Tests (10 min)

```bash
cd /home/vince/Projects/llama-orch/bin/llorch-cpud

# Run checkpoint 1 test
cargo test --test real_gpt2_checkpoint_01 test_checkpoint_01_multi_reference -- --nocapture

# Expected: Shows PyTorch AND llama.cpp validation
```

---

## Phase 6: Documentation and Handoff (30 min)

### Objective
Document the instrumentation for future maintenance.

### Deliverables

#### 1. Instrumentation Map (15 min)

**File:** `reference/llama.cpp/LLORCH_INSTRUMENTATION.md`

```markdown
# TEAM-004: llama.cpp Instrumentation for llorch-cpud Validation

## Overview

This document maps checkpoint extraction points in llama.cpp for multi-reference validation.

## Checkpoints

### Checkpoint 1: LayerNorm Output
- **File:** `src/llama-graph.cpp`
- **Function:** `[function_name]`
- **Line:** ~[line]
- **Tensor:** `[variable]`
- **Shape:** [2, 768]

[... for each checkpoint ...]

## Building with Checkpoints

```bash
cmake .. -DLLORCH_VALIDATE=ON
make -j$(nproc)
```

## Running with Checkpoints

```bash
LLORCH_VALIDATE=1 ./build/bin/llama-cli -m model.gguf -p "Hello" -n 1
```

## Checkpoint Format

Binary format:
- int32: number of dimensions
- int64[n_dims]: shape
- float32[n_elements]: data

Convert to NumPy:
```bash
python3 bin/llorch-cpud/.test_helpers/convert_llama_cpp_checkpoints.py
```
```

#### 2. Update TEAM_004_BRUTAL_AUDIT.md (10 min)

Add section:
```markdown
## Update: llama.cpp Instrumentation Complete (2025-10-08 17:XX)

TEAM-004 successfully instrumented llama.cpp as second reference:

### Work Completed
- ✅ Located all 6 checkpoint extraction points
- ✅ Implemented checkpoint utilities
- ✅ Added instrumentation to llama.cpp
- ✅ Verified checkpoints match expected shapes
- ✅ Integrated with llorch-cpud tests
- ✅ Cross-validation passing

### Confidence Improvement
- **Before:** 70% (PyTorch only)
- **After:** 75% (PyTorch + llama.cpp)

### Time Spent
- Reconnaissance: 1 hour
- Instrumentation: 3 hours
- Testing: 1 hour
- **Total:** 5 hours (as estimated)

See `reference/llama.cpp/LLORCH_INSTRUMENTATION.md` for details.
```

#### 3. Update STRATEGIC_ANALYSIS.md (5 min)

Mark Week 2 complete:
```markdown
## Week 2 Status: ✅ COMPLETE

- ✅ llama.cpp instrumented
- ✅ Checkpoints 1-6 validated against llama.cpp
- ✅ Cross-validation passing
- ✅ 75% confidence achieved

**Result:** Dual-reference validation complete, stakeholder confidence achieved
```

---

## Success Criteria

### Phase 1-2: Reconnaissance Complete
- [ ] All 6 checkpoint locations documented
- [ ] File, function, line numbers recorded
- [ ] Tensor names and shapes verified

### Phase 3: Instrumentation Complete
- [ ] Checkpoint utilities implemented
- [ ] All 6 checkpoints instrumented
- [ ] Code compiles without errors
- [ ] TEAM-004 signatures on all changes

### Phase 4: Testing Complete
- [ ] llama.cpp builds successfully
- [ ] Checkpoints extracted when LLORCH_VALIDATE=1
- [ ] All 6 checkpoint files created
- [ ] Shapes match expected values
- [ ] Converted to NumPy format

### Phase 5: Integration Complete
- [ ] Tests updated to use llama.cpp references
- [ ] All 6 tests pass with llama.cpp validation
- [ ] Cross-validation passes (PyTorch vs llama.cpp)
- [ ] No "reference not available" warnings

### Phase 6: Documentation Complete
- [ ] LLORCH_INSTRUMENTATION.md created
- [ ] TEAM_004_BRUTAL_AUDIT.md updated
- [ ] STRATEGIC_ANALYSIS.md updated
- [ ] Confidence updated to 75%

---

## Risk Mitigation

### Risk 1: Can't find checkpoint locations
**Mitigation:** llama.cpp is well-structured, use grep and code search
**Fallback:** Instrument at higher level (after full forward pass)

### Risk 2: Checkpoint shapes don't match
**Mitigation:** Verify with llama.cpp's own tests
**Fallback:** Adjust shape expectations, document differences

### Risk 3: Cross-validation fails (references disagree)
**Mitigation:** Check tolerance (< 1e-3), may need adjustment
**Fallback:** Document differences, investigate numerical precision

### Risk 4: Build issues with instrumentation
**Mitigation:** Use conditional compilation (#ifdef LLORCH_VALIDATE)
**Fallback:** Separate instrumented branch, don't modify main code

---

## Timeline

**Total Estimated Time:** 5-6 hours

- Phase 1: Reconnaissance (1 hour)
- Phase 2: Mapping (1 hour)
- Phase 3: Instrumentation (2-3 hours)
- Phase 4: Testing (1 hour)
- Phase 5: Integration (30 min)
- Phase 6: Documentation (30 min)

**Target Completion:** End of Week 2

---

## Next Steps After Completion

1. **Celebrate:** 75% confidence achieved! 🎉
2. **Demo to stakeholders:** Show dual-reference validation
3. **Document lessons learned:** What worked, what didn't
4. **Plan for checkpoints 7-12:** Apply same approach
5. **Consider third reference:** tinygrad for 80% confidence?

---

**Status:** 📋 PLAN COMPLETE - READY FOR EXECUTION  
**Team:** TEAM-004  
**Confidence Target:** 75%  
**Approach:** Surgical instrumentation, no shortcuts, maintain validation discipline

**Let's make a name for ourselves. Let's do this right. 🚀**

# Llama-2 Testing Strategy: End-to-End Validation

**Created by:** TEAM-008  
**Date:** 2025-10-08  
**Purpose:** Define complete testing approach from GGUF loading to output comparison

---

## Overview

The testing strategy follows a simple principle: **Model in → Something out → Compare with llama.cpp**

We don't need to implement everything before we can test. We can validate incrementally at each stage.

---

## Testing Levels

### Level 1: GGUF Parser ✅ COMPLETE
**Status:** ✅ Working  
**Test:** `cargo run --example test_gguf_parser`

**What we validate:**
- Model loads without errors
- Metadata parsed correctly
- All 291 tensors found
- Tensor shapes match expected (e.g., `[4096, 32000]`)
- Total size calculated correctly (6.67 GB)

**Reference:** GGUF file structure

**Result:**
```
✅ Validated as Llama architecture
✅ 32 layers found
✅ All key tensors present
Total size: 6.67 GB
```

---

### Level 2: Component Tests (Week 2)
**Status:** 🔄 In Progress  
**Components:** RMSNorm, RoPE, SwiGLU

#### 2.1 RMSNorm Test
**Test:** Unit test with known inputs
**File:** `tests/test_rms_norm.rs`

```rust
#[test]
fn test_rms_norm_against_reference() {
    // Input: known tensor
    let input = Array2::from_shape_vec((2, 4096), vec![...]).unwrap();
    let weight = Array1::from_vec(vec![...]); // From GGUF
    
    // Our implementation
    let rms_norm = RMSNorm::new(weight, 1e-5);
    let output = rms_norm.forward(&input);
    
    // Reference from llama.cpp checkpoint extractor
    let reference = load_reference_checkpoint("checkpoint_01_rms_norm.npy");
    
    // Compare
    assert_approx_eq!(output, reference, 1e-5);
}
```

**Reference:** llama.cpp checkpoint extractor

#### 2.2 RoPE Test
**Test:** Unit test with position encoding
**File:** `tests/test_rope.rs`

```rust
#[test]
fn test_rope_against_reference() {
    let q = Array3::from_shape_vec((1, 32, 128), vec![...]).unwrap();
    let k = Array3::from_shape_vec((1, 32, 128), vec![...]).unwrap();
    
    let (q_rope, k_rope) = apply_rope(&q, &k, position=0);
    
    let reference_q = load_reference("checkpoint_03_rope_q.npy");
    let reference_k = load_reference("checkpoint_03_rope_k.npy");
    
    assert_approx_eq!(q_rope, reference_q, 1e-5);
    assert_approx_eq!(k_rope, reference_k, 1e-5);
}
```

**Reference:** llama.cpp checkpoint extractor

#### 2.3 SwiGLU Test
**Test:** Unit test with FFN
**File:** `tests/test_swiglu.rs`

```rust
#[test]
fn test_swiglu_against_reference() {
    let input = Array2::from_shape_vec((2, 4096), vec![...]).unwrap();
    let gate_weight = load_weight("blk.0.ffn_gate.weight");
    let up_weight = load_weight("blk.0.ffn_up.weight");
    let down_weight = load_weight("blk.0.ffn_down.weight");
    
    let output = swiglu_ffn(&input, &gate_weight, &up_weight, &down_weight);
    
    let reference = load_reference("checkpoint_06_ffn_output.npy");
    
    assert_approx_eq!(output, reference, 1e-4);
}
```

**Reference:** llama.cpp checkpoint extractor

---

### Level 3: Checkpoint Validation (Week 3)
**Status:** ⏳ Pending  
**Test:** Compare intermediate outputs with llama.cpp

#### Using Team 006's Checkpoint Extractor

**Extract reference checkpoints:**
```bash
cd bin/llorch-cpud/tools/checkpoint-extractor
./build/llorch-checkpoint-extractor \
  /.test-models/llama2-7b/llama-2-7b.Q8_0.gguf \
  "Hello" \
  /tmp/llama2_reference
```

**This produces:**
- `checkpoint_01_rms_norm.npy` - First RMSNorm output
- `checkpoint_02_qkv.npy` - QKV projection
- `checkpoint_03_rope.npy` - After RoPE
- `checkpoint_04_attn_scores.npy` - Attention scores
- `checkpoint_05_attn_output.npy` - Attention output
- `checkpoint_06_ffn_output.npy` - FFN output
- `checkpoint_07_block_output.npy` - First block
- `checkpoint_08_logits.npy` - Full model logits
- ... (all 12 checkpoints)

**Our implementation:**
```rust
// In inference code, add checkpoint emission
#[cfg(feature = "checkpoints")]
{
    let checkpoint_dir = env::var("LLORCH_CHECKPOINT_DIR")
        .unwrap_or("/tmp/llorch_checkpoints".to_string());
    
    // After RMSNorm
    save_checkpoint(&output, &format!("{}/checkpoint_01_rms_norm.npy", checkpoint_dir));
    
    // After QKV
    save_checkpoint(&qkv, &format!("{}/checkpoint_02_qkv.npy", checkpoint_dir));
    
    // ... etc
}
```

**Comparison script:**
```python
import numpy as np

ref = np.load('/tmp/llama2_reference/checkpoint_01_rms_norm.npy')
ours = np.load('/tmp/llorch_checkpoints/checkpoint_01_rms_norm.npy')

diff = np.abs(ref - ours)
print(f"Max diff: {diff.max()}")
print(f"Mean diff: {diff.mean()}")

assert diff.max() < 1e-5, "Checkpoint 1 failed!"
print("✅ Checkpoint 1 passed!")
```

---

### Level 4: End-to-End Output (Week 4)
**Status:** ⏳ Pending  
**Test:** Full generation comparison

#### 4.1 Simple Comparison Script ✅ READY
**File:** `scripts/compare_with_llamacpp.sh`

**Usage:**
```bash
./scripts/compare_with_llamacpp.sh "Hello" 10
```

**What it does:**
1. Runs llama.cpp with prompt "Hello", generates 10 tokens
2. Saves output to temp file
3. (When ready) Runs llorch-cpud with same parameters
4. Compares outputs token-by-token

**Current output (llama.cpp only):**
```
Prompt: "Hello"
Output: ", I am interested in [1000"
```

**When llorch-cpud is ready:**
```bash
# Run both
./scripts/compare_with_llamacpp.sh "Hello" 10

# Expected:
# llama.cpp:   ", I am interested in [1000"
# llorch-cpud: ", I am interested in [1000"
# ✅ Outputs match!
```

#### 4.2 Deterministic Test Cases

**Test 1: Simple prompt**
```
Prompt: "Hello"
Temp: 0.0 (greedy)
Seed: 42
Expected: ", I am interested in [1000"
```

**Test 2: Question**
```
Prompt: "What is 2+2?"
Temp: 0.0
Seed: 42
Expected: (TBD - run llama.cpp to get reference)
```

**Test 3: Longer generation**
```
Prompt: "Once upon a time"
Temp: 0.0
Seed: 42
Tokens: 50
Expected: (TBD)
```

#### 4.3 Token-by-Token Comparison

**Script:** `scripts/compare_tokens.py`
```python
#!/usr/bin/env python3
import sys

def load_tokens(file_path):
    with open(file_path) as f:
        return f.read().strip()

llamacpp_output = load_tokens(sys.argv[1])
llorch_output = load_tokens(sys.argv[2])

if llamacpp_output == llorch_output:
    print("✅ Outputs match exactly!")
    sys.exit(0)
else:
    print("❌ Outputs differ:")
    print(f"llama.cpp:   {llamacpp_output}")
    print(f"llorch-cpud: {llorch_output}")
    
    # Show first difference
    for i, (c1, c2) in enumerate(zip(llamacpp_output, llorch_output)):
        if c1 != c2:
            print(f"First difference at position {i}:")
            print(f"  llama.cpp:   '{c1}' (ord={ord(c1)})")
            print(f"  llorch-cpud: '{c2}' (ord={ord(c2)})")
            break
    
    sys.exit(1)
```

---

## Testing Workflow

### Week 1 ✅ COMPLETE
```bash
# Test GGUF parser
cargo run --example test_gguf_parser
# ✅ Model loads, tensors verified
```

### Week 2 (Current)
```bash
# 1. Extract reference checkpoints
cd tools/checkpoint-extractor
./build/llorch-checkpoint-extractor \
  ../../.test-models/llama2-7b/llama-2-7b.Q8_0.gguf \
  "Hello" \
  /tmp/llama2_ref

# 2. Implement RMSNorm
# ... code ...

# 3. Test RMSNorm
cargo test test_rms_norm_against_reference
# ✅ Checkpoint 1 passes

# 4. Implement RoPE
# ... code ...

# 5. Test RoPE
cargo test test_rope_against_reference
# ✅ Checkpoint 3 passes

# 6. Implement SwiGLU
# ... code ...

# 7. Test SwiGLU
cargo test test_swiglu_against_reference
# ✅ Checkpoint 6 passes
```

### Week 3
```bash
# 1. Implement attention + full model
# ... code ...

# 2. Test all checkpoints
cargo test --features checkpoints
# ✅ All checkpoints 1-8 pass

# 3. Test first generation
cargo run --release -- \
  --model /.test-models/llama2-7b/llama-2-7b.Q8_0.gguf \
  --prompt "Hello" \
  --n-predict 1 \
  --temp 0.0

# Should output: ","
```

### Week 4
```bash
# 1. Full generation test
./scripts/compare_with_llamacpp.sh "Hello" 10
# ✅ Outputs match!

# 2. Multiple test cases
./scripts/compare_with_llamacpp.sh "What is 2+2?" 20
./scripts/compare_with_llamacpp.sh "Once upon a time" 50
# ✅ All match!

# 3. Production ready
cargo build --release
# ✅ Ready for deployment
```

---

## Success Criteria

### Minimum (Week 3)
- ✅ Checkpoint 12 passes (end-to-end generation)
- ✅ Single test case matches llama.cpp exactly

### Recommended (Week 4)
- ✅ All checkpoints 1-12 pass
- ✅ Multiple test cases match llama.cpp
- ✅ Deterministic output verified (temp=0)

### Production (Future)
- ✅ All test cases pass
- ✅ Temperature sampling works (temp>0)
- ✅ Performance acceptable (>50% of llama.cpp speed)
- ✅ Memory usage acceptable

---

## Reference Outputs (llama.cpp)

### Test Case 1: "Hello"
```
Prompt: "Hello"
Tokens: 10
Temp: 0.0
Seed: 42
Output: ", I am interested in [1000"
```

### Test Case 2: TBD
(Run llama.cpp to get reference outputs for more test cases)

---

## Tools Available

### 1. GGUF Parser ✅
**File:** `src/model/gguf_parser.rs`  
**Test:** `cargo run --example test_gguf_parser`

### 2. Checkpoint Extractor ✅
**Location:** `tools/checkpoint-extractor/`  
**Built:** Yes (by Team 007)  
**Usage:** `./build/llorch-checkpoint-extractor <model> <prompt> <output_dir>`

### 3. Comparison Script ✅
**File:** `scripts/compare_with_llamacpp.sh`  
**Usage:** `./scripts/compare_with_llamacpp.sh <prompt> <n_tokens>`

### 4. llama.cpp Binary ✅
**Location:** `reference/llama.cpp/build/bin/llama-cli`  
**Usage:** `./llama-cli -m <model> -p <prompt> -n <tokens> --temp 0.0`

---

## Next Steps

### Immediate (Week 2)
1. ✅ Extract reference checkpoints for "Hello" prompt
2. ⏳ Implement RMSNorm
3. ⏳ Test RMSNorm against checkpoint 1
4. ⏳ Implement RoPE
5. ⏳ Test RoPE against checkpoint 3
6. ⏳ Implement SwiGLU
7. ⏳ Test SwiGLU against checkpoint 6

### Short-term (Week 3)
1. Implement attention
2. Implement full model
3. Test all checkpoints 1-8
4. Generate first token

### Long-term (Week 4)
1. Full generation working
2. Compare with llama.cpp
3. Multiple test cases passing
4. Production ready

---

## Key Insight

**You don't need to implement everything before testing!**

- ✅ Week 1: Test GGUF loading → Model loads
- ✅ Week 2: Test components → Checkpoints 1, 3, 6 pass
- ✅ Week 3: Test full model → Checkpoints 1-8 pass
- ✅ Week 4: Test generation → Output matches llama.cpp

Each week builds on the previous, with concrete validation at every step.

---

## Sign-off

**Created by:** TEAM-008 (Foundation Implementation)  
**Date:** 2025-10-08  
**Status:** Testing infrastructure ready

**Available now:**
- ✅ GGUF parser test
- ✅ Checkpoint extractor (Team 006)
- ✅ Comparison script
- ✅ llama.cpp reference

**Ready for:**
- Component testing (Week 2)
- Checkpoint validation (Week 3)
- End-to-end comparison (Week 4)

---

*"Test early, test often, test against truth."*  
— TEAM-008, Foundation Implementation Division

**END TESTING STRATEGY**

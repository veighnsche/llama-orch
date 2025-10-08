# Isolated Checkpoint 1 Validation Setup

**Purpose:** Extract and compare JUST LayerNorm (Checkpoint 1) from each reference implementation.

**Critical Lesson from worker-orcd:** Compare at EVERY step, not just end-to-end.

---

## The Problem We're Solving

From worker-orcd post-mortem:
> **Day 1-23:** Fix softmax → Still broken, Fix sampling → Still broken...  
> **Should have been:** Compare layer 1 output → If different, fix layer 1 → Move to layer 2

**We need to prove Checkpoint 1 works BEFORE moving to Checkpoint 2.**

---

## What This Test Does

1. **Generates identical input** for all 4 implementations
2. **Runs ONLY LayerNorm** (not full model)
3. **Compares outputs** with tolerance checking
4. **Reports differences** element-by-element

**This is component-level validation, not end-to-end.**

---

## Test File

**Location:** `tests/isolated_checkpoint_01.rs`

**Tests:**
1. `test_isolated_checkpoint_01_our_determinism` ✅ (always runs)
2. `test_isolated_checkpoint_01_vs_tinygrad` (manual)
3. `test_isolated_checkpoint_01_vs_candle` (manual)
4. `test_isolated_checkpoint_01_all` ✅ (summary)

---

## Setup Instructions

### 1. Tinygrad Setup

**Install tinygrad:**
```bash
cd /home/vince/Projects/llama-orch/reference/tinygrad
pip install -e .
```

**Verify:**
```bash
python3 -c "from tinygrad import Tensor; from tinygrad.nn import LayerNorm; print('✅ tinygrad ready')"
```

**Run test:**
```bash
cd /home/vince/Projects/llama-orch/bin/llorch-cpud
cargo test --test isolated_checkpoint_01 test_isolated_checkpoint_01_vs_tinygrad -- --ignored --nocapture
```

### 2. Candle Setup

**Create standalone test binary:**

```bash
cd /home/vince/Projects/llama-orch/bin/llorch-cpud
mkdir -p .test_helpers/candle_ln
cd .test_helpers/candle_ln
```

**Cargo.toml:**
```toml
[package]
name = "candle_ln_test"
version = "0.1.0"
edition = "2021"

[dependencies]
candle-core = { path = "../../../../reference/candle/candle-core" }
candle-nn = { path = "../../../../reference/candle/candle-nn" }
bincode = "1.3"
```

**src/main.rs:**
```rust
use candle_core::{Tensor, Device, DType};
use candle_nn::LayerNorm;
use std::fs;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::Cpu;
    
    // Load input
    let input_data: Vec<f32> = bincode::deserialize(
        &fs::read("/tmp/llorch_test_input.bin")?
    )?;
    let input = Tensor::from_vec(input_data, (2, 1024), &device)?;
    
    // Create LayerNorm with same params as llorch-cpud
    let weight = Tensor::ones((1024,), DType::F32, &device)?;
    let bias = Tensor::zeros((1024,), DType::F32, &device)?;
    let ln = LayerNorm::new(weight, bias, 1e-5);
    
    // Run forward
    let output = ln.forward(&input)?;
    
    // Save output
    let output_vec = output.flatten_all()?.to_vec1::<f32>()?;
    fs::write("/tmp/llorch_test_output_candle.bin", 
              bincode::serialize(&output_vec)?)?;
    
    eprintln!("✅ Candle LayerNorm output shape: {:?}", output.shape());
    eprintln!("   Sample (first 5): {:?}", &output_vec[..5]);
    
    Ok(())
}
```

**Build:**
```bash
cargo build --release
```

**Update test to call this binary** (modify `run_candle_layernorm` in `isolated_checkpoint_01.rs`)

### 3. Mistral.rs Setup

Mistral.rs uses Candle internally, so if Candle matches, Mistral.rs should too.

**Optional:** Create similar standalone test using mistralrs-core.

---

## Running the Tests

### Quick Check (Our Implementation Only)
```bash
cargo test --test isolated_checkpoint_01 test_isolated_checkpoint_01_all -- --nocapture
```

**Expected output:**
```
Our LayerNorm output (first 10): [-0.24642323, -0.23148638, -0.216551, ...]
Row 0: mean=0.000000e+0, variance=1.000000
Row 1: mean=0.000000e+0, variance=1.000000
✅ Our LayerNorm is mathematically correct
```

### vs Tinygrad
```bash
cargo test --test isolated_checkpoint_01 test_isolated_checkpoint_01_vs_tinygrad -- --ignored --nocapture
```

**Expected output:**
```
=== Tinygrad Comparison ===
Shape: [2, 1024]
Max absolute difference: 1.234e-05
Max relative difference: 2.345e-05
Tolerance: 1.000e-04
Our output (first 5):  [-0.24642323, -0.23148638, -0.216551, -0.20161863, -0.18669073]
Ref output (first 5):  [-0.24642320, -0.23148635, -0.216551, -0.20161860, -0.18669070]
✅ PASS: All elements within tolerance
```

### vs Candle
```bash
cargo test --test isolated_checkpoint_01 test_isolated_checkpoint_01_vs_candle -- --ignored --nocapture
```

---

## What Success Looks Like

### ✅ Perfect Match (Unlikely but Ideal)
```
Max absolute difference: 0.000e+00
Max relative difference: 0.000e+00
✅ PASS: All elements within tolerance
```

### ✅ Excellent Match (Expected)
```
Max absolute difference: 1.234e-05
Max relative difference: 2.345e-05
✅ PASS: All elements within tolerance
```

### ✅ Good Match (Acceptable)
```
Max absolute difference: 9.876e-05
Max relative difference: 1.234e-04
✅ PASS: All elements within tolerance
```

### ❌ Mismatch (Problem)
```
Max absolute difference: 5.678e-03
❌ 1024 elements exceed tolerance
  Element 0: ours=-0.246423, ref=-0.241234, diff=5.189e-03 (2.11%)
```

**Action:** Investigate why our LayerNorm differs from reference.

---

## Troubleshooting

### Tinygrad Import Error
```bash
cd /home/vince/Projects/llama-orch/reference/tinygrad
pip install -e .
# Or: pip install numpy tinygrad
```

### Candle Compilation Error
```bash
cd .test_helpers/candle_ln
cargo clean
cargo build --release
```

### NumPy File Format Error
The test includes a simple NPY writer/reader. If it fails:
```bash
pip install numpy
python3 -c "import numpy as np; np.save('/tmp/test.npy', np.array([[1,2],[3,4]]))"
```

---

## Why This Matters

### From worker-orcd Post-Mortem:

**What Went Wrong:**
- Compared end-to-end only
- Never isolated components
- 23 days debugging symptoms

**What Should Have Been Done:**
- Day 1: Compare embedding output
- Day 2: Compare layer 1 output
- Day 3: Compare layer 2 output
- **Estimated time:** 1-2 days

**For llorch-cpud:**
- ✅ Checkpoint 1: Isolate LayerNorm, compare NOW
- ⏳ Checkpoint 2: Isolate QKV, compare BEFORE moving to 3
- ⏳ Checkpoint 3: Isolate KV Cache, compare BEFORE moving to 4

**This is the golden rule: COMPARE AT EVERY STEP.**

---

## Next Steps

### After Checkpoint 1 Passes

1. ✅ LayerNorm matches all references (< 1e-4 difference)
2. ⏭️ Move to Checkpoint 2 (QKV Projection)
3. ⏭️ Create `isolated_checkpoint_02.rs`
4. ⏭️ Repeat validation process

### If Checkpoint 1 Fails

1. ❌ DO NOT move to Checkpoint 2
2. 🔍 Investigate LayerNorm implementation
3. 🔧 Fix until it matches references
4. ✅ Re-run validation
5. ⏭️ Only then move to Checkpoint 2

---

## Stakeholder Proof

This test provides:

1. **Component-Level Validation** - Not end-to-end, just LayerNorm
2. **Multiple References** - Tinygrad, Candle, Mistral.rs
3. **Identical Input** - Same data for all implementations
4. **Detailed Comparison** - Element-by-element with tolerance
5. **Clear Pass/Fail** - No ambiguity

**This is what stakeholders want: Proof that Checkpoint 1 works BEFORE moving forward.**

---

## Files

- `tests/isolated_checkpoint_01.rs` - Test suite
- `ISOLATED_CHECKPOINT_01_SETUP.md` - This file
- `.test_helpers/candle_ln/` - Candle standalone test (optional)

---

Built by TEAM CASCADE 🌊

*"Compare at every step. Fix before moving forward. No more 23-day debugging sessions."*

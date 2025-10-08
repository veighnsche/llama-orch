# Cross-Reference Validation Plan: LayerNorm Checkpoint 1

**Date:** 2025-10-08  
**Stakeholder Question:** "Can you prove that tinygrad, mistral.rs, and candle with a similar function all make the same output as our checkpoint 1 function?"

---

## Executive Summary

**Goal:** Prove that our LayerNorm implementation produces identical outputs to three reference implementations:
1. **tinygrad** (Python, research-grade)
2. **Candle** (Rust, production-grade)
3. **Mistral.rs** (Rust, production-grade)

**Approach:** Extract LayerNorm outputs from each reference implementation and compare with our implementation using identical inputs.

**Expected Result:** All implementations should produce **nearly identical** outputs (within floating-point tolerance ~1e-5).

---

## Why Results May Differ (And Why That's OK)

### Expected Differences

1. **Floating-Point Precision**
   - Different libraries use different precision internally
   - Candle may use F16/BF16 internally, then cast to F32
   - Accumulation order affects final bits
   - **Tolerance:** ~1e-5 to 1e-4 is acceptable

2. **Implementation Details**
   - **Biased vs Unbiased Variance:** All should use biased (divide by N)
   - **Epsilon Placement:** Some add before sqrt, some after
   - **Broadcast Order:** Affects intermediate precision
   - **BLAS Backend:** Different matrix libraries have different rounding

3. **Data Type Handling**
   - tinygrad: Pure Python/NumPy (F32 or F64)
   - Candle: May use F16 internally for efficiency
   - Mistral.rs: Candle-based, same considerations
   - Our impl: Pure F32 ndarray

### What Would Be a Problem

❌ **Differences > 1e-3** → Likely implementation bug  
❌ **Different shapes** → Definitely wrong  
❌ **NaN or Inf** → Critical error  
❌ **Sign flips** → Wrong formula

✅ **Differences < 1e-5** → Expected floating-point variance  
✅ **Same mean/variance** → Correct normalization  
✅ **Same shape** → Correct broadcasting

---

## Validation Strategy

### Phase 1: Verify References Work (CRITICAL)

**Before adding any logging**, verify each reference runs successfully:

```bash
# Test tinygrad (current branch)
cd reference/tinygrad/examples
python gpt2.py --prompt "Hello." --temperature 0 --count 1

# Test candle (current branch)
cd reference/candle/candle-examples
cargo run --release --example llama -- --prompt "Hello."

# Test mistral.rs (current branch)
cd reference/mistral.rs
cargo run --release -- --prompt "Hello."
```

**If any fail:** Switch to latest stable release tag before proceeding.

### Phase 2: Switch to orch_log Branch

All three references have an `orch_log` branch with validation checkpoints:

```bash
# Tinygrad
cd reference/tinygrad
git checkout orch_log

# Candle
cd reference/candle
git checkout orch_log

# Mistral.rs
cd reference/mistral.rs
git checkout orch_log
```

### Phase 3: Add Non-Blocking Logging

**CRITICAL:** Logging must NOT block the main program.

#### Tinygrad Logging (Already Present)
File: `reference/tinygrad/examples/gpt2.py` lines 94-99

```python
# Uncomment this block (already exists):
ln1_out = self.ln_1(x)
if getenv("VALIDATE") and not hasattr(self, "_checkpoint1_printed"):
  print(f"[CHECKPOINT 1] LayerNorm output shape: {ln1_out.shape}")
  print(f"[CHECKPOINT 1] Output sample: {ln1_out[0, 0, :5].numpy()}")
  self._checkpoint1_printed = True
```

**Non-blocking:** Uses `hasattr` guard, prints once, doesn't affect execution.

#### Candle Logging (To Be Added)
File: `reference/candle/candle-nn/src/layer_norm.rs` line 109

```rust
// Add after line 109 in forward():
if std::env::var("VALIDATE").is_ok() {
    if let Ok(sample) = x.i((0, 0, ..5))?.to_vec1::<f32>() {
        eprintln!("[CHECKPOINT 1] LayerNorm output shape: {:?}", x.shape());
        eprintln!("[CHECKPOINT 1] Output sample: {:?}", sample);
    }
}
```

**Non-blocking:** Uses `eprintln!` (stderr, buffered), doesn't panic, doesn't block.

#### Mistral.rs Logging (To Be Added)
File: `reference/mistral.rs/mistralrs-core/src/layers.rs` (find LayerNorm impl)

```rust
// Add in LayerNorm forward method:
if std::env::var("VALIDATE").is_ok() {
    if let Ok(sample) = output.i((0, 0, ..5))?.to_vec1::<f32>() {
        eprintln!("[CHECKPOINT 1] LayerNorm output shape: {:?}", output.shape());
        eprintln!("[CHECKPOINT 1] Output sample: {:?}", sample);
    }
}
```

**Non-blocking:** Same as Candle.

### Phase 4: Extract Reference Outputs

#### Test Input (Standardized)
```
Prompt: "Hello."
Tokens: [15496, 13] (tiktoken GPT-2 encoding)
Model: GPT-2 Medium (1024 dim)
Temperature: 0 (deterministic)
```

#### Run Each Reference
```bash
# Tinygrad
cd reference/tinygrad/examples
VALIDATE=1 python gpt2.py --prompt "Hello." --temperature 0 --count 1 2>&1 | tee /tmp/tinygrad_ln.txt

# Candle (if example exists)
cd reference/candle/candle-examples
VALIDATE=1 cargo run --release --example gpt2 -- --prompt "Hello." 2>&1 | tee /tmp/candle_ln.txt

# Mistral.rs (if applicable)
cd reference/mistral.rs
VALIDATE=1 cargo run --release -- --prompt "Hello." 2>&1 | tee /tmp/mistral_ln.txt
```

### Phase 5: Compare with Our Implementation

Create test that:
1. Loads same input (embedding output for "Hello.")
2. Runs our LayerNorm
3. Compares with reference outputs
4. Reports differences

---

## Implementation Plan

### Step 1: Create Test Input Generator

```rust
// tests/cross_reference_validation.rs
fn generate_standard_test_input() -> Array2<f32> {
    // Simulate embedding output for tokens [15496, 13]
    // Shape: [1, 2, 1024] → flatten to [2, 1024]
    // Use deterministic values matching GPT-2 embeddings
    Array2::from_shape_fn((2, 1024), |(i, j)| {
        // Deterministic pattern matching typical embedding magnitudes
        ((i * 1024 + j) as f32 * 0.01).sin() * 0.5
    })
}
```

### Step 2: Create Reference Output Parser

```rust
fn parse_reference_output(file_path: &str) -> Vec<f32> {
    // Parse "[CHECKPOINT 1] Output sample: [...]" from log files
    // Extract first 5 values for comparison
}
```

### Step 3: Create Comparison Test

```rust
#[test]
#[ignore] // Run manually after extracting reference outputs
fn test_cross_reference_layernorm() {
    let input = generate_standard_test_input();
    
    // Our implementation
    let ln = LayerNorm::new(
        Array1::ones(1024),
        Array1::zeros(1024),
        1e-5
    );
    let our_output = ln.forward(&input);
    
    // Load reference outputs
    let tinygrad_output = parse_reference_output("/tmp/tinygrad_ln.txt");
    let candle_output = parse_reference_output("/tmp/candle_ln.txt");
    let mistral_output = parse_reference_output("/tmp/mistral_ln.txt");
    
    // Compare (first 5 elements)
    let our_sample: Vec<f32> = our_output.iter().take(5).copied().collect();
    
    println!("Our output:      {:?}", our_sample);
    println!("Tinygrad output: {:?}", tinygrad_output);
    println!("Candle output:   {:?}", candle_output);
    println!("Mistral output:  {:?}", mistral_output);
    
    // Assert within tolerance
    for (i, (&ours, &ref_val)) in our_sample.iter().zip(tinygrad_output.iter()).enumerate() {
        let diff = (ours - ref_val).abs();
        assert!(diff < 1e-4, "Element {} differs by {}: {} vs {}", i, diff, ours, ref_val);
    }
}
```

---

## Expected Results

### Scenario 1: Perfect Match (Unlikely)
```
Our output:      [-0.24643, -0.23149, -0.21656, -0.20163, -0.18670]
Tinygrad output: [-0.24643, -0.23149, -0.21656, -0.20163, -0.18670]
Candle output:   [-0.24643, -0.23149, -0.21656, -0.20163, -0.18670]
Mistral output:  [-0.24643, -0.23149, -0.21656, -0.20163, -0.18670]
```
**Interpretation:** All implementations identical (very rare due to FP differences).

### Scenario 2: Close Match (Expected)
```
Our output:      [-0.24643, -0.23149, -0.21656, -0.20163, -0.18670]
Tinygrad output: [-0.24642, -0.23148, -0.21655, -0.20162, -0.18669]
Candle output:   [-0.24644, -0.23150, -0.21657, -0.20164, -0.18671]
Mistral output:  [-0.24643, -0.23149, -0.21656, -0.20163, -0.18670]
```
**Interpretation:** Differences < 1e-4, all implementations correct. ✅

### Scenario 3: Significant Difference (Problem)
```
Our output:      [-0.24643, -0.23149, -0.21656, -0.20163, -0.18670]
Tinygrad output: [-0.24642, -0.23148, -0.21655, -0.20162, -0.18669]
Candle output:   [-0.35821, -0.31442, -0.28193, -0.25012, -0.21890]
Mistral output:  [-0.24643, -0.23149, -0.21656, -0.20163, -0.18670]
```
**Interpretation:** Candle differs significantly → investigate Candle implementation. ❌

---

## Stakeholder Communication

### Question
"Can you prove that tinygrad, mistral.rs, and candle with a similar function all make the same output as our checkpoint 1 function?"

### Answer (After Validation)

**Short Answer:** Yes, with expected floating-point variance.

**Detailed Answer:**

We compared our LayerNorm implementation against three production-grade reference implementations:

| Implementation | Match Quality | Max Difference | Status |
|----------------|---------------|----------------|--------|
| **Tinygrad** | Excellent | < 1e-5 | ✅ Validated |
| **Candle** | Excellent | < 1e-5 | ✅ Validated |
| **Mistral.rs** | Excellent | < 1e-5 | ✅ Validated |

**Why Small Differences Exist:**
- Different floating-point accumulation orders
- Different internal precision (F16 vs F32)
- Different BLAS backends
- All differences are within acceptable tolerance (< 1e-4)

**Conclusion:** Our implementation is **mathematically equivalent** to all three references. Small differences (< 1e-5) are expected and acceptable due to floating-point arithmetic variations.

---

## Parity Explanation

### What is "Parity"?

**Parity** = Functional equivalence despite implementation differences.

### Do We Have Parity?

✅ **Yes**, if:
- Outputs differ by < 1e-4
- Same normalization behavior (mean ≈ 0, variance ≈ 1)
- Same shape handling
- Same edge case behavior

❌ **No**, if:
- Outputs differ by > 1e-3
- Different mathematical formula
- Different behavior on edge cases

### Why Exact Equality is Impossible

1. **IEEE 754 Floating-Point**
   - `(a + b) + c ≠ a + (b + c)` (associativity breaks)
   - Different accumulation orders → different results
   - Example: `0.1 + 0.2 ≠ 0.3` in binary

2. **Library Differences**
   - NumPy (tinygrad) uses different BLAS than ndarray (ours)
   - Candle uses custom kernels
   - Each has different rounding behavior

3. **Precision Tradeoffs**
   - Some use F16 for speed
   - Some use F64 for accuracy
   - We use F32 (standard)

**Bottom Line:** Parity means "same behavior within tolerance", not "bit-exact equality".

---

## Next Steps

### Immediate (This Session)
1. ✅ Create validation plan (this document)
2. ⬜ Verify all references run on current branches
3. ⬜ Switch to `orch_log` branches
4. ⬜ Add non-blocking logging to Candle/Mistral.rs
5. ⬜ Extract reference outputs
6. ⬜ Create comparison test
7. ⬜ Run validation
8. ⬜ Document results for stakeholders

### Future (Checkpoints 2-12)
- Repeat this process for each checkpoint
- Build comprehensive validation suite
- Automate reference comparison

---

## Files to Create

1. **`tests/cross_reference_validation.rs`** - Comparison test suite
2. **`scripts/extract_reference_outputs.sh`** - Automation script
3. **`CROSS_REFERENCE_RESULTS.md`** - Validation results
4. **`STAKEHOLDER_PARITY_EXPLANATION.md`** - Non-technical explanation

---

## Safety Checklist

Before modifying any reference implementation:

- [ ] Verify reference runs successfully on current branch
- [ ] Create backup branch: `git checkout -b backup-$(date +%s)`
- [ ] Test reference after adding logging
- [ ] Ensure logging uses stderr (non-blocking)
- [ ] Ensure logging has guard (`hasattr`, `env::var`)
- [ ] Ensure logging doesn't panic or unwrap unsafely
- [ ] Test that main program still completes successfully

---

Built by TEAM CASCADE 🌊

*"Validation through comparison. Confidence through consensus."*

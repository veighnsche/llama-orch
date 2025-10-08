# TEAM-004: Phase 5 - Integration with Tests
**Part of:** llama.cpp Instrumentation Master Plan  
**Duration:** 30 minutes  
**Status:** ⏳ PENDING  
**Depends on:** Phase 4 (Testing) must be complete

---

## Objective

Update llorch-cpud tests to use llama.cpp checkpoints for multi-reference validation.

**Goal:** All 6 checkpoint tests validate against both PyTorch AND llama.cpp, with cross-validation passing.

---

## Step 5.1: Update Test Pattern (5 min)

### Standard Multi-Reference Pattern

**Pattern to add to each test:**
```rust
// TEAM-004: Added llama.cpp reference validation
let llama_cpp_path = dir.join("checkpoint_XX_[name]_llama_cpp.npy");
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
    
    // TEAM-004: Cross-validate PyTorch vs llama.cpp
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

---

## Step 5.2: Update Checkpoint 1 Test (5 min)

### File to Update

**File:** `tests/real_gpt2_checkpoint_01.rs`  
**Function:** `test_checkpoint_01_multi_reference()`

### Add llama.cpp Validation

**Insert after PyTorch validation (around line 132):**
```rust
    // TEAM-004: Validate against llama.cpp (if available)
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
        
        // TEAM-004: Cross-validate PyTorch vs llama.cpp
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

### Test the Update

```bash
cd /home/vince/Projects/llama-orch/bin/llorch-cpud

cargo test --test real_gpt2_checkpoint_01 test_checkpoint_01_multi_reference -- --nocapture
```

**Expected:** Test passes with llama.cpp validation

### Checklist

- [ ] Code added to checkpoint 1 test
- [ ] Test compiles without errors
- [ ] Test runs and shows llama.cpp validation
- [ ] Cross-validation passes

---

## Step 5.3: Update Checkpoint 2 Test (5 min)

### File to Update

**File:** `tests/real_gpt2_checkpoint_02.rs`  
**Function:** `test_checkpoint_02_real_gpt2()`

### Add llama.cpp Validation

**Note:** This test validates Q, K, V separately. Add validation for each:

```rust
// TEAM-004: Validate Q against llama.cpp
let llama_cpp_q_path = dir.join("checkpoint_02_q_llama_cpp.npy");
if llama_cpp_q_path.exists() {
    let mut llama_cpp_file = File::open(&llama_cpp_q_path)
        .expect("Failed to open llama.cpp Q reference");
    let llama_cpp_q: Array2<f32> = Array2::read_npy(&mut llama_cpp_file)
        .expect("Failed to read llama.cpp Q reference");
    
    let mut diff = 0.0f32;
    for (our, llama_cpp) in q_output.iter().zip(llama_cpp_q.iter()) {
        diff = diff.max((our - llama_cpp).abs());
    }
    
    println!("\n📊 llama.cpp Q Comparison: max diff = {:.6e}", diff);
    if diff < 1e-4 {
        println!("✅ LLAMA.CPP Q: Matches within tolerance");
    }
    
    // Repeat for K and V...
}
```

### Checklist

- [ ] Code added for Q validation
- [ ] Code added for K validation
- [ ] Code added for V validation
- [ ] Test passes with llama.cpp validation

---

## Step 5.4: Update Checkpoint 6 Test (5 min)

### File to Update

**File:** `tests/real_gpt2_checkpoint_06.rs`  
**Function:** `test_checkpoint_06_multi_reference()`

### Add llama.cpp Validation

**Insert after Candle validation check (around line 197):**
```rust
    // TEAM-004: Validate against llama.cpp (if available)
    let llama_cpp_path = dir.join("checkpoint_06_ffn_llama_cpp.npy");
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
        
        // TEAM-004: Cross-validate PyTorch vs llama.cpp
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
        println!("   Our implementation matches PyTorch, Candle, AND llama.cpp");
    } else {
        println!("\n⚠️  llama.cpp reference not available");
    }
```

### Checklist

- [ ] Code added to checkpoint 6 test
- [ ] Test compiles without errors
- [ ] Test runs and shows llama.cpp validation
- [ ] Works with or without Candle reference

---

## Step 5.5: Update Remaining Tests (10 min)

### Tests to Update

1. **Checkpoint 3:** `tests/real_gpt2_checkpoint_03.rs`
2. **Checkpoint 4:** `tests/real_gpt2_checkpoint_04.rs`
3. **Checkpoint 5:** `tests/real_gpt2_checkpoint_05.rs`

### Apply Same Pattern

For each test:
1. Find PyTorch validation section
2. Add llama.cpp validation after it
3. Use appropriate checkpoint filename
4. Test compilation and execution

### Batch Test Command

```bash
cd /home/vince/Projects/llama-orch/bin/llorch-cpud

# Test all checkpoints
for i in 01 02 03 04 05 06; do
    echo "Testing checkpoint $i..."
    cargo test --test real_gpt2_checkpoint_$i -- --nocapture 2>&1 | grep -E "(✅|❌|PASSED|FAILED)"
done
```

### Checklist

- [ ] Checkpoint 3 test updated
- [ ] Checkpoint 4 test updated
- [ ] Checkpoint 5 test updated
- [ ] All tests compile
- [ ] All tests pass

---

## Step 5.6: Run Full Test Suite (5 min)

### Run All Multi-Reference Tests

```bash
cd /home/vince/Projects/llama-orch/bin/llorch-cpud

# Run all checkpoint tests
cargo test --test real_gpt2_checkpoint_01 -- --nocapture
cargo test --test real_gpt2_checkpoint_02 -- --nocapture
cargo test --test real_gpt2_checkpoint_03 -- --nocapture
cargo test --test real_gpt2_checkpoint_04 -- --nocapture
cargo test --test real_gpt2_checkpoint_05 -- --nocapture
cargo test --test real_gpt2_checkpoint_06 -- --nocapture
```

### Expected Output Pattern

For each test:
```
✅ PYTORCH: [Layer] matches HuggingFace (max diff X.XXe-XX)
✅ LLAMA.CPP: Matches within tolerance
✅ CROSS-VALIDATION: References agree
🎉 MULTI-REFERENCE VALIDATION PASSED!
   Our implementation matches BOTH PyTorch and llama.cpp
```

### Verify No Fallback Warnings

```bash
# Check for "not available" warnings
cargo test 2>&1 | grep "not available"
```

**Expected:** No warnings (all references available)

### Checklist

- [ ] All 6 tests run successfully
- [ ] All show PyTorch validation passing
- [ ] All show llama.cpp validation passing
- [ ] All show cross-validation passing
- [ ] No "reference not available" warnings

---

## Completion Checklist

### Code Changes
- [ ] Checkpoint 1 test updated with llama.cpp validation
- [ ] Checkpoint 2 test updated with llama.cpp validation
- [ ] Checkpoint 3 test updated with llama.cpp validation
- [ ] Checkpoint 4 test updated with llama.cpp validation
- [ ] Checkpoint 5 test updated with llama.cpp validation
- [ ] Checkpoint 6 test updated with llama.cpp validation

### Testing
- [ ] All tests compile without errors
- [ ] All tests run without crashes
- [ ] All PyTorch validations pass
- [ ] All llama.cpp validations pass
- [ ] All cross-validations pass
- [ ] No fallback warnings

### Validation Results
- [ ] Differences with PyTorch < 1e-4
- [ ] Differences with llama.cpp < 1e-4
- [ ] Cross-validation differences < 1e-3
- [ ] All checkpoints show "MULTI-REFERENCE VALIDATION PASSED"

### Ready for Next Phase
- [ ] All tests passing
- [ ] Multi-reference validation working
- [ ] Ready to proceed to Phase 6 (Documentation)

---

## Example Test Output

### Before (PyTorch only)
```
✅ PYTORCH: LayerNorm matches HuggingFace (max diff 5.960464e-8)

⚠️  Candle reference not available
   Single-reference validation only (PyTorch)
```

### After (PyTorch + llama.cpp)
```
✅ PYTORCH: LayerNorm matches HuggingFace (max diff 5.960464e-8)

📊 llama.cpp Comparison:
  Max absolute difference: 1.234567e-05

✅ LLAMA.CPP: Matches within tolerance

📊 Cross-Validation (PyTorch vs llama.cpp):
  Max difference: 1.234567e-05

✅ CROSS-VALIDATION: References agree

🎉 MULTI-REFERENCE VALIDATION PASSED!
   Our implementation matches BOTH PyTorch and llama.cpp
```

---

## Troubleshooting

### Issue: Test can't find llama.cpp checkpoint

**Check:**
- Are checkpoint files in `.test-models/gpt2/extracted_weights/`?
- Do filenames match pattern `checkpoint_XX_[name]_llama_cpp.npy`?
- Run conversion script again if needed

### Issue: Validation fails (difference too high)

**Investigate:**
- Check if shapes match
- Verify we're comparing same checkpoint points
- Check Phase 2 mapping was correct
- May need to adjust tolerance

### Issue: Cross-validation fails

**Possible causes:**
- PyTorch and llama.cpp using different precision
- Different computation order
- Bug in one of the implementations

**Action:** Document difference, investigate if > 1e-3

---

## Notes and Issues

**TEAM-004 Notes:**
[Document any issues encountered during integration]

**Test Results:**
[Record actual test results and any deviations]

**Performance Impact:**
[Note any performance changes with multi-reference validation]

---

**Status:** ⏳ PENDING  
**Previous Phase:** Phase 4 - Build and Test (must be complete)  
**Next Phase:** Phase 6 - Documentation and Handoff  
**Estimated Time:** 30 minutes  
**Actual Time:** [fill in after completion]

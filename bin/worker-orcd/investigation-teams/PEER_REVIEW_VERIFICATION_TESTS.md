# Peer Review: Team Alpha Verification Tests

**Date**: 2025-10-06 15:33 UTC  
**Reviewer**: Verification Team  
**Status**: 🔄 IN PROGRESS

---

## Executive Summary

This document contains the verification test suite designed to independently validate Team Alpha's findings. Each test targets a specific claim made in their investigation.

---

## Test Suite Overview

### Test 1: cuBLAS Correctness Verification
**Claim**: cuBLAS output matches manual dot product computation within FP16 precision
**Method**: Independent manual computation and comparison
**Status**: ⏳ Pending execution

### Test 2: Hidden State Range Verification
**Claim**: Hidden state values are in normal range [-13.8, 23.9], no NaN or extreme values
**Method**: Statistical analysis of hidden state distribution
**Status**: ⏳ Pending execution

### Test 3: Attention Softmax Verification
**Claim**: Softmax sum before normalization can vary, but weights after normalization sum to 1.0
**Method**: Verify mathematical correctness of softmax implementation
**Status**: ⏳ Pending execution

### Test 4: Argmax Correctness Verification
**Claim**: Argmax correctly identifies token 137131 as having the highest logit (14.71)
**Method**: Independent search for maximum logit value
**Status**: ⏳ Pending execution

---

## Detailed Test Specifications

### Test 1: cuBLAS Verification

**File**: `cuda/src/transformer/qwen_transformer.cpp`  
**Function**: `project_to_vocab`

**Test Implementation**:
```cpp
// Add to project_to_vocab function
if (first_call) {
    fprintf(stderr, "\n[PEER_REVIEW] === TEST 1: cuBLAS VERIFICATION ===\n");
    
    // Test positions: 0, 8850, 44394, 137131
    int test_positions[] = {0, 8850, 44394, 137131};
    int num_tests = 4;
    
    // Copy hidden state to host
    half h_hidden[896];
    cudaMemcpy(h_hidden, hidden_half, 896*sizeof(half), cudaMemcpyDeviceToHost);
    
    // Copy cuBLAS output
    float h_logits[151936];
    cudaMemcpy(h_logits, logits, 151936*sizeof(float), cudaMemcpyDeviceToHost);
    
    bool all_passed = true;
    for (int t = 0; t < num_tests; t++) {
        int pos = test_positions[t];
        
        // Manual computation: logit[pos] = sum(hidden[j] * lm_head[j][pos])
        // lm_head is stored row-major [896, 151936]
        // So lm_head[j][pos] is at: lm_head_half + j*151936 + pos
        float manual_logit = 0.0f;
        for (int j = 0; j < 896; j++) {
            half lm_weight;
            cudaMemcpy(&lm_weight, lm_head_half + j*151936 + pos, sizeof(half), cudaMemcpyDeviceToHost);
            manual_logit += __half2float(h_hidden[j]) * __half2float(lm_weight);
        }
        
        float cublas_logit = h_logits[pos];
        float diff = fabs(manual_logit - cublas_logit);
        
        fprintf(stderr, "[PEER_REVIEW] Position %d:\n", pos);
        fprintf(stderr, "  Manual:  %.6f\n", manual_logit);
        fprintf(stderr, "  cuBLAS:  %.6f\n", cublas_logit);
        fprintf(stderr, "  Diff:    %.6f\n", diff);
        
        if (diff < 0.0001) {
            fprintf(stderr, "  ✅ PASS (diff < 0.0001)\n");
        } else {
            fprintf(stderr, "  ❌ FAIL (diff >= 0.0001)\n");
            all_passed = false;
        }
    }
    
    fprintf(stderr, "\n[PEER_REVIEW] Test 1 Result: %s\n", 
            all_passed ? "✅ ALL TESTS PASSED" : "❌ SOME TESTS FAILED");
    fprintf(stderr, "[PEER_REVIEW] Team Alpha Claim: %s\n\n",
            all_passed ? "VERIFIED ✅" : "DISPUTED ❌");
}
```

**Expected Result**: All differences < 0.0001, confirming Team Alpha's claim

---

### Test 2: Hidden State Range Verification

**File**: `cuda/src/transformer/qwen_transformer.cpp`  
**Function**: `project_to_vocab`

**Test Implementation**:
```cpp
if (first_call) {
    fprintf(stderr, "\n[PEER_REVIEW] === TEST 2: HIDDEN STATE VERIFICATION ===\n");
    
    // Copy entire hidden state
    half h_hidden[896];
    cudaMemcpy(h_hidden, hidden_half, 896*sizeof(half), cudaMemcpyDeviceToHost);
    
    // Statistical analysis
    float min_val = INFINITY;
    float max_val = -INFINITY;
    float sum = 0.0f;
    float sum_sq = 0.0f;
    int nan_count = 0;
    int inf_count = 0;
    
    for (int i = 0; i < 896; i++) {
        float val = __half2float(h_hidden[i]);
        
        if (isnan(val)) {
            nan_count++;
            continue;
        }
        if (isinf(val)) {
            inf_count++;
            continue;
        }
        
        if (val < min_val) min_val = val;
        if (val > max_val) max_val = val;
        sum += val;
        sum_sq += val * val;
    }
    
    float mean = sum / 896.0f;
    float variance = (sum_sq / 896.0f) - (mean * mean);
    float std_dev = sqrtf(variance);
    
    fprintf(stderr, "[PEER_REVIEW] Hidden State Statistics:\n");
    fprintf(stderr, "  Range: [%.4f, %.4f]\n", min_val, max_val);
    fprintf(stderr, "  Mean: %.4f\n", mean);
    fprintf(stderr, "  Std Dev: %.4f\n", std_dev);
    fprintf(stderr, "  NaN count: %d\n", nan_count);
    fprintf(stderr, "  Inf count: %d\n", inf_count);
    
    // Verify Team Alpha's claims
    bool range_ok = (min_val >= -20.0f && max_val <= 30.0f);
    bool no_nan = (nan_count == 0);
    bool no_inf = (inf_count == 0);
    
    fprintf(stderr, "\n[PEER_REVIEW] Checks:\n");
    fprintf(stderr, "  Range in [-20, 30]: %s\n", range_ok ? "✅ PASS" : "❌ FAIL");
    fprintf(stderr, "  No NaN values: %s\n", no_nan ? "✅ PASS" : "❌ FAIL");
    fprintf(stderr, "  No Inf values: %s\n", no_inf ? "✅ PASS" : "❌ FAIL");
    
    bool all_passed = range_ok && no_nan && no_inf;
    fprintf(stderr, "\n[PEER_REVIEW] Test 2 Result: %s\n", 
            all_passed ? "✅ ALL CHECKS PASSED" : "❌ SOME CHECKS FAILED");
    fprintf(stderr, "[PEER_REVIEW] Team Alpha Claim: %s\n\n",
            all_passed ? "VERIFIED ✅" : "DISPUTED ❌");
}
```

**Expected Result**: Range approximately [-13.8, 23.9], no NaN/Inf values

---

### Test 3: Attention Softmax Verification

**File**: `cuda/kernels/gqa_attention.cu`  
**Function**: `gqa_attention_kernel`

**Test Implementation**:
```cpp
// Add after softmax normalization (around line 213)
if (tid == 0 && batch == 0 && q_head == 0 && cache_len < 5) {
    printf("\n[PEER_REVIEW] === TEST 3: SOFTMAX VERIFICATION ===\n");
    
    // Verify sum of normalized weights
    float weight_sum = 0.0f;
    for (int i = 0; i <= cache_len; i++) {
        weight_sum += scores[i];
    }
    
    printf("[PEER_REVIEW] Softmax Statistics:\n");
    printf("  Sum before norm: %.6f (Team Alpha reported: ~1.97)\n", sum_exp[0]);
    printf("  Sum after norm:  %.6f (should be 1.0)\n", weight_sum);
    
    // Check if sum is close to 1.0
    float diff_from_one = fabs(weight_sum - 1.0f);
    bool sum_correct = (diff_from_one < 0.001f);
    
    printf("\n[PEER_REVIEW] Checks:\n");
    printf("  Weight sum ≈ 1.0: %s (diff=%.6f)\n", 
           sum_correct ? "✅ PASS" : "❌ FAIL", diff_from_one);
    
    printf("\n[PEER_REVIEW] Test 3 Result: %s\n", 
           sum_correct ? "✅ TEST PASSED" : "❌ TEST FAILED");
    printf("[PEER_REVIEW] Team Alpha Claim: %s\n\n",
           sum_correct ? "VERIFIED ✅" : "DISPUTED ❌");
}
```

**Expected Result**: Normalized weights sum to 1.0 within 0.001 tolerance

---

### Test 4: Argmax Verification

**File**: `cuda/kernels/sampling_wrapper.cu`  
**Function**: `argmax_kernel`

**Test Implementation**:
```cpp
// Add inside argmax_kernel after finding max
if (threadIdx.x == 0 && blockIdx.x == 0) {
    static int verification_count = 0;
    
    if (verification_count == 0) {
        printf("\n[PEER_REVIEW] === TEST 4: ARGMAX VERIFICATION ===\n");
        
        // Independent verification: scan all logits
        float verified_max = -INFINITY;
        int verified_idx = -1;
        
        for (int i = 0; i < vocab_size; i++) {
            if (logits[i] > verified_max) {
                verified_max = logits[i];
                verified_idx = i;
            }
        }
        
        printf("[PEER_REVIEW] Argmax Results:\n");
        printf("  Original max: %.6f at token %d\n", max_val, max_idx);
        printf("  Verified max: %.6f at token %d\n", verified_max, verified_idx);
        
        bool indices_match = (max_idx == verified_idx);
        bool values_match = (fabs(max_val - verified_max) < 0.0001f);
        
        printf("\n[PEER_REVIEW] Checks:\n");
        printf("  Indices match: %s\n", indices_match ? "✅ PASS" : "❌ FAIL");
        printf("  Values match:  %s\n", values_match ? "✅ PASS" : "❌ FAIL");
        
        // Check if token 137131 is indeed the max (as Team Alpha claimed)
        bool is_token_137131 = (verified_idx == 137131);
        printf("  Token is 137131: %s (Team Alpha's observation)\n", 
               is_token_137131 ? "✅ CONFIRMED" : "❌ DIFFERENT TOKEN");
        
        bool all_passed = indices_match && values_match;
        printf("\n[PEER_REVIEW] Test 4 Result: %s\n", 
               all_passed ? "✅ TEST PASSED" : "❌ TEST FAILED");
        printf("[PEER_REVIEW] Team Alpha Claim: %s\n\n",
               all_passed ? "VERIFIED ✅" : "DISPUTED ❌");
        
        verification_count++;
    }
}
```

**Expected Result**: Argmax correctly identifies maximum, likely token 137131 with logit ~14.71

---

## Test Execution Plan

### Step 1: Add Test Code
Add all test implementations to their respective files with `[PEER_REVIEW]` tags

### Step 2: Compile and Run
```bash
cd bin/worker-orcd
cargo clean
cargo build --release --features cuda
cargo test --release --test haiku_generation_anti_cheat \
  test_haiku_generation_stub_pipeline_only \
  --features cuda -- --ignored --nocapture --test-threads=1
```

### Step 3: Collect Results
Capture all `[PEER_REVIEW]` output and analyze results

### Step 4: Document Findings
Create final peer review document with verification status for each claim

---

## Success Criteria

For Team Alpha's findings to be **VERIFIED**:
- ✅ Test 1: All cuBLAS differences < 0.0001
- ✅ Test 2: Hidden state in reasonable range, no NaN/Inf
- ✅ Test 3: Normalized attention weights sum to 1.0 ± 0.001
- ✅ Test 4: Argmax correctly identifies maximum logit

If all tests pass, we add to their comments:
```
// [PEER_REVIEW] Verified by independent testing on 2025-10-06 15:33 UTC
// All claims confirmed through automated test suite
```

---

## Risk Assessment

**Low Risk Tests**:
- Test 2 (Hidden State): Read-only, no computation
- Test 4 (Argmax): Simple verification

**Medium Risk Tests**:
- Test 1 (cuBLAS): Requires many GPU memory copies, may be slow
- Test 3 (Softmax): Runs in kernel, minimal overhead

**Mitigation**:
- All tests only run on first call (first_call flag)
- Tests are read-only, no state modification
- Can be disabled by removing test code after verification

---

## Next Steps

1. Implement all four tests in their respective files
2. Run test suite and capture output
3. Analyze results and document verification status
4. Update Team Alpha's comments with peer review confirmation
5. Create final peer review report

---

**Status**: Test suite specification complete, ready for implementation

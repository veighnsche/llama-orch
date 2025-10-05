# MXFP4 Validation Framework

**Version**: 1.0  
**Created**: 2025-10-05  
**Owner**: GPT-Gamma  
**Purpose**: Define validation strategy for MXFP4 implementation

---

## Overview

This document specifies the validation framework for MXFP4 quantization in worker-orcd. The framework ensures numerical correctness, performance acceptability, and production readiness.

---

## 1. Validation Levels

### 1.1 Unit Tests (GT-030)
**Scope**: Individual MXFP4 dequantization operations  
**Goal**: Verify correct dequantization formula implementation

### 1.2 Integration Tests (GT-038)
**Scope**: MXFP4 weights in full inference pipeline  
**Goal**: Verify end-to-end correctness with GPT-OSS-20B

### 1.3 Numerical Validation (GT-038)
**Scope**: Output quality metrics  
**Goal**: Ensure ±1% accuracy vs Q4_K_M baseline

---

## 2. Unit Test Specifications

### 2.1 Test Cases

```rust
// Test 1: Zero value
#[test]
fn test_mxfp4_dequant_zero() {
    let fp4 = 0b0000;
    let fp8_scale = 0x7F;  // 2^0 = 1.0
    assert_eq!(mxfp4_dequant(fp4, fp8_scale), 0.0);
}

// Test 2: Positive values
#[test]
fn test_mxfp4_dequant_positive_range() {
    let test_cases = vec![
        (0b0001, 0x7F, 0.5),   // 0.5 * 1.0
        (0b0010, 0x7F, 1.0),   // 1.0 * 1.0
        (0b0011, 0x7F, 1.5),   // 1.5 * 1.0
        (0b0111, 0x7F, 3.5),   // 3.5 * 1.0
    ];
    for (fp4, scale, expected) in test_cases {
        assert_approx_eq!(mxfp4_dequant(fp4, scale), expected, 1e-6);
    }
}

// Test 3: Negative values
#[test]
fn test_mxfp4_dequant_negative_range() {
    let test_cases = vec![
        (0b1001, 0x7F, -0.5),  // -0.5 * 1.0
        (0b1010, 0x7F, -1.0),  // -1.0 * 1.0
        (0b1111, 0x7F, -3.5),  // -3.5 * 1.0
    ];
    for (fp4, scale, expected) in test_cases {
        assert_approx_eq!(mxfp4_dequant(fp4, scale), expected, 1e-6);
    }
}

// Test 4: Scale factor variations
#[test]
fn test_mxfp4_dequant_scale_variations() {
    let test_cases = vec![
        (0b0010, 0x7E, 0.5),   // 1.0 * 2^(-1) = 0.5
        (0b0010, 0x7F, 1.0),   // 1.0 * 2^0 = 1.0
        (0b0010, 0x80, 2.0),   // 1.0 * 2^1 = 2.0
        (0b0010, 0x81, 4.0),   // 1.0 * 2^2 = 4.0
    ];
    for (fp4, scale, expected) in test_cases {
        assert_approx_eq!(mxfp4_dequant(fp4, scale), expected, 1e-6);
    }
}

// Test 5: Block dequantization
#[test]
fn test_mxfp4_dequant_block() {
    let block = create_test_block();
    let output = mxfp4_dequant_block(&block);
    assert_eq!(output.len(), 32);
    // Validate each element
    for i in 0..32 {
        assert!(output[i].is_finite());
    }
}

// Test 6: Edge cases
#[test]
fn test_mxfp4_dequant_edge_cases() {
    // Maximum positive
    assert_approx_eq!(mxfp4_dequant(0b0111, 0xFF), f16::INFINITY, 1e-6);
    // Maximum negative
    assert_approx_eq!(mxfp4_dequant(0b1111, 0xFF), f16::NEG_INFINITY, 1e-6);
    // Zero scale
    assert_eq!(mxfp4_dequant(0b0010, 0x00), 0.0);
}
```

### 2.2 Acceptance Criteria

✅ All unit tests pass  
✅ Dequantization formula matches spec  
✅ Edge cases handled correctly  
✅ No NaN or Inf for valid inputs  
✅ Numerical precision within FP16 epsilon  

---

## 3. Integration Test Specifications

### 3.1 Test Scenarios

**Scenario 1: Single Layer Forward Pass**
```rust
#[test]
fn test_mxfp4_attention_qkv_projection() {
    let model = load_gpt_oss_20b_mxfp4();
    let input = create_test_input();  // FP16 hidden states
    
    // Run attention with MXFP4 weights
    let output = model.attention_layer.forward(input);
    
    // Validate output shape and range
    assert_eq!(output.shape(), expected_shape);
    assert!(output.all_finite());
}
```

**Scenario 2: Full Model Inference**
```rust
#[test]
fn test_mxfp4_full_inference() {
    let model = load_gpt_oss_20b_mxfp4();
    let prompt = "The capital of France is";
    
    // Generate tokens
    let output = model.generate(prompt, max_tokens=10);
    
    // Validate output
    assert!(output.len() > 0);
    assert!(output.is_valid_text());
}
```

**Scenario 3: Comparison with Q4_K_M Baseline**
```rust
#[test]
fn test_mxfp4_vs_q4km_baseline() {
    let model_mxfp4 = load_gpt_oss_20b_mxfp4();
    let model_q4km = load_gpt_oss_20b_q4km();
    let prompt = "The capital of France is";
    
    // Generate with both models
    let output_mxfp4 = model_mxfp4.generate(prompt, seed=42, temp=0.0);
    let output_q4km = model_q4km.generate(prompt, seed=42, temp=0.0);
    
    // Compare outputs
    let token_match_rate = compare_tokens(&output_mxfp4, &output_q4km);
    assert!(token_match_rate >= 0.95);  // ≥95% match
}
```

### 3.2 Acceptance Criteria

✅ MXFP4 model loads successfully  
✅ Inference completes without errors  
✅ Output is valid text  
✅ ≥95% token match vs Q4_K_M baseline  
✅ No VRAM leaks or crashes  

---

## 4. Numerical Validation

### 4.1 Perplexity Validation

**Dataset**: WikiText-2 (validation set)  
**Metric**: Perplexity (lower is better)  
**Baseline**: Q4_K_M quantization  
**Target**: MXFP4 perplexity within ±1% of Q4_K_M

```rust
#[test]
fn test_mxfp4_perplexity() {
    let model_mxfp4 = load_gpt_oss_20b_mxfp4();
    let model_q4km = load_gpt_oss_20b_q4km();
    let dataset = load_wikitext2_validation();
    
    let ppl_mxfp4 = calculate_perplexity(&model_mxfp4, &dataset);
    let ppl_q4km = calculate_perplexity(&model_q4km, &dataset);
    
    let relative_diff = (ppl_mxfp4 - ppl_q4km).abs() / ppl_q4km;
    assert!(relative_diff <= 0.01);  // ±1% tolerance
}
```

### 4.2 Token Accuracy Validation

**Test Set**: 100 diverse prompts  
**Metric**: Exact token match rate  
**Target**: ≥95% match vs Q4_K_M

```rust
#[test]
fn test_mxfp4_token_accuracy() {
    let model_mxfp4 = load_gpt_oss_20b_mxfp4();
    let model_q4km = load_gpt_oss_20b_q4km();
    let prompts = load_test_prompts();  // 100 prompts
    
    let mut total_tokens = 0;
    let mut matching_tokens = 0;
    
    for prompt in prompts {
        let tokens_mxfp4 = model_mxfp4.generate(prompt, seed=42, temp=0.0);
        let tokens_q4km = model_q4km.generate(prompt, seed=42, temp=0.0);
        
        total_tokens += tokens_mxfp4.len();
        matching_tokens += count_matching_tokens(&tokens_mxfp4, &tokens_q4km);
    }
    
    let accuracy = matching_tokens as f32 / total_tokens as f32;
    assert!(accuracy >= 0.95);  // ≥95% accuracy
}
```

### 4.3 Embedding Similarity Validation

**Metric**: Cosine similarity of embeddings  
**Target**: ≥0.99 similarity vs Q4_K_M

```rust
#[test]
fn test_mxfp4_embedding_similarity() {
    let model_mxfp4 = load_gpt_oss_20b_mxfp4();
    let model_q4km = load_gpt_oss_20b_q4km();
    let prompts = load_test_prompts();
    
    for prompt in prompts {
        let emb_mxfp4 = model_mxfp4.get_embeddings(prompt);
        let emb_q4km = model_q4km.get_embeddings(prompt);
        
        let similarity = cosine_similarity(&emb_mxfp4, &emb_q4km);
        assert!(similarity >= 0.99);
    }
}
```

---

## 5. Performance Validation

### 5.1 VRAM Usage

**Target**: GPT-OSS-20B fits in 24 GB VRAM

```rust
#[test]
fn test_mxfp4_vram_usage() {
    let initial_vram = get_vram_usage();
    let model = load_gpt_oss_20b_mxfp4();
    let final_vram = get_vram_usage();
    
    let vram_used = final_vram - initial_vram;
    assert!(vram_used <= 24_000_000_000);  // 24 GB
}
```

### 5.2 Inference Speed

**Target**: Faster than FP16 (due to memory bandwidth savings)

```rust
#[test]
fn test_mxfp4_inference_speed() {
    let model = load_gpt_oss_20b_mxfp4();
    let prompt = "The capital of France is";
    
    let start = Instant::now();
    let output = model.generate(prompt, max_tokens=100);
    let duration = start.elapsed();
    
    // Should complete in reasonable time
    assert!(duration.as_secs() < 60);  // < 1 minute for 100 tokens
}
```

---

## 6. Test Data

### 6.1 Test Prompts

**Diverse prompt set** (100 prompts):
- Short prompts (1-10 tokens)
- Medium prompts (10-50 tokens)
- Long prompts (50-100 tokens)
- Various domains (code, math, creative writing, factual)

**Example prompts**:
```
"The capital of France is"
"Write a Python function to calculate factorial"
"Explain quantum mechanics in simple terms"
"What is 2 + 2?"
"Once upon a time, in a land far away,"
```

### 6.2 Golden Reference Data

**Q4_K_M Baseline**:
- Pre-generated outputs for all test prompts
- Stored in `tests/data/q4km_baseline.json`
- Used for comparison with MXFP4 outputs

**Format**:
```json
{
  "model": "GPT-OSS-20B",
  "quantization": "Q4_K_M",
  "prompts": [
    {
      "text": "The capital of France is",
      "seed": 42,
      "temperature": 0.0,
      "output_tokens": [464, 3139, 315, 9822, 374],
      "output_text": " Paris"
    }
  ]
}
```

---

## 7. Validation Workflow

### 7.1 Development Phase

1. **Implement dequantization kernel** (GT-029)
2. **Run unit tests** (GT-030)
3. **Fix bugs until all unit tests pass**
4. **Integrate into weight loading** (GT-031, GT-033-GT-037)
5. **Run integration tests** (GT-038)
6. **Fix bugs until integration tests pass**

### 7.2 Validation Phase

1. **Generate Q4_K_M baseline** (if not exists)
2. **Run MXFP4 inference on test set**
3. **Compare outputs**:
   - Token accuracy
   - Perplexity
   - Embedding similarity
4. **Analyze failures**:
   - Identify problematic prompts
   - Debug numerical issues
   - Adjust tolerance if needed
5. **Iterate until validation passes**

### 7.3 Production Readiness

✅ All unit tests pass  
✅ All integration tests pass  
✅ Numerical validation passes (±1% tolerance)  
✅ Performance validation passes (VRAM, speed)  
✅ No memory leaks or crashes  
✅ Documentation complete  

---

## 8. Failure Analysis

### 8.1 Common Failure Modes

**Failure 1: Incorrect dequantization**
- **Symptom**: Garbage output, NaN values
- **Cause**: Wrong formula, byte order, scale calculation
- **Fix**: Review dequantization kernel, add debug logging

**Failure 2: Low token accuracy (<95%)**
- **Symptom**: Different tokens vs Q4_K_M
- **Cause**: Numerical instability, accumulation errors
- **Fix**: Use FP32 accumulation, adjust tolerance

**Failure 3: High perplexity (>1% diff)**
- **Symptom**: Model quality degraded
- **Cause**: Quantization error too high, calibration issues
- **Fix**: Re-quantize model with better calibration

**Failure 4: VRAM overflow**
- **Symptom**: OOM errors, crashes
- **Cause**: Memory leak, incorrect allocation
- **Fix**: Profile VRAM usage, fix leaks

### 8.2 Debug Tools

**CUDA Debugging**:
```bash
# Check for CUDA errors
cuda-memcheck ./worker-orcd

# Profile VRAM usage
nvidia-smi -l 1

# Profile kernel performance
nsys profile ./worker-orcd
```

**Rust Debugging**:
```bash
# Run tests with backtrace
RUST_BACKTRACE=1 cargo test

# Run tests with logging
RUST_LOG=debug cargo test

# Run specific test
cargo test test_mxfp4_dequant_zero -- --nocapture
```

---

## 9. Acceptance Criteria Summary

### 9.1 Unit Tests (GT-030)
✅ All unit tests pass (10+ tests)  
✅ Edge cases handled correctly  
✅ No NaN or Inf for valid inputs  

### 9.2 Integration Tests (GT-038)
✅ MXFP4 model loads successfully  
✅ Full inference completes without errors  
✅ Output is valid text  

### 9.3 Numerical Validation (GT-038)
✅ Token accuracy ≥95% vs Q4_K_M  
✅ Perplexity within ±1% of Q4_K_M  
✅ Embedding similarity ≥0.99  

### 9.4 Performance Validation (GT-038)
✅ GPT-OSS-20B fits in 24 GB VRAM  
✅ Inference completes in reasonable time  
✅ No memory leaks  

---

## 10. References

- MXFP4 Research: `mxfp4-research.md`
- Unit Test Story: GT-030
- Integration Test Story: GT-038
- Spec: `bin/.specs/01_M0_worker_orcd.md` §6.2 (M0-W-1201)

---

**Framework Complete**: ✅  
**Ready for Implementation**: GT-030 (Unit Tests), GT-038 (Integration Tests)

---
Crafted by GPT-Gamma 🤖

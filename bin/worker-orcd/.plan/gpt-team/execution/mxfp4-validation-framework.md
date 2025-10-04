# MXFP4 Validation Framework

**Team**: GPT-Gamma  
**Purpose**: Comprehensive validation strategy for MXFP4 quantization

---

## Overview

MXFP4 is a novel 4-bit quantization format critical for fitting GPT-OSS-20B in 24GB VRAM. This framework ensures MXFP4 implementation is correct, numerically accurate, and performant.

---

## MXFP4 Format Specification

### Block Structure
- **Block size**: 32 FP4 values
- **Shared scale**: 1 FP8 exponent per block
- **Total bytes**: 17 bytes per block (16 bytes FP4 + 1 byte FP8)

### Dequantization Formula
```
fp16_value = fp4_mantissa * fp8_scale
```

### Accuracy Target
- **Tolerance**: ±1% vs FP16 reference
- **Rationale**: 4-bit quantization inherent precision limit

---

## Validation Levels

### Level 1: Block-Level Validation
**Story**: GT-029, GT-030  
**Focus**: Individual MXFP4 block dequantization

**Tests**:
- [ ] Block parsing correctness
- [ ] FP4 mantissa extraction (4-bit unpacking)
- [ ] FP8 scale extraction
- [ ] Dequantization formula correctness
- [ ] Numerical accuracy per block (±1%)

**Test Data**:
- Known MXFP4 blocks from spec
- Reference FP16 values
- Edge cases (zero, max values, negative)

---

### Level 2: Tensor-Level Validation
**Story**: GT-033, GT-034  
**Focus**: Full tensor dequantization

**Tests**:
- [ ] Multi-block tensor dequantization
- [ ] Tensor alignment (256-byte boundaries)
- [ ] Large tensor handling (millions of elements)
- [ ] Memory access patterns
- [ ] Performance: <0.5ms for large matrices

**Test Data**:
- Full weight tensors from GPT-OSS-20B
- Reference FP16 tensors

---

### Level 3: Operation-Level Validation
**Story**: GT-035, GT-036, GT-037  
**Focus**: MXFP4 in compute operations

**Tests**:
- [ ] MXFP4 GEMM correctness
- [ ] MXFP4 embedding lookup correctness
- [ ] MXFP4 attention Q/K/V correctness
- [ ] MXFP4 FFN projection correctness
- [ ] MXFP4 LM head correctness
- [ ] Numerical accuracy per operation (±1%)

**Test Data**:
- Known input activations
- Reference FP16 operation outputs
- Compare MXFP4 vs FP16 results

---

### Level 4: End-to-End Validation
**Story**: GT-038, GT-040  
**Focus**: Full model inference with MXFP4

**Tests**:
- [ ] Full forward pass accuracy
- [ ] Token generation quality
- [ ] Reproducibility (temp=0)
- [ ] VRAM usage (<24GB)
- [ ] Performance vs Q4_K_M baseline

**Test Data**:
- Standard prompts
- Reference outputs from Q4_K_M
- Quality metrics (perplexity, coherence)

---

## Validation Procedure

### Step 1: Unit Tests (GT-030)
```bash
cargo test --package worker-orcd --lib mxfp4
```

**Pass Criteria**:
- All unit tests passing
- Accuracy within ±1% for all test cases

---

### Step 2: Integration Tests (GT-033 to GT-037)
```bash
cargo test --package worker-orcd --test mxfp4_integration
```

**Pass Criteria**:
- All operations produce correct results
- Accuracy within ±1% end-to-end

---

### Step 3: Numerical Validation (GT-038)
```bash
cargo test --package worker-orcd --test mxfp4_numerical_validation
```

**Pass Criteria**:
- Full forward pass within ±1% of FP16
- All layers validated individually

---

### Step 4: End-to-End Validation (GT-040)
```bash
cargo run --bin worker-orcd -- \
  --model /path/to/gpt-oss-20b-mxfp4.gguf \
  --gpu-device 0
```

**Pass Criteria**:
- Model loads successfully
- VRAM usage <24GB
- Text generation quality acceptable
- Performance comparable to Q4_K_M

---

## Regression Testing (GT-043)

### Purpose
Prevent accuracy degradation over time as code evolves.

### Approach
1. Capture baseline outputs for standard prompts
2. Store reference outputs in `tests/fixtures/mxfp4_baseline/`
3. Run regression tests on every PR
4. Alert if accuracy degrades beyond ±0.1%

### Test Suite
```bash
cargo test --package worker-orcd --test mxfp4_regression
```

---

## Performance Benchmarks

### Dequantization Performance
- **Target**: <0.5ms for large weight matrix
- **Measure**: Kernel execution time
- **Baseline**: FP16 GEMM performance

### GEMM Performance
- **Target**: <10% overhead vs FP16 GEMM
- **Measure**: Full matmul time (dequant + compute)
- **Baseline**: cuBLAS FP16 GEMM

### End-to-End Performance
- **Target**: Comparable to Q4_K_M
- **Measure**: Tokens/second
- **Baseline**: Q4_K_M inference rate

---

## Validation Checklist

### Implementation Complete
- [ ] GT-029: MXFP4 Dequantization Kernel
- [ ] GT-030: MXFP4 Unit Tests
- [ ] GT-033: MXFP4 GEMM Integration
- [ ] GT-034: MXFP4 Embedding Lookup
- [ ] GT-035: MXFP4 Attention Q/K/V
- [ ] GT-036: MXFP4 FFN Projections
- [ ] GT-037: MXFP4 LM Head
- [ ] GT-038: MXFP4 Numerical Validation

### Validation Complete
- [ ] Block-level tests passing
- [ ] Tensor-level tests passing
- [ ] Operation-level tests passing
- [ ] End-to-end tests passing
- [ ] Regression tests established
- [ ] Performance benchmarks complete

### Documentation Complete
- [ ] MXFP4 format documented
- [ ] Dequantization algorithm explained
- [ ] Accuracy expectations documented
- [ ] Performance characteristics documented
- [ ] Troubleshooting guide created

---

## Known Limitations

### Accuracy
- ±1% tolerance is inherent to 4-bit quantization
- Some operations may have slightly higher error
- Accumulation errors possible in deep networks

### Performance
- Dequantization adds overhead vs native FP16
- Target: <10% overhead, may vary by operation
- Memory bandwidth critical for performance

### Hardware
- Requires CUDA compute capability 7.0+
- Performance varies by GPU architecture
- Tensor Cores may not accelerate MXFP4

---

## Escalation

If validation fails:
1. Document failure in this file
2. Create detailed bug report
3. Identify root cause (algorithm, implementation, test)
4. Create fix story
5. Re-run validation after fix

---

**Last Updated**: [Date]  
**Updated By**: GPT-Gamma

---
Validated by Project Management Team 📋

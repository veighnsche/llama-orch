# Gate 2: GPT Basic Working

**Day**: 66  
**Participants**: GPT-Gamma  
**Purpose**: Validate GPT-OSS-20B loads and generates text with Q4_K_M quantization

---

## Gate Overview

Gate 2 validates that the basic GPT-OSS-20B inference pipeline works end-to-end with Q4_K_M quantization. This proves the GPT architecture implementation is correct before adding MXFP4 support.

Passing Gate 2 means the GPT team can proceed with MXFP4 implementation.

---

## Validation Checklist

### Model Loading
- [ ] GPT-OSS-20B GGUF file loads successfully
- [ ] All weights mapped correctly (embeddings, attention, FFN, LM head)
- [ ] Weights loaded to VRAM (no RAM fallback)
- [ ] VRAM usage tracked and reported
- [ ] Model loading completes in <60s

### Inference Pipeline
- [ ] Token + position embeddings applied
- [ ] All transformer layers execute
- [ ] LayerNorm → MHA → Residual → LayerNorm → FFN → Residual per layer
- [ ] Final LayerNorm and LM head projection work
- [ ] Sampling produces valid token IDs

### Text Generation
- [ ] Model generates coherent text from prompt
- [ ] Generation completes without errors
- [ ] Output tokens are valid vocabulary IDs
- [ ] Reproducibility: temp=0 produces same output

### Error Handling
- [ ] Insufficient VRAM detected and reported
- [ ] Invalid model path handled gracefully
- [ ] OOM during inference handled gracefully

---

## Validation Procedure

### Step 1: Load Model
```bash
cargo run --bin worker-orcd -- \
  --worker-id test-gpt \
  --model /path/to/gpt-oss-20b-q4km.gguf \
  --gpu-device 0 \
  --port 8080
```

**Expected Output**: Model loads successfully, worker ready  
**Pass Criteria**: Worker starts and reports ready

### Step 2: Generate Text
```bash
curl -X POST http://localhost:8080/execute \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Once upon a time", "max_tokens": 50, "temperature": 0.0, "seed": 42}'
```

**Expected Output**: Coherent text generation  
**Pass Criteria**: Valid tokens generated, no errors

### Step 3: Verify Reproducibility
```bash
# Run same request twice
curl -X POST http://localhost:8080/execute \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Test prompt", "max_tokens": 10, "temperature": 0.0, "seed": 42}'
```

**Expected Output**: Identical token IDs both times  
**Pass Criteria**: Reproducible output with temp=0

---

## Pass/Fail Criteria

### Pass
All checklist items must be ✅ checked.

**Action if Pass**:
- Mark Gate 2 as complete
- Proceed to Sprint 5 (MXFP4 Dequant)
- Document baseline performance

### Fail
If ANY checklist item is ❌ unchecked:

**Action if Fail**:
1. Identify root cause
2. Create fix stories
3. Block MXFP4 work
4. Re-run Gate 2 after fixes

---

## Deliverables

- [ ] Gate 2 validation report
- [ ] Sample generation outputs
- [ ] Performance measurements

---

**Status**: 📋 Ready for validation  
**Owner**: GPT-Gamma  
**Created**: 2025-10-04

---
Coordinated by Project Management Team 📋

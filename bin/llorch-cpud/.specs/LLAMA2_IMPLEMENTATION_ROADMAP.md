# Llama-2 Implementation Roadmap

**Created by:** TEAM-008  
**Date:** 2025-10-08  
**Status:** Active Development  
**Model:** Llama-2 7B Q8_0 GGUF

---

## Overview

This roadmap guides the implementation of Llama-2 7B inference in llorch-cpud. It replaces the GPT-2 roadmap as part of the Foundation Reset strategic pivot.

**Foundation Model:** `/home/vince/Projects/llama-orch/.test-models/llama2-7b/llama-2-7b.Q8_0.gguf`

---

## Week 1: GGUF Loading & Infrastructure

### Goals
- Load Llama-2 7B GGUF file
- Parse metadata and tensor info
- Extract Q8_0 quantized weights
- Verify all tensors present and correct shapes

### Tasks

#### Day 1-2: GGUF Parser
- [ ] **GGUF-001:** Implement GGUF header parser
  - Magic number validation ("GGUF")
  - Version check (v3)
  - Metadata count and tensor count
  - File: `src/model/gguf_parser.rs`
  - Signature: `// Created by: TEAM-008`

- [ ] **GGUF-002:** Parse metadata key-value pairs
  - Extract all metadata
  - Validate `general.architecture == "llama"`
  - Extract model config (layers, heads, hidden_size, etc.)
  - Build `Llama2Config` struct

- [ ] **GGUF-003:** Parse tensor metadata
  - Tensor names, shapes, types, offsets
  - Validate all required tensors present
  - Map GGUF names to model components

#### Day 3-4: Weight Loading
- [ ] **GGUF-004:** Implement Q8_0 dequantization
  - Block structure: 32 int8 + 1 float32 scale
  - Dequantize: `value = int8 * scale`
  - File: `src/model/quantization.rs`
  - Signature: `// Created by: TEAM-008`

- [ ] **GGUF-005:** Load all model weights
  - Token embeddings: `[32000, 4096]`
  - Layer weights: 32 layers × multiple tensors
  - Output weights: `[4096, 32000]`
  - Store in model struct

- [ ] **GGUF-006:** Verify weight shapes
  - Compare with expected Llama2Config
  - Print summary of loaded tensors
  - Unit test: load model and check shapes

#### Day 5: Testing & Validation
- [ ] **TEST-001:** Unit tests for GGUF parser
  - Test header parsing
  - Test metadata extraction
  - Test tensor info parsing

- [ ] **TEST-002:** Integration test for weight loading
  - Load full Llama-2 7B model
  - Verify all tensors present
  - Check shapes match config

- [ ] **TEST-003:** Proof bundle for loading
  - Generate metadata JSON
  - Record tensor shapes
  - Document loading time

**Week 1 Success Criteria:**
- ✅ Model loads without errors
- ✅ All 32000 vocab embeddings loaded
- ✅ All 32 layers loaded
- ✅ Weight shapes verified
- ✅ Unit tests pass

---

## Week 2: Core Components

### Goals
- Implement RMSNorm
- Implement RoPE (Rotary Position Embeddings)
- Implement SwiGLU FFN
- Pass Checkpoint 1 (RMSNorm)

### Tasks

#### Day 1-2: RMSNorm
- [ ] **NORM-001:** Implement RMSNorm layer
  - File: `src/layers/rms_norm.rs`
  - Signature: `// Created by: TEAM-008`
  - Algorithm: `x / sqrt(mean(x²) + eps) * weight`
  - NO bias term
  - NO mean subtraction

- [ ] **NORM-002:** Unit tests for RMSNorm
  - Test with known inputs
  - Verify output RMS ≈ 1
  - Check numerical stability

- [ ] **CHECKPOINT-001:** Validate RMSNorm
  - Extract reference from llama.cpp
  - Compare first layer, first token
  - Tolerance: 1e-5
  - **CRITICAL:** Must pass before proceeding

#### Day 3-4: RoPE
- [ ] **ROPE-001:** Implement RoPE frequency computation
  - Precompute cos/sin caches
  - File: `src/layers/rope.rs`
  - Signature: `// Created by: TEAM-008`
  - Theta: 10000.0
  - Head dim: 128

- [ ] **ROPE-002:** Implement RoPE application
  - Apply to Q and K (not V)
  - Rotate dimension pairs
  - Handle position offsets for generation

- [ ] **ROPE-003:** Unit tests for RoPE
  - Test frequency computation
  - Test rotation application
  - Verify position independence

#### Day 5: SwiGLU FFN
- [ ] **FFN-001:** Implement SwiGLU activation
  - File: `src/layers/swiglu.rs`
  - Signature: `// Created by: TEAM-008`
  - SiLU: `x * sigmoid(x)`
  - Gate, Up, Down projections

- [ ] **FFN-002:** Implement FFN forward pass
  - Three weight matrices (gate, up, down)
  - NO bias terms
  - Element-wise gating

- [ ] **FFN-003:** Unit tests for SwiGLU
  - Test SiLU activation
  - Test full FFN forward
  - Compare with reference

**Week 2 Success Criteria:**
- ✅ Checkpoint 1 passes (RMSNorm)
- ✅ All core components implemented
- ✅ Unit tests pass
- ✅ Code documented with signatures

---

## Week 3: Attention & Full Inference

### Goals
- Implement multi-head attention with RoPE
- Implement KV cache
- Wire up transformer blocks
- Pass Checkpoints 2-8

### Tasks

#### Day 1-2: Attention Mechanism
- [ ] **ATTN-001:** Implement QKV projection
  - Separate Q, K, V weights
  - NO bias terms
  - File: `src/layers/attention.rs`
  - Signature: `// Created by: TEAM-008`

- [ ] **ATTN-002:** Implement multi-head reshape
  - Reshape to `[batch, num_heads, seq_len, head_dim]`
  - Transpose for attention computation

- [ ] **ATTN-003:** Implement attention scores
  - Scaled dot-product: `QK^T / sqrt(head_dim)`
  - Causal masking for prompt
  - Softmax over keys

- [ ] **ATTN-004:** Implement attention output
  - Apply weights to values
  - Reshape and project
  - Output projection (no bias)

- [ ] **CHECKPOINT-002:** Validate QKV projection
- [ ] **CHECKPOINT-003:** Validate RoPE application
- [ ] **CHECKPOINT-004:** Validate attention scores
- [ ] **CHECKPOINT-005:** Validate attention output

#### Day 3: KV Cache
- [ ] **CACHE-001:** Implement KV cache structure
  - File: `src/cache/kv_cache.rs`
  - Signature: `// Created by: TEAM-008`
  - Per-layer K, V storage
  - Position tracking

- [ ] **CACHE-002:** Implement cache update
  - Append new K, V at current position
  - Handle prompt (multiple tokens) vs generation (single token)

- [ ] **CACHE-003:** Implement cache retrieval
  - Return cached K, V up to current position
  - Efficient slicing

- [ ] **CHECKPOINT-003:** Validate KV cache state

#### Day 4: Transformer Block
- [ ] **BLOCK-001:** Wire up transformer block
  - File: `src/model/transformer_block.rs`
  - Signature: `// Created by: TEAM-008`
  - Pre-norm architecture
  - Residual connections

- [ ] **BLOCK-002:** Implement block forward pass
  - Attention norm → Attention → Residual
  - FFN norm → FFN → Residual

- [ ] **CHECKPOINT-006:** Validate FFN output
- [ ] **CHECKPOINT-007:** Validate first block output

#### Day 5: Full Model
- [ ] **MODEL-001:** Implement full forward pass
  - Token embedding
  - 32 transformer blocks
  - Final RMSNorm
  - LM head projection

- [ ] **MODEL-002:** Implement generation loop
  - Prompt processing (parallel)
  - Token generation (sequential)
  - Cache management

- [ ] **CHECKPOINT-008:** Validate full logits

**Week 3 Success Criteria:**
- ✅ Checkpoints 2-8 pass
- ✅ Full forward pass works
- ✅ Generation loop implemented
- ✅ KV cache working

---

## Week 4: Sampling & Production Ready

### Goals
- Implement sampling strategies
- Integrate with HTTP server
- Pass Checkpoints 9-12
- Performance optimization

### Tasks

#### Day 1: Sampling
- [ ] **SAMPLE-001:** Implement greedy sampling
  - Argmax over logits
  - Deterministic (temp=0)
  - File: `src/model/sampling.rs`
  - Signature: `// Created by: TEAM-008`

- [ ] **SAMPLE-002:** Implement temperature sampling
  - Scale logits by temperature
  - Softmax to probabilities
  - Sample from distribution

- [ ] **SAMPLE-003:** Implement top-p (nucleus) sampling
  - Sort probabilities
  - Cumulative sum
  - Sample from top-p mass

- [ ] **CHECKPOINT-009:** Validate logit selection
- [ ] **CHECKPOINT-010:** Validate greedy sampling
- [ ] **CHECKPOINT-011:** Validate softmax probabilities

#### Day 2: End-to-End Validation
- [ ] **E2E-001:** Run full generation test
  - Prompt: "Hello"
  - Generate 10 tokens
  - Compare with llama.cpp output

- [ ] **CHECKPOINT-012:** End-to-end validation
  - **CRITICAL:** Exact match with reference
  - If passes: Implementation correct!

#### Day 3: HTTP Server Integration
- [ ] **HTTP-001:** Integrate with existing HTTP server
  - File: `src/main.rs`
  - Add Llama-2 model loading
  - Add inference endpoint

- [ ] **HTTP-002:** Add streaming support
  - SSE (Server-Sent Events)
  - Token-by-token streaming
  - Proper error handling

- [ ] **HTTP-003:** Add configuration
  - Model path from config
  - Generation parameters
  - Resource limits

#### Day 4: Performance Optimization
- [ ] **PERF-001:** Profile inference
  - Identify bottlenecks
  - Measure tokens/second

- [ ] **PERF-002:** Optimize hot paths
  - BLAS for matrix multiply (if available)
  - Efficient memory layout
  - Reduce allocations

- [ ] **PERF-003:** Benchmark
  - Compare with llama.cpp
  - Document performance
  - Set baseline metrics

#### Day 5: Documentation & Cleanup
- [ ] **DOC-001:** Update README
  - Llama-2 support documented
  - Usage examples
  - Performance characteristics

- [ ] **DOC-002:** Generate proof bundles
  - All checkpoints documented
  - Validation results
  - Performance benchmarks

- [ ] **CLEAN-001:** Remove GPT-2 code
  - Delete GPT-2 specific files
  - Remove dead code
  - Update imports

- [ ] **CLEAN-002:** Code review
  - All signatures present
  - Documentation complete
  - Tests passing

**Week 4 Success Criteria:**
- ✅ All 12 checkpoints pass
- ✅ HTTP server integrated
- ✅ Performance acceptable
- ✅ Documentation complete
- ✅ Production ready

---

## Checkpoint Reference

| # | Checkpoint | Component | Week | Critical |
|---|------------|-----------|------|----------|
| 1 | RMSNorm Output | Normalization | 2 | 🔴 CRITICAL |
| 2 | QKV Projection | Attention | 3 | 🔴 CRITICAL |
| 3 | After RoPE | Position Encoding | 3 | 🔴 CRITICAL |
| 4 | Attention Scores | Attention | 3 | ⚠️ HIGH |
| 5 | Attention Output | Attention | 3 | ⚠️ HIGH |
| 6 | FFN Output | Feed-Forward | 3 | ⚠️ HIGH |
| 7 | First Block | Architecture | 3 | 🟢 VALIDATION |
| 8 | Full Logits | All Layers | 3 | 🟢 VALIDATION |
| 9 | Selected Logits | Output | 4 | 🔴 CRITICAL |
| 10 | Argmax Sampling | Sampling | 4 | 🔴 CRITICAL |
| 11 | Softmax Probs | Sampling | 4 | ⚠️ MEDIUM |
| 12 | End-to-End | **FINAL** | 4 | 🟢 FINAL |

---

## File Structure

```
bin/llorch-cpud/
├── src/
│   ├── model/
│   │   ├── gguf_parser.rs          # TEAM-008: GGUF format parsing
│   │   ├── quantization.rs         # TEAM-008: Q8_0 dequantization
│   │   ├── llama2_config.rs        # TEAM-008: Model configuration
│   │   ├── llama2_model.rs         # TEAM-008: Full model
│   │   ├── transformer_block.rs    # TEAM-008: Transformer block
│   │   └── sampling.rs             # TEAM-008: Sampling strategies
│   ├── layers/
│   │   ├── rms_norm.rs             # TEAM-008: RMSNorm
│   │   ├── rope.rs                 # TEAM-008: Rotary embeddings
│   │   ├── attention.rs            # TEAM-008: Multi-head attention
│   │   └── swiglu.rs               # TEAM-008: SwiGLU FFN
│   ├── cache/
│   │   └── kv_cache.rs             # TEAM-008: KV cache
│   ├── tensor/
│   │   └── ops.rs                  # Tensor operations (reuse existing)
│   └── main.rs                     # HTTP server (update for Llama-2)
├── tests/
│   ├── test_gguf_parser.rs         # TEAM-008: GGUF tests
│   ├── test_rms_norm.rs            # TEAM-008: RMSNorm tests
│   ├── test_rope.rs                # TEAM-008: RoPE tests
│   ├── test_swiglu.rs              # TEAM-008: SwiGLU tests
│   └── test_e2e.rs                 # TEAM-008: End-to-end tests
└── .proof_bundle/
    └── checkpoint/
        └── <run_id>/               # Checkpoint validation results
```

---

## Dependencies

**Cargo.toml additions:**
```toml
[dependencies]
ndarray = "0.15"
byteorder = "1.5"      # For GGUF binary parsing
memmap2 = "0.9"        # For efficient file mapping
rand = "0.8"           # For sampling
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
```

---

## Reference Extraction

**Extract checkpoints from llama.cpp:**
```bash
cd bin/llorch-cpud/tools/checkpoint-extractor
./build.sh  # Already built by Team 007

# Extract all checkpoints
./build/llorch-checkpoint-extractor \
  /.test-models/llama2-7b/llama-2-7b.Q8_0.gguf \
  "Hello" \
  /tmp/llama2_reference

# Checkpoints saved to /tmp/llama2_reference/
```

---

## Testing Strategy

### Unit Tests
- Each component tested in isolation
- Known inputs with expected outputs
- Numerical stability checks

### Integration Tests
- Full forward pass
- Generation loop
- Cache management

### Validation Tests
- All 12 checkpoints
- Compare with llama.cpp
- Tolerance checks

### Performance Tests
- Tokens per second
- Memory usage
- Latency measurements

---

## Risk Mitigation

### High Risk Items
1. **GGUF Parsing:** Complex binary format
   - Mitigation: Study llama.cpp implementation
   - Fallback: Use existing GGUF library

2. **RoPE Implementation:** Tricky rotation math
   - Mitigation: Unit test thoroughly
   - Reference: Multiple implementations

3. **KV Cache:** Complex state management
   - Mitigation: Careful position tracking
   - Validation: Checkpoint 3

4. **Quantization:** Q8_0 dequantization
   - Mitigation: Verify with known values
   - Test: Load and compare weights

### Medium Risk Items
1. **Performance:** May be slower than llama.cpp
   - Mitigation: Profile and optimize
   - Acceptable: 50% of llama.cpp speed

2. **Memory:** 7B model is large
   - Mitigation: Efficient memory layout
   - Monitor: Memory usage during inference

---

## Success Metrics

### Week 1
- Model loads: ✅
- Weights verified: ✅
- Tests pass: ✅

### Week 2
- Checkpoint 1 passes: ✅
- Core components work: ✅
- Unit tests pass: ✅

### Week 3
- Checkpoints 2-8 pass: ✅
- Full inference works: ✅
- Generation loop works: ✅

### Week 4
- All checkpoints pass: ✅
- HTTP server integrated: ✅
- Production ready: ✅

---

## Sign-off

**Created by:** TEAM-008 (Foundation Implementation)  
**Date:** 2025-10-08  
**Status:** Active development roadmap

This roadmap provides a structured path to implementing Llama-2 7B inference in llorch-cpud.

---

*"Plan the work, work the plan."*  
— TEAM-008, Foundation Implementation Division

**END ROADMAP**

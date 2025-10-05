# GPT Team - Getting Started Guide

**Team**: GPT-Gamma 🤖  
**Purpose**: Quick start guide for continuing GPT-OSS-20B implementation

---

## Overview

This guide helps you continue the GPT team's work on implementing GPT architecture support in worker-orcd. The GPT team is responsible for:

- **HuggingFace tokenizer integration** (for GPT-OSS-20B)
- **GPT-specific CUDA kernels** (LayerNorm, GELU, MHA)
- **MXFP4 quantization** (novel 4-bit format)
- **GPTInferenceAdapter** (architecture adapter pattern)

---

## Current Status

### ✅ Completed (6.5 / 48 stories)

**Sprint 0**: MXFP4 Research
- GT-000: Comprehensive MXFP4 format study
- Documentation: `docs/mxfp4-research.md`, `docs/mxfp4-validation-framework.md`

**Sprint 1**: HF Tokenizer (Partial)
- GT-001: HF tokenizers crate integration ✅
- GT-005: GPT config struct (Rust side) ✅
- Pending: C++ GGUF parser, security validation

**Sprint 2**: GPT Kernels (Partial)
- GT-008: Positional embedding kernel ✅
- GT-009/010: LayerNorm kernel ✅
- GT-012: GELU activation kernel ✅
- GT-016: Integration tests (partial) ✅
- Pending: FFN kernel, residual kernel, comprehensive tests

### 🔄 In Progress

**Sprint 2 Completion**:
- GT-014: GPT FFN kernel (up + GELU + down)
- GT-015: Residual connection kernel
- GT-011/013: Comprehensive unit tests

**Sprint 3**: MHA Attention
- GT-017-023: Multi-head attention implementation

---

## Quick Start

### 1. Review Existing Work

```bash
# Read research documents
cat .plan/gpt-team/docs/mxfp4-research.md
cat .plan/gpt-team/docs/mxfp4-validation-framework.md

# Review implementation summary
cat .plan/gpt-team/IMPLEMENTATION_SUMMARY.md

# Check sprint progress
cat .plan/gpt-team/execution/SPRINT_0_1_PROGRESS.md
cat .plan/gpt-team/execution/SPRINT_2_PROGRESS.md
```

### 2. Understand the Codebase

**Rust Code**:
```bash
# Tokenizer backend
src/tokenizer/hf_json.rs       # HuggingFace tokenizer
src/tokenizer/backend.rs       # Tokenizer abstraction

# Model configuration
src/model/gpt_config.rs        # GPT config struct
```

**CUDA Kernels**:
```bash
# GPT-specific kernels
cuda/kernels/layernorm.cu              # LayerNorm (not RMSNorm)
cuda/kernels/gelu.cu                   # GELU (not SwiGLU)
cuda/kernels/positional_embedding.cu   # Absolute pos (not RoPE)

# Tests
cuda/tests/test_gpt_kernels.cu        # Kernel unit tests
```

### 3. Build and Test

**Build worker-orcd**:
```bash
cd /home/vince/Projects/llama-orch/bin/worker-orcd
cargo build --release
```

**Run Rust tests**:
```bash
# Tokenizer tests
cargo test hf_json
cargo test backend

# Config tests
cargo test gpt_config
```

**Run CUDA tests** (requires CUDA toolkit):
```bash
# Build CUDA tests
cd cuda
mkdir -p build && cd build
cmake ..
make test_gpt_kernels

# Run tests
./test_gpt_kernels
```

---

## Next Stories to Implement

### Priority 1: Complete Sprint 2

#### GT-014: GPT FFN Kernel

**Goal**: Implement feed-forward network (up + GELU + down)

**Files to create**:
- `cuda/kernels/ffn.cu` - FFN implementation

**Implementation**:
```cuda
// Pseudo-code
__global__ void gpt_ffn_kernel(
    half* output,           // [batch, seq, d_model]
    const half* input,      // [batch, seq, d_model]
    const half* w_up,       // [d_model, ffn_size]
    const half* w_down,     // [ffn_size, d_model]
    int batch_size,
    int seq_len,
    int d_model,
    int ffn_size
) {
    // 1. Up projection: hidden = input @ w_up
    // 2. GELU activation: hidden = GELU(hidden)
    // 3. Down projection: output = hidden @ w_down
}
```

**Dependencies**:
- cuBLAS GEMM wrapper (for matrix multiplication)
- GELU kernel (already implemented)

**Testing**:
- Unit tests with known values
- Validate against reference implementation
- Check numerical stability

#### GT-015: Residual Connection Kernel

**Goal**: Element-wise addition for residual connections

**Files to create**:
- `cuda/kernels/residual.cu` (or add to existing file)

**Implementation**:
```cuda
__global__ void residual_add_kernel(
    half* output,
    const half* input,
    const half* residual,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = __hadd(input[idx], residual[idx]);
    }
}
```

**Note**: Basic residual is simple. Consider fused variants:
- Residual + LayerNorm (already implemented)
- Residual + dropout (future)

### Priority 2: Sprint 3 (MHA Attention)

#### GT-017: MHA Attention Prefill

**Goal**: Multi-head attention for prefill phase

**Key differences from GQA**:
- MHA: All heads have separate K/V (num_heads = num_kv_heads)
- GQA: Grouped K/V (num_kv_heads < num_heads)

**Implementation**:
```cuda
// MHA attention
// Q: [batch, num_heads, seq_q, head_dim]
// K: [batch, num_heads, seq_k, head_dim]
// V: [batch, num_heads, seq_v, head_dim]
// Output: [batch, num_heads, seq_q, head_dim]

// 1. Compute attention scores: scores = Q @ K^T / sqrt(head_dim)
// 2. Apply softmax: attn = softmax(scores)
// 3. Apply attention: output = attn @ V
```

**Files to create**:
- `cuda/kernels/mha_attention.cu`

---

## Architecture Differences: GPT vs Llama

### Key Differences

| Component | GPT | Llama |
|-----------|-----|-------|
| **Normalization** | LayerNorm | RMSNorm |
| **Activation** | GELU | SwiGLU |
| **Position** | Absolute learned | RoPE |
| **Attention** | MHA | GQA |
| **FFN** | Standard | Gated |

### Why This Matters

**LayerNorm vs RMSNorm**:
- LayerNorm: Centers by mean, then normalizes by std
- RMSNorm: Only normalizes by RMS (no mean centering)
- Impact: LayerNorm requires 2 passes, RMSNorm requires 1

**GELU vs SwiGLU**:
- GELU: Smooth activation, `x * Φ(x)`
- SwiGLU: Gated activation, `(W1*x ⊙ σ(W2*x)) * W3`
- Impact: SwiGLU requires 2 weight matrices, GELU requires 1

**Absolute vs RoPE**:
- Absolute: Add learned position embeddings to token embeddings
- RoPE: Apply rotations to Q/K in attention
- Impact: Absolute is simpler but less flexible for long contexts

**MHA vs GQA**:
- MHA: Each head has separate K/V
- GQA: Multiple heads share K/V (grouped)
- Impact: MHA uses more memory but may be more expressive

---

## MXFP4 Implementation Guide

### Understanding MXFP4

**Format**: 32 FP4 values + 1 FP8 scale = 17 bytes per block

**Dequantization**:
```cuda
__device__ half mxfp4_dequant(uint8_t fp4_mantissa, half fp8_scale) {
    // Map 4-bit mantissa to float value
    float mantissa_val = fp4_to_float_table[fp4_mantissa];
    
    // Multiply by scale
    float result = mantissa_val * __half2float(fp8_scale);
    
    return __float2half(result);
}
```

**Integration Points**:
- Embedding lookup
- Attention Q/K/V projections
- Attention output projection
- FFN up/down projections
- LM head projection

### Implementation Steps (Sprint 5-6)

1. **GT-029**: Implement dequantization kernel
2. **GT-030**: Unit tests with known values
3. **GT-031-037**: Wire into all weight consumers
4. **GT-038**: End-to-end validation vs Q4_K_M baseline

### Validation Requirements

**Numerical Accuracy**:
- ±1% perplexity vs Q4_K_M baseline
- ≥95% token accuracy
- Cosine similarity ≥0.99

**Performance**:
- GPT-OSS-20B fits in 24GB VRAM
- Faster than FP16 (due to bandwidth savings)

---

## Testing Strategy

### Unit Tests

**Rust**:
```bash
# Run specific test
cargo test test_gpt_config_creation

# Run all GPT tests
cargo test gpt
```

**CUDA**:
```bash
# Build and run
cd cuda/build
make test_gpt_kernels
./test_gpt_kernels
```

### Integration Tests

**Full pipeline test** (future):
```bash
cargo test gpt_oss_20b_inference --release -- --nocapture
```

### Validation Tests

**Numerical validation**:
1. Load GPT-OSS-20B with Q4_K_M (baseline)
2. Load GPT-OSS-20B with MXFP4 (test)
3. Generate text with same seed
4. Compare outputs (token accuracy, perplexity)

---

## Common Issues & Solutions

### Issue 1: CUDA Compilation Errors

**Symptom**: `nvcc` errors during build

**Solution**:
```bash
# Check CUDA installation
nvcc --version

# Ensure CUDA_HOME is set
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

### Issue 2: Tokenizer Loading Fails

**Symptom**: `LoadFailed` error when loading tokenizer.json

**Solution**:
```rust
// Ensure tokenizer.json exists
let path = "models/gpt-oss-20b/tokenizer.json";
assert!(std::path::Path::new(path).exists());

// Load with error handling
let tokenizer = HfJsonTokenizer::from_file(path)?;
```

### Issue 3: Numerical Instability

**Symptom**: NaN or Inf values in output

**Solution**:
- Check epsilon in LayerNorm (should be ~1e-5)
- Validate input ranges (no extreme values)
- Use FP32 accumulation for critical paths
- Add gradient clipping if training

### Issue 4: VRAM Overflow

**Symptom**: OOM errors when loading GPT-OSS-20B

**Solution**:
- Use MXFP4 quantization (3.76x compression)
- Reduce batch size
- Reduce context length
- Enable gradient checkpointing (if training)

---

## Resources

### Documentation

**Internal**:
- `docs/mxfp4-research.md` - MXFP4 format study
- `docs/mxfp4-validation-framework.md` - Validation strategy
- `IMPLEMENTATION_SUMMARY.md` - Complete implementation overview
- `execution/SPRINT_*_PROGRESS.md` - Sprint reports

**External**:
- [OCP MX Specification](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf)
- [MXFP4 Paper](https://arxiv.org/abs/2310.10537)
- [HuggingFace Tokenizers](https://docs.rs/tokenizers/latest/tokenizers/)
- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)

### Code References

**Llama Team** (for comparison):
- `cuda/kernels/rmsnorm.cu` - Compare with LayerNorm
- `cuda/kernels/rope.cu` - Compare with positional embedding
- `cuda/kernels/gqa_attention.cu` - Compare with MHA

**Foundation Team** (dependencies):
- FFI interface definitions
- cuBLAS GEMM wrapper
- Memory management utilities

---

## Contact & Coordination

**Team**: GPT-Gamma 🤖  
**Personality**: Explorer, validation-focused, precision-oriented

**Coordination**:
- Dependencies on Foundation-Alpha (FFI, GEMM)
- Dependencies on Llama-Beta (GGUF patterns)
- Provides: Architecture detection, GPTInferenceAdapter

**Gates**:
- Gate 1 (Day 53): GPT kernels complete
- Gate 2 (Day 66): GPT basic working
- Gate 3 (Day 96): MXFP4 + adapter complete

---

## Quick Reference

### File Locations

```
bin/worker-orcd/
├── src/
│   ├── tokenizer/
│   │   ├── hf_json.rs          # HF tokenizer backend
│   │   └── backend.rs          # Tokenizer abstraction
│   └── model/
│       └── gpt_config.rs       # GPT configuration
├── cuda/
│   ├── kernels/
│   │   ├── layernorm.cu        # LayerNorm kernel
│   │   ├── gelu.cu             # GELU activation
│   │   └── positional_embedding.cu  # Positional embeddings
│   └── tests/
│       └── test_gpt_kernels.cu # Kernel tests
└── .plan/gpt-team/
    ├── docs/                   # Research documents
    ├── execution/              # Progress reports
    ├── stories/                # Story cards
    └── sprints/                # Sprint plans
```

### Build Commands

```bash
# Rust build
cargo build --release

# Rust tests
cargo test

# CUDA build
cd cuda/build && cmake .. && make

# CUDA tests
./test_gpt_kernels

# Run worker
cargo run --release
```

### Key Constants

```rust
// GPT-OSS-20B approximate config
const CONTEXT_LENGTH: u32 = 8192;
const EMBEDDING_LENGTH: u32 = 6144;
const BLOCK_COUNT: u32 = 44;
const ATTENTION_HEAD_COUNT: u32 = 64;
const FFN_LENGTH: u32 = 24576;
const VOCAB_SIZE: u32 = 50257;

// MXFP4 constants
const MXFP4_BLOCK_SIZE: usize = 32;
const MXFP4_BLOCK_BYTES: usize = 17;
const MXFP4_COMPRESSION_RATIO: f32 = 3.76;
```

---

## Next Steps Checklist

- [ ] Review existing implementation (`IMPLEMENTATION_SUMMARY.md`)
- [ ] Understand MXFP4 format (`docs/mxfp4-research.md`)
- [ ] Build and test existing code
- [ ] Implement GT-014 (GPT FFN kernel)
- [ ] Implement GT-015 (Residual connection)
- [ ] Complete Sprint 2 tests
- [ ] Begin Sprint 3 (MHA attention)
- [ ] Reach Gate 1 (Day 53)

---

**Ready to continue**: Pick up at GT-014 (GPT FFN kernel) or GT-017 (MHA attention)

---
Crafted by GPT-Gamma 🤖

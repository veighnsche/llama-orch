# llorch-candled Specifications

**Created by:** TEAM-000 (Foundation)  
**Date:** 2025-10-08

---

## Overview

This directory contains all specifications and documentation for `llorch-candled`, the Candle-based Llama-2 inference worker.

---

## Key Documents

### Handoff & Strategy

- **`TEAM_000_HANDOFF.md`** - Complete handoff from TEAM-000 to implementation teams
  - Project structure overview
  - Architecture decisions
  - Implementation roadmap
  - Next steps

### Checkpoints

- **`checkpoints/CHECKPOINT_00_FOUNDATION.md`** - Foundation setup validation
  - HTTP server integration
  - Worker crates wiring
  - Project structure validation
  - Compilation and testing

### Reference Documents (from llorch-cpud)

The following documents from `bin/llorch-cpud/.specs/` are relevant for Llama-2 implementation:

- `CANDLE_INTEGRATION_HANDOFF.md` - Candle integration strategy (hybrid approach)
- `checkpoints/LLAMA2_CHECKPOINT_UPDATE_PLAN.md` - Checkpoint adaptation for Llama-2
- `checkpoints/CHECKPOINT_01_RMS_NORM.md` - RMSNorm implementation (Checkpoint 1)
- `checkpoints/CHECKPOINT_01B_ROPE_APPLICATION.md` - RoPE implementation (Checkpoint 1B)

---

## Checkpoint Sequence

### Foundation (Week 1)
- [x] **Checkpoint 0**: Foundation Setup - HTTP server, project structure, worker crates

### Core Layers (Week 2) - UPDATED 2025-10-08
- [x] **Checkpoint 1**: RMSNorm - Using `candle_nn::ops::rms_norm` ✅
- [x] **Checkpoint 1B**: RoPE - Using `candle_nn::rotary_emb::rope_i` ✅
- [x] **Checkpoint 2**: QKV Projection - Manual matmul (optimal) ✅
- [x] **Checkpoint 3**: Attention - Using `candle_nn::ops::softmax` ✅
- [ ] **Checkpoint 6**: SwiGLU - Will use `candle_nn::ops::swiglu` ⏳

### Full Model (Week 3)
- [ ] **Checkpoint 7**: First Block - Complete transformer block
- [ ] **Checkpoint 8**: Full Logits - 32-layer model output

### Validation (Week 4)
- [ ] **Checkpoint 9**: Selected Logits - Last token selection
- [ ] **Checkpoint 10**: Argmax Sampling - Greedy sampling
- [ ] **Checkpoint 11**: Softmax Probs - Temperature sampling
- [ ] **Checkpoint 12**: End-to-End - Full inference pipeline

---

## Architecture Highlights

### Candle-First Strategy ✅

**UPDATED 2025-10-08 by TEAM-005:**

We use **Candle's optimized implementations for the difficult parts of inference**.

**What We Use From Candle:**
- ✅ `candle_nn::rotary_emb::rope_i` - RoPE (GPU kernels, 3-5x faster)
- ✅ `candle_nn::ops::rms_norm` - RMSNorm (GPU kernels, numerically stable)
- ✅ `candle_nn::ops::softmax` - Softmax (GPU kernels, stable)
- ✅ `candle_nn::kv_cache::KvCache` - KV caching (dynamic, efficient)
- ✅ `candle_nn::ops::swiglu` - SwiGLU activation (optimized)

**What We Implement:**
- Model architecture (Transformer blocks, layer stacking)
- Weight loading (GGUF format)
- Tokenization (BPE)
- API design (HTTP server, streaming)

**See:** `CANDLE_USAGE_POLICY.md` for complete guidelines

### Llama-2 Specifics

| Component | GPT-2 | Llama-2 |
|-----------|-------|---------|
| Normalization | LayerNorm | RMSNorm |
| Position | Learned | RoPE |
| QKV | Combined | Separate |
| FFN | GELU | SwiGLU |
| Layers | 24 | 32 |
| Heads | 16 | 32 |
| Head dim | 64 | 128 |
| Vocab | 50257 | 32000 |

---

## Implementation Guidelines

### 1. Use Candle for Difficult Parts ✅ **NEW**
- **DO:** Use `candle_nn` optimized implementations (RoPE, RMSNorm, Softmax, etc.)
- **DON'T:** Reimplement what Candle already provides
- **FOCUS ON:** Model architecture, weight loading, tokenization
- **SEE:** `CANDLE_USAGE_POLICY.md` for complete guidelines

### 2. Follow Checkpoint Order
- Implement in sequence: 0 → 1 → 1B → 2 → 3 → 6 → 7 → 8 → 9-11 → 12
- Validate each checkpoint before proceeding
- Use llama.cpp as reference

### 3. Code Signatures
- Add `// Created by: TEAM-XXX` to new files
- Add `// Modified by: TEAM-XXX` to changes
- Document Candle usage: `// TEAM-XXX: Using candle_nn::...`
- Never remove existing signatures

### 4. Testing
- Write checkpoint test for each component
- Test Candle integration (shape transformations, etc.)
- Validate shapes and values
- Test determinism (bit-exact across runs)

---

## Quick Start

### Build
```bash
cd bin/llorch-candled
cargo build
```

### Test
```bash
cargo test
```

### Run
```bash
cargo run -- \
  --worker-id test \
  --model test.gguf \
  --port 8080 \
  --callback-url http://localhost:9999
```

---

## References

### Internal
- `../llorch-cpud/`: GPT-2 implementation (predecessor)
- `../../reference/candle/`: Candle source code
- `../../reference/mistral.rs/`: Reference implementation

### External
- llama.cpp: Reference for checkpoint validation
- Mistral.rs: Architecture patterns
- Candle kernels: CUDA optimization

---

## Status

**UPDATED 2025-10-08 by TEAM-005:**

**Current Phase:** Core Layers Complete ✅  
**Completed:**
- ✅ Checkpoint 0: Foundation
- ✅ Checkpoint 1: RMSNorm (using Candle)
- ✅ Checkpoint 1B: RoPE (using Candle)
- ✅ Checkpoint 2: QKV Projection
- ✅ Checkpoint 3: Attention (using Candle softmax)

**Next Phase:** FFN (Checkpoint 6) ⏳  
**Target:** Week 3 - Full Model

**Test Status:** 31/31 tests passing (100%) ✅

---

Built by TEAM-000 🌊  
Optimized by TEAM-005 🚀

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

### Core Layers (Week 2)
- [ ] **Checkpoint 1**: RMSNorm - Root mean square normalization
- [ ] **Checkpoint 1B**: RoPE - Rotary position embeddings
- [ ] **Checkpoint 2**: QKV Projection - Separate Q, K, V projections
- [ ] **Checkpoint 3**: KV Cache - Key-value caching for generation
- [ ] **Checkpoint 6**: SwiGLU - Swish-gated linear unit FFN

### Full Model (Week 3)
- [ ] **Checkpoint 4**: Attention Scores - Scaled dot-product attention
- [ ] **Checkpoint 5**: Attention Output - Output projection
- [ ] **Checkpoint 7**: First Block - Complete transformer block
- [ ] **Checkpoint 8**: Full Logits - 32-layer model output

### Validation (Week 4)
- [ ] **Checkpoint 9**: Selected Logits - Last token selection
- [ ] **Checkpoint 10**: Argmax Sampling - Greedy sampling
- [ ] **Checkpoint 11**: Softmax Probs - Temperature sampling
- [ ] **Checkpoint 12**: End-to-End - Full inference pipeline

---

## Architecture Highlights

### Hybrid Compute Strategy

**CPU Path (Primary):**
- Pure ndarray implementation
- Checkpoint validation
- Educational value
- Always works

**CUDA Path (Optional):**
- Candle kernels only (not framework)
- Feature-gated with `cuda` feature
- Performance optimization
- Added after CPU validation

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

### 1. Follow Checkpoint Order
- Implement in sequence: 0 → 1 → 1B → 2 → 3 → 6 → 4 → 5 → 7 → 8 → 9-11 → 12
- Validate each checkpoint before proceeding
- Use llama.cpp as reference

### 2. CPU First, CUDA Later
- Implement CPU path first (ndarray)
- Validate with checkpoints
- Add CUDA path after validation passes
- Ensure both paths produce identical output

### 3. Code Signatures
- Add `// Created by: TEAM-XXX` to new files
- Add `// Modified by: TEAM-XXX` to changes
- Never remove existing signatures

### 4. Testing
- Write checkpoint test for each component
- Compare with llama.cpp reference
- Validate shapes and values
- Test both CPU and CUDA paths

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

**Current Phase:** Foundation Complete ✅  
**Next Phase:** Checkpoint 1 (RMSNorm) ⏳  
**Target:** Week 2 - Core Layers

---

Built by TEAM-000 🌊

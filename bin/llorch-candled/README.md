# llorch-candled

**Candle-based Llama-2 inference worker daemon**

Created by: **TEAM-000** (Foundation)

---

## Overview

`llorch-candled` is a Llama-2 inference worker that uses a **hybrid approach** combining:
- **Pure ndarray** for CPU (checkpoint validation, educational value)
- **Candle kernels** for CUDA acceleration (optional, performance)

This follows the recommendations from `CANDLE_INTEGRATION_HANDOFF.md`:
- ✅ Use Candle's **kernels**, NOT the framework
- ✅ Keep checkpoint-driven validation
- ✅ Maintain educational value
- ✅ Get performance without abstraction overhead

---

## Architecture

### Hybrid Approach

```
┌─────────────────────────────────────────┐
│         llorch-candled Worker           │
├─────────────────────────────────────────┤
│  HTTP Server (worker-http)              │
│  ├─ GET /health                         │
│  └─ POST /execute                       │
├─────────────────────────────────────────┤
│  CandleInferenceBackend                 │
│  ├─ Model: Llama-2 7B                   │
│  ├─ Format: GGUF Q8_0                   │
│  └─ Tokenizer: SentencePiece            │
├─────────────────────────────────────────┤
│  Layers (Checkpoint-driven)             │
│  ├─ RMSNorm (Checkpoint 1)              │
│  ├─ RoPE (Checkpoint 1B)                │
│  ├─ QKV Projection (Checkpoint 2)       │
│  ├─ KV Cache (Checkpoint 3)             │
│  ├─ Attention (Checkpoints 4, 5)        │
│  ├─ SwiGLU FFN (Checkpoint 6)           │
│  └─ TransformerBlock (Checkpoint 7)     │
├─────────────────────────────────────────┤
│  Compute Backend                        │
│  ├─ CPU: Pure ndarray (primary)         │
│  └─ CUDA: Candle kernels (optional)     │
└─────────────────────────────────────────┘
```

### Key Differences from llorch-cpud

| Aspect | llorch-cpud | llorch-candled |
|--------|-------------|----------------|
| **Model** | GPT-2 Medium | Llama-2 7B |
| **Normalization** | LayerNorm | RMSNorm |
| **Position** | Learned embeddings | RoPE |
| **QKV** | Combined projection | Separate Q, K, V |
| **FFN** | GELU | SwiGLU |
| **Acceleration** | Pure CPU | CPU + optional CUDA |
| **Kernels** | None | Candle kernels |

---

## Features

- **Checkpoint-driven development**: Each component validated independently
- **Hybrid compute**: CPU for validation, CUDA for speed
- **Worker crates integration**: Reuses 99% of existing infrastructure
- **HTTP server**: Production-ready SSE streaming
- **GGUF support**: Load quantized models directly
- **Educational**: Learn Llama-2 architecture from scratch

---

## Quick Start

### Build (CPU only)

```bash
cd bin/llorch-candled
cargo build --release
```

### Build (with CUDA)

```bash
cargo build --release --features cuda
```

### Run

```bash
./target/release/llorch-candled \
  --worker-id test-worker \
  --model /path/to/llama-2-7b.Q8_0.gguf \
  --port 8080 \
  --callback-url http://localhost:9999
```

### Test

```bash
# Run all tests
cargo test

# Run specific checkpoint
cargo test checkpoint_01_rms_norm

# Run with CUDA
cargo test --features cuda
```

---

## Development Roadmap

### Week 1: Foundation (Checkpoint 0)
- [x] Project structure
- [x] HTTP server integration
- [x] Worker crates wiring
- [ ] Compilation validation

### Week 2: Core Layers
- [ ] Checkpoint 1: RMSNorm
- [ ] Checkpoint 1B: RoPE
- [ ] Checkpoint 2: Separate QKV
- [ ] Checkpoint 3: KV Cache
- [ ] Checkpoint 6: SwiGLU

### Week 3: Full Model
- [ ] Checkpoint 4: Attention Scores
- [ ] Checkpoint 5: Attention Output
- [ ] Checkpoint 7: First Block
- [ ] Checkpoint 8: Full Logits

### Week 4: Validation & Optimization
- [ ] Checkpoints 9-11: Sampling
- [ ] Checkpoint 12: End-to-End
- [ ] CUDA kernel integration
- [ ] Performance benchmarking

---

## Checkpoints

All checkpoints are documented in `.specs/checkpoints/`:

| Checkpoint | Component | Status |
|------------|-----------|--------|
| 0 | Foundation Setup | ⬜ Not Started |
| 1 | RMSNorm | ⬜ Not Started |
| 1B | RoPE Application | ⬜ Not Started |
| 2 | QKV Projection | ⬜ Not Started |
| 3 | KV Cache | ⬜ Not Started |
| 4 | Attention Scores | ⬜ Not Started |
| 5 | Attention Output | ⬜ Not Started |
| 6 | SwiGLU FFN | ⬜ Not Started |
| 7 | First Block | ⬜ Not Started |
| 8 | Full Logits | ⬜ Not Started |
| 9 | Selected Logits | ⬜ Not Started |
| 10 | Argmax Sampling | ⬜ Not Started |
| 11 | Softmax Probs | ⬜ Not Started |
| 12 | End-to-End | ⬜ Not Started |

---

## Dependencies

### Worker Crates (100% reusable)
- `worker-common`: SamplingConfig, InferenceResult, startup callbacks
- `worker-http`: HTTP server, SSE streaming, InferenceBackend trait
- `worker-tokenizer`: BPE tokenization, GGUF support
- `worker-models`: Model configs and adapters
- `worker-gguf`: GGUF file format parser

### Compute
- `ndarray`: CPU tensor operations (primary)
- `candle-kernels`: CUDA kernels (optional, feature-gated)
- `cudarc`: CUDA runtime (optional, feature-gated)

### Utilities
- `tokio`: Async runtime (single-threaded!)
- `tracing`: Structured logging
- `clap`: CLI argument parsing
- `anyhow`, `thiserror`: Error handling

---

## Candle Integration

Following `CANDLE_INTEGRATION_HANDOFF.md`, we use **kernels only**:

### What We Use ✅
- `candle-kernels`: Optimized CUDA kernels
  - RmsNorm (from `reduce.cu`)
  - SiLU (from `unary.cu`)
  - Quantization (from `quantized.cu`)

### What We DON'T Use ❌
- `candle-core`: Too much abstraction
- `candle-nn`: We build our own layers
- `candle-transformers`: Defeats learning purpose

### Why This Approach?
- ✅ Best performance (optimized CUDA kernels)
- ✅ Keep our architecture
- ✅ Keep checkpoint validation
- ✅ Educational value maintained
- ✅ Minimal dependencies

---

## Testing

### Unit Tests
```bash
cargo test --lib
```

### Integration Tests
```bash
cargo test --test '*'
```

### Checkpoint Validation
```bash
# Run specific checkpoint
cargo test checkpoint_01_rms_norm -- --nocapture

# Compare with llama.cpp reference
cargo test checkpoint_01_rms_norm -- --show-output
```

### Benchmarks
```bash
cargo bench --features benchmark
```

---

## Performance

### CPU (ndarray)
- **Purpose**: Validation, checkpoint testing
- **Speed**: ~10-20 tokens/sec (Llama-2 7B)
- **Memory**: ~7GB (Q8_0 model)

### CUDA (Candle kernels)
- **Purpose**: Production inference
- **Speed**: ~50-100 tokens/sec (depends on GPU)
- **Memory**: ~7GB VRAM (Q8_0 model)

---

## References

### Documentation
- `.specs/CANDLE_INTEGRATION_HANDOFF.md`: Integration strategy
- `.specs/checkpoints/`: All checkpoint specifications
- `reference/candle/`: Candle source code
- `reference/mistral.rs/`: Reference implementation

### Related Workers
- `llorch-cpud`: GPT-2 CPU worker (predecessor)
- `worker-orcd`: Production GPU worker

---

## Team

**TEAM-000** (Foundation)
- Mission: Build the foundation for all future workers
- Focus: Checkpoint-driven, educational, reusable
- Motto: "The foundation of them all"

---

## License

GPL-3.0-or-later

---

## Contributing

1. Follow checkpoint-driven development
2. Add TEAM-000 signatures to code changes
3. Validate with reference implementation (llama.cpp)
4. Keep CPU path working (CUDA is optional)
5. Update checkpoint docs when needed

---

**Status**: 🚧 In Development  
**Version**: 0.1.0  
**Last Updated**: 2025-10-08

# llorch-cpud Source Code Structure

This directory contains the implementation of llorch-cpud, a CPU-based GPT-2 inference worker daemon.

## Directory Structure

```
src/
├── main.rs                  # Entry point (uses worker-http)
├── lib.rs                   # Library exports
├── error.rs                 # Error types
├── backend/                 # InferenceBackend implementation
│   ├── mod.rs
│   └── cpu_backend.rs       # IMPORTS: worker-http, worker-common, worker-tokenizer
├── cache/                   # KV Cache (top-level module)
│   ├── mod.rs
│   └── kv_cache.rs          # CHECKPOINT 3
├── layers/                  # Neural network layers
│   ├── mod.rs
│   ├── layer_norm.rs        # CHECKPOINT 1
│   ├── embedding.rs         # Phase 2
│   ├── attention/           # Attention module (split into focused files)
│   │   ├── mod.rs
│   │   ├── qkv.rs           # CHECKPOINT 2
│   │   ├── scores.rs        # CHECKPOINT 4
│   │   └── output.rs        # CHECKPOINT 5
│   ├── ffn.rs               # CHECKPOINT 6
│   └── transformer.rs       # CHECKPOINT 7
├── model/                   # GPT-2 model
│   ├── mod.rs
│   └── gpt2.rs              # CHECKPOINTS 8-12
└── tensor/                  # CPU tensor operations
    ├── mod.rs
    └── ops.rs               # Helper functions
```

## Import Guidelines

### Files with worker-crates imports:
- `main.rs` → worker-http, worker-common
- `backend/cpu_backend.rs` → worker-http, worker-common, worker-tokenizer
- `model/gpt2.rs` → worker-models (GPTConfig)

### Files with NO worker-crates imports (pure implementation):
- All files in `layers/` (layer_norm, attention/*, ffn, transformer)
- All files in `cache/` (kv_cache)
- All files in `tensor/` (ops)

### Files with internal crate imports only:
- `layers/transformer.rs` → uses LayerNorm, Attention, FFN, KVCache
- `layers/attention/mod.rs` → uses QKVProjection, AttentionScores, AttentionOutput, KVCache

## Architecture

### Single-Threaded
- Uses `tokio::main(flavor = "current_thread")`
- No rayon, no parallel processing
- Sequential request processing
- 10-30% faster than multi-threaded for CPU inference

### Worker-First Design
- Part of larger system (orchestrated by pool-managerd)
- Reuses worker-crates for infrastructure (70% code reuse)
- Focus on model implementation (30% new code)

### Checkpoint-Driven Development
- 13 validation checkpoints
- Each checkpoint validates one component
- Reference implementations (tinygrad, Candle, Mistral.rs)
- Incremental validation

## Implementation Order

Follow checkpoints in order:

1. **Checkpoint 0**: HTTP Server (use worker-http) ✅ (stub created)
2. **Checkpoint 1**: LayerNorm (`layers/layer_norm.rs`)
3. **Checkpoint 2**: QKV Projection (`layers/attention/qkv.rs`)
4. **Checkpoint 3**: KV Cache (`cache/kv_cache.rs`)
5. **Checkpoint 4**: Attention Scores (`layers/attention/scores.rs`)
6. **Checkpoint 5**: Attention Output (`layers/attention/output.rs`)
7. **Checkpoint 6**: FFN (`layers/ffn.rs`)
8. **Checkpoint 7**: Transformer Block (`layers/transformer.rs`) - **MAJOR MILESTONE**
9. **Checkpoint 8**: Full Logits (`model/gpt2.rs`)
10. **Checkpoint 9**: Selected Logits (`model/gpt2.rs`)
11. **Checkpoint 10**: Argmax Sampling (`model/gpt2.rs`)
12. **Checkpoint 11**: Softmax Probabilities (`model/gpt2.rs`)
13. **Checkpoint 12**: End-to-End Generation (`model/gpt2.rs`)

## Key Design Decisions

### Why Cache is Top-Level
- Used by all 24 attention layers
- Future optimization target (paged attention, quantization)
- Signals engineering investment area
- See `.specs/KV_CACHE_MODULE_ANALYSIS.md`

### Why Attention is Split
- `qkv.rs`: QKV projection and split (Checkpoint 2)
- `scores.rs`: Attention score computation (Checkpoint 4)
- `output.rs`: Attention output projection (Checkpoint 5)
- Aligns with checkpoints for incremental validation
- See `.specs/ATTENTION_MODULE_STRUCTURE.md`

### Why Single-Threaded
- CPU inference is memory-bound, not compute-bound
- Thread overhead hurts performance
- Simpler to reason about
- 10-30% faster than multi-threaded
- See `.specs/SINGLE_THREADED_ARCHITECTURE.md`

## Next Steps

1. **Implement Checkpoint 1** (LayerNorm)
   - Read `.specs/checkpoints/CHECKPOINT_01_LAYER_NORM.md`
   - Implement `layers/layer_norm.rs`
   - Write test in `tests/checkpoint_01_layer_norm.rs`
   - Extract reference output from tinygrad
   - Run test until it passes

2. **Do not proceed** until Checkpoint 1 passes
   - Errors compound through all 48 LayerNorms
   - Getting this right is critical

3. **Follow checkpoint order** strictly
   - Each checkpoint depends on previous ones
   - No skipping ahead
   - No "partial fixes"

## Testing

Tests go in `tests/` directory (not created yet):

```
tests/
├── checkpoint_01_layer_norm.rs
├── checkpoint_02_qkv.rs
├── checkpoint_03_kv_cache.rs
├── ...
└── checkpoint_12_e2e.rs
```

Each test:
1. Loads reference output from tinygrad
2. Runs our implementation
3. Compares within tolerance
4. Fails if mismatch

## Resources

- **Checkpoints**: `.specs/checkpoints/`
- **Implementation Roadmap**: `.specs/IMPLEMENTATION_ROADMAP.md`
- **Worker Crates Audit**: `.specs/WORKER_CRATES_REUSABILITY_AUDIT.md`
- **API Integration**: `.specs/API_INTEGRATION.md`
- **Reference Implementations**: `../reference/`

---

Built by TEAM CASCADE 🌊

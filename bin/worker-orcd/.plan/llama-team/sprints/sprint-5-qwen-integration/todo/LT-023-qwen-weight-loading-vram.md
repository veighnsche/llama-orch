# LT-023: Qwen Weight Loading to VRAM

**Team**: Llama-Beta  
**Sprint**: Sprint 5 - Qwen Integration  
**Size**: M (2 days)  
**Days**: 58-59  
**Spec Ref**: M0-W-1220, M0-W-1221

---

## Story Description

Load Qwen2.5-0.5B model weights from memory-mapped GGUF file to GPU VRAM using chunked transfer. Transfer all weights (embedding, 24 transformer layers, output) efficiently while tracking progress and validating VRAM residency.

---

## Acceptance Criteria

- [ ] Load token embedding weights to VRAM
- [ ] Load all 24 layer weights to VRAM (attention + FFN)
- [ ] Load output weights to VRAM
- [ ] Use chunked H2D transfer (256MB chunks)
- [ ] Emit progress events for large transfers
- [ ] Validate VRAM residency after loading
- [ ] Calculate and log total VRAM usage
- [ ] Verify VRAM usage matches pre-load validation
- [ ] Unit tests validate weight loading
- [ ] Integration tests validate full model loading
- [ ] Error handling for VRAM allocation failures
- [ ] Error handling for transfer failures
- [ ] Log loading progress and completion

---

## Dependencies

### Upstream (Blocks This Story)
- LT-022: Qwen Weight Mapping (needs weight mapping)
- LT-004: Chunked H2D Transfer (needs transfer function)
- LT-003: Memory-Mapped I/O (needs mmap pointers)

### Downstream (This Story Blocks)
- LT-024: Qwen Forward Pass (needs loaded weights)

---

## Technical Details

### Files to Create/Modify
- `bin/worker-orcd/cuda/src/models/qwen_loader.cpp` - Weight loader
- `bin/worker-orcd/cuda/src/models/qwen_loader.h` - Loader interface
- `bin/worker-orcd/src/models/qwen_loader.rs` - Rust loader wrapper

### Weight Loading Interface
```cpp
struct QwenModel {
    QwenWeights weights;        // Weight pointers (VRAM)
    LlamaConfig config;         // Model configuration
    size_t total_vram_bytes;    // Total VRAM used
};

class QwenLoader {
public:
    // Load Qwen model from GGUF file to VRAM
    static Result<QwenModel> load(
        const std::string& gguf_path,
        const MmapFile& mmap_file,
        const LlamaConfig& config
    );
    
private:
    static void load_embedding(QwenWeights& weights, const MmapFile& mmap);
    static void load_layer(QwenWeights::Layer& layer, int layer_idx, const MmapFile& mmap);
    static void load_output(QwenWeights& weights, const MmapFile& mmap);
    static size_t calculate_vram_usage(const QwenWeights& weights, const LlamaConfig& config);
};
```

```rust
pub struct QwenModel {
    pub weights: QwenWeights,
    pub config: LlamaConfig,
    pub total_vram_bytes: usize,
}

impl QwenLoader {
    pub fn load(
        gguf_path: &Path,
        mmap_file: &MmapFile,
        config: &LlamaConfig,
    ) -> Result<QwenModel, LoadError>;
}
```

### Implementation Notes

**Loading Sequence**:
```cpp
Result<QwenModel> load(const std::string& gguf_path, const MmapFile& mmap, const LlamaConfig& config) {
    QwenModel model;
    model.config = config;
    
    tracing::info!("Loading Qwen2.5-0.5B model from {}", gguf_path);
    
    // 1. Map weights
    auto weight_mapping = map_qwen_weights(mmap);
    
    // 2. Allocate VRAM for all weights
    allocate_vram_for_weights(&model.weights, config);
    
    // 3. Load embedding (vocab_size × hidden_dim)
    load_embedding(model.weights, mmap);
    emit_progress("Embedding loaded", 1, 26);
    
    // 4. Load 24 transformer layers
    for (int layer = 0; layer < 24; ++layer) {
        load_layer(model.weights.layers[layer], layer, mmap);
        emit_progress(fmt::format("Layer {} loaded", layer), layer + 2, 26);
    }
    
    // 5. Load output weights
    load_output(model.weights, mmap);
    emit_progress("Output weights loaded", 26, 26);
    
    // 6. Validate VRAM residency
    validate_vram_residency(model.weights);
    
    // 7. Calculate total VRAM usage
    model.total_vram_bytes = calculate_vram_usage(model.weights, config);
    
    tracing::info!("Model loaded: {} MB VRAM", model.total_vram_bytes / (1024 * 1024));
    
    return Ok(model);
}
```

**Chunked Transfer**:
```cpp
void load_embedding(QwenWeights& weights, const MmapFile& mmap) {
    // Get source pointer from mmap
    const half* src = mmap.get_tensor_data("token_embd.weight");
    size_t size = vocab_size * hidden_dim * sizeof(half);
    
    // Allocate VRAM
    cudaMalloc(&weights.token_embedding, size);
    
    // Chunked transfer
    TransferConfig config = {256 * 1024 * 1024, true};  // 256MB chunks
    ChunkedTransfer::h2d_chunked(
        weights.token_embedding,
        src,
        size,
        config
    );
}
```

**VRAM Allocation**:
```cpp
void allocate_vram_for_weights(QwenWeights* weights, const LlamaConfig& config) {
    // Embedding
    cudaMalloc(&weights->token_embedding, config.vocab_size * config.hidden_dim * sizeof(half));
    
    // Per-layer weights
    for (int layer = 0; layer < config.block_count; ++layer) {
        auto& l = weights->layers[layer];
        
        // Attention
        cudaMalloc(&l.attn_norm_weight, config.hidden_dim * sizeof(half));
        cudaMalloc(&l.attn_q_weight, config.hidden_dim * config.hidden_dim * sizeof(half));
        cudaMalloc(&l.attn_k_weight, config.kv_dim * config.hidden_dim * sizeof(half));
        cudaMalloc(&l.attn_v_weight, config.kv_dim * config.hidden_dim * sizeof(half));
        cudaMalloc(&l.attn_output_weight, config.hidden_dim * config.hidden_dim * sizeof(half));
        
        // FFN
        cudaMalloc(&l.ffn_norm_weight, config.hidden_dim * sizeof(half));
        cudaMalloc(&l.ffn_gate_weight, config.ffn_dim * config.hidden_dim * sizeof(half));
        cudaMalloc(&l.ffn_up_weight, config.ffn_dim * config.hidden_dim * sizeof(half));
        cudaMalloc(&l.ffn_down_weight, config.hidden_dim * config.ffn_dim * sizeof(half));
    }
    
    // Output
    cudaMalloc(&weights->output_norm_weight, config.hidden_dim * sizeof(half));
    cudaMalloc(&weights->output_weight, config.vocab_size * config.hidden_dim * sizeof(half));
}
```

**VRAM Usage Calculation**:
```cpp
size_t calculate_vram_usage(const QwenWeights& weights, const LlamaConfig& config) {
    size_t total = 0;
    
    // Embedding: vocab_size × hidden_dim
    total += config.vocab_size * config.hidden_dim * sizeof(half);
    
    // Per-layer (24 layers)
    for (int layer = 0; layer < config.block_count; ++layer) {
        // Attention: norm + Q + K + V + output
        total += config.hidden_dim * sizeof(half);  // norm
        total += config.hidden_dim * config.hidden_dim * sizeof(half);  // Q
        total += config.kv_dim * config.hidden_dim * sizeof(half);  // K
        total += config.kv_dim * config.hidden_dim * sizeof(half);  // V
        total += config.hidden_dim * config.hidden_dim * sizeof(half);  // output
        
        // FFN: norm + gate + up + down
        total += config.hidden_dim * sizeof(half);  // norm
        total += config.ffn_dim * config.hidden_dim * sizeof(half);  // gate
        total += config.ffn_dim * config.hidden_dim * sizeof(half);  // up
        total += config.hidden_dim * config.ffn_dim * sizeof(half);  // down
    }
    
    // Output: norm + weight
    total += config.hidden_dim * sizeof(half);  // norm
    total += config.vocab_size * config.hidden_dim * sizeof(half);  // weight
    
    return total;
}
```

---

## Testing Strategy

### Unit Tests
- Test embedding loading
- Test single layer loading
- Test output loading
- Test VRAM allocation
- Test VRAM usage calculation
- Test error handling (allocation failure)

### Integration Tests
- Test full model loading (all 24 layers)
- Test progress events emitted
- Test VRAM residency validation
- Test with real Qwen2.5-0.5B GGUF file

### Performance Tests
- Measure loading time (should be <10 seconds)
- Measure transfer throughput (GB/s)
- Verify VRAM usage matches calculation

### Manual Verification
1. Load Qwen2.5-0.5B model
2. Verify all weights loaded to VRAM
3. Verify VRAM usage ~900MB (for 0.5B model)
4. Check logs show loading progress
5. Verify no CUDA errors

---

## Definition of Done

- [ ] All acceptance criteria met
- [ ] Code reviewed
- [ ] Unit tests passing (6+ tests)
- [ ] Integration tests passing
- [ ] Performance benchmarks recorded
- [ ] Documentation updated
- [ ] Story marked complete in day-tracker.md

---

## References

- Spec: `bin/.specs/01_M0_worker_orcd.md` Section 6.6 (Model Loading)
- Related Stories: LT-022, LT-004, LT-003, LT-024

---

**Status**: Ready for execution  
**Owner**: Llama-Beta  
**Created**: 2025-10-04

---

Detailed by Project Management Team — ready to implement 📋

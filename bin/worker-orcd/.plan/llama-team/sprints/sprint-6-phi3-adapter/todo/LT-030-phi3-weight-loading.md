# LT-030: Phi-3 Weight Loading

**Team**: Llama-Beta  
**Sprint**: Sprint 6 - Phi-3 + Adapter  
**Size**: M (2 days)  
**Days**: 69-70  
**Spec Ref**: M0-W-1230

---

## Story Description

Load Phi-3-mini-4k model weights from GGUF file to GPU VRAM. Adapt Qwen weight loading pipeline for Phi-3's larger model size (32 layers, 3072 hidden dim) and MHA attention configuration.

---

## Acceptance Criteria

- [ ] Map Phi-3 weight tensors (32 layers)
- [ ] Load token embedding weights to VRAM
- [ ] Load all 32 layer weights to VRAM (attention + FFN)
- [ ] Load output weights to VRAM
- [ ] Handle larger model size (~3.8B parameters)
- [ ] Use chunked H2D transfer for large weights
- [ ] Validate VRAM residency after loading
- [ ] Calculate and log total VRAM usage (~7-8GB)
- [ ] Unit tests validate Phi-3 weight loading
- [ ] Integration tests validate full model loading
- [ ] Error handling for VRAM allocation failures
- [ ] Log loading progress and completion

---

## Dependencies

### Upstream (Blocks This Story)
- LT-029: Phi-3 Metadata Analysis (needs config)
- LT-023: Qwen Weight Loading (needs loading pipeline)

### Downstream (This Story Blocks)
- LT-031: Phi-3 Forward Pass (needs loaded weights)
- LT-032: Tokenizer Conformance Tests Phi-3 (needs loaded model)

---

## Technical Details

### Files to Create/Modify
- `bin/worker-orcd/cuda/src/models/phi3_weights.cpp` - Phi-3 weight mapping
- `bin/worker-orcd/cuda/src/models/phi3_weights.h` - Phi-3 weight structures
- `bin/worker-orcd/cuda/src/models/phi3_loader.cpp` - Phi-3 loader
- `bin/worker-orcd/src/models/phi3_loader.rs` - Rust loader wrapper

### Phi-3 Weight Structure
```cpp
struct Phi3Weights {
    // Embedding
    half* token_embedding;  // [vocab_size, hidden_dim]
    
    // Per-layer weights (32 layers)
    struct Layer {
        // Attention (MHA: 32 Q heads, 32 KV heads)
        half* attn_norm_weight;     // [hidden_dim]
        half* attn_q_weight;        // [hidden_dim, hidden_dim]
        half* attn_k_weight;        // [hidden_dim, hidden_dim]  // Note: same as Q for MHA
        half* attn_v_weight;        // [hidden_dim, hidden_dim]  // Note: same as Q for MHA
        half* attn_output_weight;   // [hidden_dim, hidden_dim]
        
        // FFN
        half* ffn_norm_weight;      // [hidden_dim]
        half* ffn_gate_weight;      // [ffn_dim, hidden_dim]
        half* ffn_up_weight;        // [ffn_dim, hidden_dim]
        half* ffn_down_weight;      // [hidden_dim, ffn_dim]
    } layers[32];  // 32 layers (vs 24 for Qwen)
    
    // Output
    half* output_norm_weight;   // [hidden_dim]
    half* output_weight;        // [vocab_size, hidden_dim]
};
```

### Weight Dimensions

**Phi-3-mini-4k Dimensions**:
- Embedding: [32000, 3072]
- Attention Q/K/V: [3072, 3072] each (MHA)
- Attention output: [3072, 3072]
- FFN gate/up: [8192, 3072] each
- FFN down: [3072, 8192]
- Output: [32000, 3072]

**VRAM Calculation**:
```cpp
size_t calculate_phi3_vram() {
    size_t total = 0;
    
    // Embedding: 32000 × 3072 × 2 bytes = 196 MB
    total += 32000 * 3072 * 2;
    
    // Per-layer (32 layers):
    for (int layer = 0; layer < 32; ++layer) {
        // Attention: norm + Q + K + V + output
        total += 3072 * 2;                    // norm: 6 KB
        total += 3072 * 3072 * 2;            // Q: 18 MB
        total += 3072 * 3072 * 2;            // K: 18 MB
        total += 3072 * 3072 * 2;            // V: 18 MB
        total += 3072 * 3072 * 2;            // output: 18 MB
        
        // FFN: norm + gate + up + down
        total += 3072 * 2;                    // norm: 6 KB
        total += 8192 * 3072 * 2;            // gate: 48 MB
        total += 8192 * 3072 * 2;            // up: 48 MB
        total += 3072 * 8192 * 2;            // down: 48 MB
    }
    
    // Output: norm + weight
    total += 3072 * 2;                        // norm: 6 KB
    total += 32000 * 3072 * 2;               // weight: 196 MB
    
    // Total: ~7.5 GB
    return total;
}
```

### Implementation Notes

**Weight Loading**:
```cpp
Result<Phi3Model> load_phi3(const std::string& gguf_path) {
    Phi3Model model;
    
    // 1. Parse GGUF and extract config
    auto mmap = MmapFile::open(gguf_path);
    auto header = parse_gguf_header(mmap);
    model.config = parse_phi3_metadata(header);
    
    tracing::info!("Loading Phi-3-mini-4k model from {}", gguf_path);
    tracing::info!("  Layers: {}", model.config.block_count);
    tracing::info!("  Hidden dim: {}", model.config.embedding_length);
    tracing::info!("  Attention: MHA ({} heads)", model.config.attention_head_count);
    
    // 2. Allocate VRAM
    allocate_phi3_vram(&model.weights, model.config);
    
    // 3. Load embedding
    load_embedding(model.weights.token_embedding, mmap, "token_embd.weight");
    emit_progress("Embedding loaded", 1, 34);
    
    // 4. Load 32 transformer layers
    for (int layer = 0; layer < 32; ++layer) {
        load_phi3_layer(model.weights.layers[layer], layer, mmap);
        emit_progress(fmt::format("Layer {} loaded", layer), layer + 2, 34);
    }
    
    // 5. Load output weights
    load_output(model.weights, mmap);
    emit_progress("Output weights loaded", 34, 34);
    
    // 6. Validate VRAM residency
    validate_vram_residency(model.weights);
    
    // 7. Calculate total VRAM usage
    model.total_vram_bytes = calculate_phi3_vram();
    
    tracing::info!("Phi-3 model loaded: {} GB VRAM", model.total_vram_bytes / (1024.0 * 1024.0 * 1024.0));
    
    return Ok(model);
}
```

**Layer Loading**:
```cpp
void load_phi3_layer(Phi3Weights::Layer& layer, int layer_idx, const MmapFile& mmap) {
    std::string prefix = fmt::format("blk.{}", layer_idx);
    
    // Attention weights
    load_weight(layer.attn_norm_weight, mmap, prefix + ".attn_norm.weight");
    load_weight(layer.attn_q_weight, mmap, prefix + ".attn_q.weight");
    load_weight(layer.attn_k_weight, mmap, prefix + ".attn_k.weight");
    load_weight(layer.attn_v_weight, mmap, prefix + ".attn_v.weight");
    load_weight(layer.attn_output_weight, mmap, prefix + ".attn_output.weight");
    
    // FFN weights
    load_weight(layer.ffn_norm_weight, mmap, prefix + ".ffn_norm.weight");
    load_weight(layer.ffn_gate_weight, mmap, prefix + ".ffn_gate.weight");
    load_weight(layer.ffn_up_weight, mmap, prefix + ".ffn_up.weight");
    load_weight(layer.ffn_down_weight, mmap, prefix + ".ffn_down.weight");
}

void load_weight(half* dst, const MmapFile& mmap, const std::string& name) {
    const half* src = mmap.get_tensor_data(name);
    size_t size = mmap.get_tensor_size(name);
    
    // Chunked transfer (256MB chunks)
    TransferConfig config = {256 * 1024 * 1024, true};
    ChunkedTransfer::h2d_chunked(dst, src, size, config);
}
```

**MHA Compatibility**:
```cpp
// Phi-3 uses MHA (num_q_heads == num_kv_heads == 32)
// Our GQA kernel supports this case automatically
GQAAttentionConfig phi3_attn_config = {
    1,      // batch_size
    seq_len,
    32,     // num_q_heads
    32,     // num_kv_heads (same as Q for MHA)
    96,     // head_dim
    1.0f / sqrtf(96)  // scale
};

// GQA kernel will work correctly:
// - Each Q head uses corresponding KV head (1:1 mapping)
// - No head grouping needed
```

---

## Testing Strategy

### Unit Tests
- Test Phi-3 weight mapping (32 layers)
- Test embedding loading
- Test single layer loading
- Test output loading
- Test VRAM allocation (7-8GB)
- Test VRAM usage calculation

### Integration Tests
- Test full Phi-3 model loading
- Test with real Phi-3 GGUF file
- Test progress events emitted
- Test VRAM residency validation

### Performance Tests
- Measure loading time (should be <30 seconds)
- Measure transfer throughput (GB/s)
- Verify VRAM usage ~7.5GB

### Manual Verification
1. Load Phi-3-mini-4k model
2. Verify all 32 layers loaded
3. Verify VRAM usage ~7.5GB
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
- Phi-3 Model Card: https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf
- Related Stories: LT-029, LT-023, LT-031

---

**Status**: Ready for execution  
**Owner**: Llama-Beta  
**Created**: 2025-10-04

---

Detailed by Project Management Team — ready to implement 📋

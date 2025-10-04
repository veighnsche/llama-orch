# LT-022: Qwen Weight Mapping

**Team**: Llama-Beta  
**Sprint**: Sprint 5 - Qwen Integration  
**Size**: M (3 days)  
**Days**: 55-57  
**Spec Ref**: M0-W-1230

---

## Story Description

Map Qwen2.5-0.5B GGUF tensor names to Llama model architecture components. Create weight mapping table to correctly load model weights from GGUF file into appropriate kernel parameters for inference.

---

## Acceptance Criteria

- [ ] Map all Qwen2.5-0.5B GGUF tensors to model components
- [ ] Map embedding weights (token_embd.weight)
- [ ] Map attention weights (Q, K, V, output projections) for all 24 layers
- [ ] Map FFN weights (gate, up, down projections) for all 24 layers
- [ ] Map normalization weights (attention_norm, ffn_norm) for all 24 layers
- [ ] Map output weights (output_norm, output.weight)
- [ ] Handle weight name variations (qwen vs llama naming)
- [ ] Create weight mapping table (GGUF name → component → dimensions)
- [ ] Validate all tensors are mapped (no missing weights)
- [ ] Unit tests validate mapping correctness
- [ ] Document weight mapping in markdown table
- [ ] Error handling for missing or misnamed tensors
- [ ] Log weight mapping at INFO level

---

## Dependencies

### Upstream (Blocks This Story)
- LT-002: GGUF Metadata Extraction (needs config)
- LT-020: Gate 1 Participation (needs validated kernels)

### Downstream (This Story Blocks)
- LT-023: Qwen Weight Loading (needs weight mapping)
- LT-024: Qwen Forward Pass (needs mapped weights)

---

## Technical Details

### Files to Create/Modify
- `bin/worker-orcd/cuda/src/models/qwen_weights.cpp` - Weight mapping
- `bin/worker-orcd/cuda/src/models/qwen_weights.h` - Weight structures
- `bin/worker-orcd/src/models/qwen_weights.rs` - Rust weight mapping
- `bin/worker-orcd/.docs/qwen_weight_mapping.md` - Weight mapping documentation

### Weight Mapping Structure
```cpp
struct QwenWeights {
    // Embedding
    half* token_embedding;  // [vocab_size, hidden_dim]
    
    // Per-layer weights (24 layers)
    struct Layer {
        // Attention
        half* attn_norm_weight;     // [hidden_dim]
        half* attn_q_weight;        // [hidden_dim, hidden_dim]
        half* attn_k_weight;        // [kv_dim, hidden_dim]
        half* attn_v_weight;        // [kv_dim, hidden_dim]
        half* attn_output_weight;   // [hidden_dim, hidden_dim]
        
        // FFN
        half* ffn_norm_weight;      // [hidden_dim]
        half* ffn_gate_weight;      // [ffn_dim, hidden_dim]
        half* ffn_up_weight;        // [ffn_dim, hidden_dim]
        half* ffn_down_weight;      // [hidden_dim, ffn_dim]
    } layers[24];
    
    // Output
    half* output_norm_weight;   // [hidden_dim]
    half* output_weight;        // [vocab_size, hidden_dim]
};

// Weight mapping table
struct WeightMapping {
    std::string gguf_name;      // Name in GGUF file
    std::string component;      // Model component
    std::vector<int> dims;      // Expected dimensions
    half** target_ptr;          // Pointer to weight location
};
```

### GGUF Tensor Name Patterns

**Qwen2.5-0.5B GGUF Names**:
```
token_embd.weight                           → token_embedding
blk.{layer}.attn_norm.weight               → layers[layer].attn_norm_weight
blk.{layer}.attn_q.weight                  → layers[layer].attn_q_weight
blk.{layer}.attn_k.weight                  → layers[layer].attn_k_weight
blk.{layer}.attn_v.weight                  → layers[layer].attn_v_weight
blk.{layer}.attn_output.weight             → layers[layer].attn_output_weight
blk.{layer}.ffn_norm.weight                → layers[layer].ffn_norm_weight
blk.{layer}.ffn_gate.weight                → layers[layer].ffn_gate_weight
blk.{layer}.ffn_up.weight                  → layers[layer].ffn_up_weight
blk.{layer}.ffn_down.weight                → layers[layer].ffn_down_weight
output_norm.weight                          → output_norm_weight
output.weight                               → output_weight
```

### Implementation Notes

**Weight Mapping Algorithm**:
```cpp
QwenWeights map_qwen_weights(const GGUFHeader& header) {
    QwenWeights weights;
    
    // Map embedding
    weights.token_embedding = find_tensor(header, "token_embd.weight");
    
    // Map per-layer weights
    for (int layer = 0; layer < 24; ++layer) {
        auto& l = weights.layers[layer];
        
        // Attention
        l.attn_norm_weight = find_tensor(header, fmt::format("blk.{}.attn_norm.weight", layer));
        l.attn_q_weight = find_tensor(header, fmt::format("blk.{}.attn_q.weight", layer));
        l.attn_k_weight = find_tensor(header, fmt::format("blk.{}.attn_k.weight", layer));
        l.attn_v_weight = find_tensor(header, fmt::format("blk.{}.attn_v.weight", layer));
        l.attn_output_weight = find_tensor(header, fmt::format("blk.{}.attn_output.weight", layer));
        
        // FFN
        l.ffn_norm_weight = find_tensor(header, fmt::format("blk.{}.ffn_norm.weight", layer));
        l.ffn_gate_weight = find_tensor(header, fmt::format("blk.{}.ffn_gate.weight", layer));
        l.ffn_up_weight = find_tensor(header, fmt::format("blk.{}.ffn_up.weight", layer));
        l.ffn_down_weight = find_tensor(header, fmt::format("blk.{}.ffn_down.weight", layer));
    }
    
    // Map output
    weights.output_norm_weight = find_tensor(header, "output_norm.weight");
    weights.output_weight = find_tensor(header, "output.weight");
    
    return weights;
}

half* find_tensor(const GGUFHeader& header, const std::string& name) {
    for (const auto& tensor : header.tensors) {
        if (tensor.name == name) {
            return tensor.data_ptr;  // Pointer to mmap'd data
        }
    }
    throw std::runtime_error("Tensor not found: " + name);
}
```

**Dimension Validation**:
```cpp
void validate_weight_dimensions(const QwenWeights& weights, const LlamaConfig& config) {
    // Embedding: [vocab_size, hidden_dim]
    assert(get_tensor_dims(weights.token_embedding) == std::vector<int>{config.vocab_size, config.hidden_dim});
    
    // Per-layer validation
    for (int layer = 0; layer < config.block_count; ++layer) {
        auto& l = weights.layers[layer];
        
        // Attention weights
        assert(get_tensor_dims(l.attn_q_weight) == std::vector<int>{config.hidden_dim, config.hidden_dim});
        assert(get_tensor_dims(l.attn_k_weight) == std::vector<int>{config.kv_dim, config.hidden_dim});
        // ... validate all weights
    }
}
```

---

## Testing Strategy

### Unit Tests
- Test weight mapping for all 24 layers
- Test embedding weight mapping
- Test output weight mapping
- Test dimension validation
- Test error handling for missing tensors
- Test error handling for mismatched dimensions

### Integration Tests
- Test mapping with real Qwen2.5-0.5B GGUF file
- Test all tensors are found and mapped
- Test weight structure construction

### Manual Verification
1. Load Qwen2.5-0.5B GGUF file
2. Map all weights
3. Verify all 24 layers mapped correctly
4. Verify dimensions match config (896, 4864, etc.)
5. Check logs show weight mapping summary

---

## Definition of Done

- [ ] All acceptance criteria met
- [ ] Code reviewed
- [ ] Unit tests passing (6+ tests)
- [ ] Integration tests passing
- [ ] Weight mapping documentation complete
- [ ] Documentation updated
- [ ] Story marked complete in day-tracker.md

---

## References

- Spec: `bin/.specs/01_M0_worker_orcd.md` Section 6.6 (Model Loading)
- Qwen2.5 Architecture: https://qwenlm.github.io/blog/qwen2.5/
- GGUF Spec: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md
- Related Stories: LT-002, LT-023, LT-024

---

**Status**: Ready for execution  
**Owner**: Llama-Beta  
**Created**: 2025-10-04

---

Detailed by Project Management Team — ready to implement 📋

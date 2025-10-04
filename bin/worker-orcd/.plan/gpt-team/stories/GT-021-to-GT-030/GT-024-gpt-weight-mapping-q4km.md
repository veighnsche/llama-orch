# GT-024: GPT Weight Mapping (Q4_K_M)

**Team**: GPT-Gamma  
**Sprint**: Sprint 4 (GPT Basic)  
**Size**: L (3 days)  
**Days**: 58-60  
**Spec Ref**: M0-W-1211, M0-W-1220

---

## Story Description

Implement weight tensor mapping for GPT architecture in Q4_K_M quantization format. Map GGUF tensor names to GPT model structure (embeddings, attention, FFN, layer norms).

---

## Acceptance Criteria

- [ ] Map token embedding weights from GGUF
- [ ] Map position embedding weights from GGUF
- [ ] Map attention Q/K/V/O weights for all layers
- [ ] Map FFN up/down projection weights for all layers
- [ ] Map LayerNorm gamma/beta for all layers
- [ ] Map LM head weights
- [ ] Validate all required tensors present
- [ ] Unit tests validate weight mapping correctness
- [ ] Documentation updated with tensor naming

---

## Dependencies

### Upstream (Blocks This Story)
- GT-023: FFI Integration Tests GPT (needs FFI validated)
- GT-006: GGUF v3 Tensor Support (needs tensor parser)

### Downstream (This Story Blocks)
- GT-025: GPT Weight Loading (needs weight mapping)

---

## Technical Details

### Files to Create/Modify
- `bin/worker-orcd/cuda/src/model/gpt_weights.cpp` - Weight mapping
- `bin/worker-orcd/cuda/src/model/gpt_weights.h` - Weight structures

### GPT Weight Structure
```cpp
struct GPTWeights {
    // Embeddings
    DeviceMemory token_embeddings;    // [vocab_size, d_model]
    DeviceMemory position_embeddings; // [max_pos, d_model]
    
    // Per-layer weights
    struct Layer {
        // Attention
        DeviceMemory attn_q_weight;   // [d_model, d_model]
        DeviceMemory attn_k_weight;   // [d_model, d_model]
        DeviceMemory attn_v_weight;   // [d_model, d_model]
        DeviceMemory attn_o_weight;   // [d_model, d_model]
        
        // LayerNorm 1 (pre-attention)
        DeviceMemory ln1_gamma;       // [d_model]
        DeviceMemory ln1_beta;        // [d_model]
        
        // FFN
        DeviceMemory ffn_up_weight;   // [d_model, ffn_dim]
        DeviceMemory ffn_down_weight; // [ffn_dim, d_model]
        
        // LayerNorm 2 (pre-FFN)
        DeviceMemory ln2_gamma;       // [d_model]
        DeviceMemory ln2_beta;        // [d_model]
    };
    std::vector<Layer> layers;
    
    // LM Head
    DeviceMemory lm_head_weight;      // [d_model, vocab_size]
};
```

---

## Testing Strategy

### Unit Tests
- Test tensor name parsing
- Test weight structure construction
- Test dimension validation

---

## Definition of Done

- [ ] All acceptance criteria met
- [ ] Tests passing
- [ ] Documentation updated

---

## References

- Spec: `bin/.specs/01_M0_worker_orcd.md` Section 6.3

---

**Status**: Ready for execution  
**Owner**: GPT-Gamma  
**Created**: 2025-10-04

---
Detailed by Project Management Team — ready to implement 📋

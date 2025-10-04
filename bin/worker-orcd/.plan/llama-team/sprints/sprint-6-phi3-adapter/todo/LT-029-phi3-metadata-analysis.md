# LT-029: Phi-3 Metadata Analysis

**Team**: Llama-Beta  
**Sprint**: Sprint 6 - Phi-3 + Adapter  
**Size**: S (1 day)  
**Days**: 68  
**Spec Ref**: M0-W-1230

---

## Story Description

Analyze Phi-3-mini-4k GGUF metadata to understand model architecture differences from Qwen. Identify configuration parameters, weight dimensions, and any architectural variations to enable Phi-3 model loading.

---

## Acceptance Criteria

- [ ] Parse Phi-3-mini-4k GGUF metadata
- [ ] Extract model configuration (layers, dimensions, heads)
- [ ] Compare Phi-3 config with Qwen config
- [ ] Identify architectural differences (MHA vs GQA)
- [ ] Document weight tensor names and dimensions
- [ ] Create Phi-3 configuration struct
- [ ] Validate Phi-3 is Llama-compatible
- [ ] Unit tests validate metadata parsing
- [ ] Document findings in markdown report
- [ ] Error handling for unsupported architectures
- [ ] Log metadata analysis results

---

## Dependencies

### Upstream (Blocks This Story)
- LT-027: Gate 2 Checkpoint (needs validated pipeline)
- LT-002: GGUF Metadata Extraction (needs metadata parser)

### Downstream (This Story Blocks)
- LT-030: Phi-3 Weight Loading (needs config)
- LT-031: Phi-3 Forward Pass (needs config)

---

## Technical Details

### Files to Create/Modify
- `bin/worker-orcd/tests/analysis/phi3_metadata_analysis.cpp` - Metadata analysis
- `bin/worker-orcd/.docs/phi3_architecture_analysis.md` - Analysis report
- `bin/worker-orcd/src/models/phi3_config.rs` - Phi-3 config struct

### Phi-3 Configuration
```cpp
struct Phi3Config {
    std::string architecture;      // "llama" (Phi-3 uses Llama architecture)
    uint32_t context_length;       // 4096
    uint32_t embedding_length;     // 3072 (d_model)
    uint32_t block_count;          // 32 layers
    uint32_t attention_head_count; // 32 heads
    uint32_t attention_head_count_kv; // 32 heads (MHA, not GQA!)
    uint32_t ffn_length;           // 8192 (4 * d_model)
    uint32_t rope_dimension_count; // 96
    float rope_freq_base;          // 10000.0
    uint32_t vocab_size;           // ~32000
    
    // Derived
    uint32_t head_dim;             // 96 (3072 / 32)
    bool is_mha;                   // true (MHA, not GQA)
};
```

### Metadata Analysis
```cpp
void analyze_phi3_metadata() {
    // 1. Load Phi-3 GGUF file
    auto mmap = MmapFile::open("phi-3-mini-4k-instruct.gguf");
    auto header = parse_gguf_header(mmap);
    auto metadata = parse_gguf_metadata(mmap);
    
    // 2. Extract configuration
    Phi3Config config;
    config.architecture = metadata.get_string("general.architecture");
    config.context_length = metadata.get_uint32("llama.context_length");
    config.embedding_length = metadata.get_uint32("llama.embedding_length");
    config.block_count = metadata.get_uint32("llama.block_count");
    config.attention_head_count = metadata.get_uint32("llama.attention.head_count");
    config.attention_head_count_kv = metadata.get_uint32("llama.attention.head_count_kv");
    config.ffn_length = metadata.get_uint32("llama.feed_forward_length");
    config.rope_dimension_count = metadata.get_uint32("llama.rope.dimension_count");
    config.rope_freq_base = metadata.get_float("llama.rope.freq_base");
    
    // 3. Calculate derived parameters
    config.head_dim = config.embedding_length / config.attention_head_count;
    config.is_mha = (config.attention_head_count == config.attention_head_count_kv);
    
    // 4. Log configuration
    tracing::info!("Phi-3 Configuration:");
    tracing::info!("  Architecture: {}", config.architecture);
    tracing::info!("  Context length: {}", config.context_length);
    tracing::info!("  Embedding dim: {}", config.embedding_length);
    tracing::info!("  Layers: {}", config.block_count);
    tracing::info!("  Attention heads: {}", config.attention_head_count);
    tracing::info!("  KV heads: {}", config.attention_head_count_kv);
    tracing::info!("  Head dim: {}", config.head_dim);
    tracing::info!("  FFN dim: {}", config.ffn_length);
    tracing::info!("  Attention type: {}", config.is_mha ? "MHA" : "GQA");
    
    // 5. Compare with Qwen
    QwenConfig qwen_config = {/* ... */};
    
    tracing::info!("Comparison with Qwen:");
    tracing::info!("  Context: Phi-3={}, Qwen={}", config.context_length, qwen_config.context_length);
    tracing::info!("  Layers: Phi-3={}, Qwen={}", config.block_count, qwen_config.block_count);
    tracing::info!("  Attention: Phi-3=MHA, Qwen=GQA");
    tracing::info!("  Hidden dim: Phi-3={}, Qwen={}", config.embedding_length, qwen_config.embedding_length);
}
```

### Architecture Comparison

**Qwen2.5-0.5B**:
- Context: 32768
- Layers: 24
- Hidden dim: 896
- Attention: GQA (14 Q heads, 2 KV heads)
- Head dim: 64
- FFN dim: 4864

**Phi-3-mini-4k**:
- Context: 4096
- Layers: 32
- Hidden dim: 3072
- Attention: MHA (32 Q heads, 32 KV heads)
- Head dim: 96
- FFN dim: 8192

**Key Differences**:
1. **Attention**: Phi-3 uses MHA (all heads are KV heads), Qwen uses GQA
2. **Context**: Phi-3 has shorter context (4K vs 32K)
3. **Size**: Phi-3 is larger (3.8B params vs 0.5B)
4. **Layers**: Phi-3 has more layers (32 vs 24)

**Compatibility**:
- ✅ Both use Llama architecture
- ✅ Both use RoPE, RMSNorm, SwiGLU
- ✅ GQA kernel supports MHA (when num_q_heads == num_kv_heads)
- ✅ Same weight tensor naming convention

### Weight Tensor Analysis
```cpp
void analyze_phi3_weights() {
    // Expected tensor names (same as Qwen)
    std::vector<std::string> expected_tensors = {
        "token_embd.weight",
        "blk.0.attn_norm.weight",
        "blk.0.attn_q.weight",
        "blk.0.attn_k.weight",
        "blk.0.attn_v.weight",
        "blk.0.attn_output.weight",
        "blk.0.ffn_norm.weight",
        "blk.0.ffn_gate.weight",
        "blk.0.ffn_up.weight",
        "blk.0.ffn_down.weight",
        // ... repeat for 32 layers
        "output_norm.weight",
        "output.weight",
    };
    
    // Verify all tensors exist
    for (const auto& name : expected_tensors) {
        auto tensor = find_tensor(header, name);
        tracing::info!("  {}: dims={}", name, tensor.dimensions);
    }
}
```

---

## Testing Strategy

### Unit Tests
- Test Phi-3 metadata parsing
- Test configuration extraction
- Test derived parameter calculation
- Test MHA detection (num_q_heads == num_kv_heads)

### Analysis Tests
- Test weight tensor enumeration
- Test dimension validation
- Test compatibility check with Llama pipeline

### Manual Verification
1. Load Phi-3 GGUF file
2. Parse metadata
3. Verify configuration matches expected values
4. Review architecture comparison
5. Check logs show complete analysis

---

## Definition of Done

- [ ] All acceptance criteria met
- [ ] Code reviewed
- [ ] Unit tests passing (4+ tests)
- [ ] Analysis report complete
- [ ] Phi-3 config struct created
- [ ] Documentation updated
- [ ] Story marked complete in day-tracker.md

---

## References

- Spec: `bin/.specs/01_M0_worker_orcd.md` Section 6.2 (Model Validation)
- Phi-3 Paper: https://arxiv.org/abs/2404.14219
- Phi-3 Model Card: https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf
- Related Stories: LT-002, LT-027, LT-030

---

**Status**: Ready for execution  
**Owner**: Llama-Beta  
**Created**: 2025-10-04

---

Detailed by Project Management Team — ready to implement 📋

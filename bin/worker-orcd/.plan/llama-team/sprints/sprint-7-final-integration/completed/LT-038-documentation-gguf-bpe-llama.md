# LT-038: Documentation (GGUF, BPE, Llama)

**Team**: Llama-Beta  
**Sprint**: Sprint 7 - Final Integration  
**Size**: M (2 days)  
**Days**: 85-86  
**Spec Ref**: (Docs)

---

## Story Description

Create comprehensive documentation for all Llama team deliverables including GGUF parsing, BPE tokenization, Llama kernels, model integration, and adapter usage. Provide guides, API documentation, and examples for future developers.

---

## Acceptance Criteria

- [ ] Document GGUF format parsing and loading
- [ ] Document BPE tokenization (encoding/decoding)
- [ ] Document Llama kernels (RoPE, RMSNorm, GQA, SwiGLU)
- [ ] Document Qwen model integration
- [ ] Document Phi-3 model integration
- [ ] Document LlamaInferenceAdapter usage
- [ ] Create API documentation for all public interfaces
- [ ] Create usage examples and tutorials
- [ ] Create troubleshooting guide
- [ ] Document performance characteristics
- [ ] Document security considerations (GGUF bounds validation)
- [ ] All documentation reviewed and approved

---

## Dependencies

### Upstream (Blocks This Story)
- LT-037: VRAM Pressure Tests (needs test results)
- LT-035: Llama Integration Test Suite (needs test documentation)

### Downstream (This Story Blocks)
- None (final story)

---

## Technical Details

### Files to Create/Modify
- `bin/worker-orcd/.docs/llama/01_gguf_format.md` - GGUF format guide
- `bin/worker-orcd/.docs/llama/02_bpe_tokenization.md` - BPE tokenization guide
- `bin/worker-orcd/.docs/llama/03_llama_kernels.md` - Kernel documentation
- `bin/worker-orcd/.docs/llama/04_qwen_integration.md` - Qwen guide
- `bin/worker-orcd/.docs/llama/05_phi3_integration.md` - Phi-3 guide
- `bin/worker-orcd/.docs/llama/06_adapter_usage.md` - Adapter guide
- `bin/worker-orcd/.docs/llama/07_api_reference.md` - API documentation
- `bin/worker-orcd/.docs/llama/08_examples.md` - Usage examples
- `bin/worker-orcd/.docs/llama/09_troubleshooting.md` - Troubleshooting guide
- `bin/worker-orcd/.docs/llama/10_performance.md` - Performance guide
- `bin/worker-orcd/.docs/llama/README.md` - Llama documentation index

### Documentation Structure

**1. GGUF Format Guide** (`01_gguf_format.md`):
```markdown
# GGUF Format Parsing

## Overview
GGUF (GGML Universal File Format) is a binary format for storing LLM weights and metadata.

## File Structure
- Magic bytes: 0x47475546 ("GGUF")
- Version: 3
- Tensor count: Number of weight tensors
- Metadata: Key-value pairs (model config)
- Tensor data: Raw weight data

## Parsing Process
1. Parse header (magic, version, counts)
2. Parse metadata (model configuration)
3. Map tensor names to model components
4. Memory-map file for zero-copy access
5. Transfer weights to VRAM

## Security Considerations
⚠️ **Critical**: All tensor offsets and sizes MUST be validated to prevent heap overflow (CWE-119/787).

See `SECURITY_ALERT_GGUF_PARSING.md` for details.

## Example
```cpp
auto mmap = MmapFile::open("model.gguf");
auto header = parse_gguf_header(mmap);
auto metadata = parse_gguf_metadata(mmap);
auto weights = map_weights(header, metadata);
```

## References
- GGUF Spec: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md
- Story: LT-001 (GGUF Header Parser)
```

**2. BPE Tokenization Guide** (`02_bpe_tokenization.md`):
```markdown
# Byte-Level BPE Tokenization

## Overview
Byte-level Byte Pair Encoding (BPE) tokenizer for Llama models.

## Components
- **Encoder**: Text → Token IDs
- **Decoder**: Token IDs → Text
- **Streaming Decoder**: UTF-8 safe streaming decode

## Encoding Process
1. Convert text to byte-level representation
2. Apply BPE merges iteratively
3. Convert tokens to IDs using vocabulary

## Decoding Process
1. Convert IDs to tokens using vocabulary
2. Concatenate byte-level tokens
3. Convert bytes to UTF-8 text

## UTF-8 Safety
The streaming decoder buffers incomplete UTF-8 sequences at token boundaries to prevent broken characters in streaming output.

## Example
```rust
let encoder = BPEEncoder::from_gguf("model.gguf")?;
let decoder = BPEDecoder::from_gguf("model.gguf")?;

// Encode
let ids = encoder.encode("Hello, world!");

// Decode
let text = decoder.decode(&ids)?;

// Streaming decode
let mut streaming = StreamingDecoder::new(decoder);
for id in ids {
    let partial = streaming.decode_token(id);
    print!("{}", partial);
}
let remaining = streaming.flush();
print!("{}", remaining);
```

## References
- BPE Paper: https://arxiv.org/abs/1508.07909
- Stories: LT-007 to LT-011
```

**3. Llama Kernels Guide** (`03_llama_kernels.md`):
```markdown
# Llama CUDA Kernels

## Overview
CUDA kernels for Llama transformer architecture.

## Kernels

### RoPE (Rotary Position Embedding)
Applies rotary embeddings to Q and K tensors.

**Configuration**:
- Frequency base: 10000.0 (standard) or 1000000.0 (extended context)
- Dimensions: head_dim (64 or 96)

**Usage**:
```cpp
RoPEConfig config = {seq_len, num_heads, head_dim, freq_base, rope_dim};
rope_forward(q_out, k_out, q_in, k_in, config);
```

### RMSNorm (Root Mean Square Normalization)
Normalizes activations using RMS.

**Usage**:
```cpp
RMSNormConfig config = {batch_size, seq_len, hidden_dim, eps};
rmsnorm_forward(output, input, weight, config);
```

### GQA Attention (Grouped Query Attention)
Attention with grouped KV heads for efficiency.

**Supports**:
- GQA: num_q_heads > num_kv_heads (e.g., Qwen: 14 Q, 2 KV)
- MHA: num_q_heads == num_kv_heads (e.g., Phi-3: 32 Q, 32 KV)

**Usage**:
```cpp
GQAAttentionConfig config = {batch_size, seq_len, num_q_heads, num_kv_heads, head_dim, scale};
gqa_attention_prefill(output, q, k, v, kv_cache_k, kv_cache_v, config);
```

### SwiGLU FFN (Swish-Gated Linear Unit)
Feed-forward network with SwiGLU activation.

**Usage**:
```cpp
SwiGLUConfig config = {batch_size, seq_len, hidden_dim, ffn_dim};
swiglu_ffn_forward(output, input, w_gate, w_up, w_down, config);
```

## Performance
- RoPE: ~0.1ms (10 tokens)
- RMSNorm: ~0.05ms (10 tokens)
- GQA Attention: ~2ms (10 tokens, prefill)
- SwiGLU FFN: ~5ms (10 tokens)

## References
- Stories: LT-012 to LT-017
```

**4. Qwen Integration Guide** (`04_qwen_integration.md`):
```markdown
# Qwen2.5-0.5B Integration

## Model Specifications
- Architecture: Llama (with GQA)
- Layers: 24
- Hidden dim: 896
- Attention: 14 Q heads, 2 KV heads (GQA)
- FFN dim: 4864
- Context: 32768 tokens
- VRAM: ~900 MB

## Loading
```rust
let model = QwenLoader::load("qwen2.5-0.5b.gguf")?;
```

## Generation
```rust
let encoder = BPEEncoder::from_gguf("qwen2.5-0.5b.gguf")?;
let decoder = BPEDecoder::from_gguf("qwen2.5-0.5b.gguf")?;

let input_ids = encoder.encode("Write a haiku");
let output_ids = QwenForward::prefill(&model, &input_ids, &mut kv_cache, &config)?;

for _ in 0..30 {
    let token = QwenForward::decode(&model, current_token, &mut kv_cache, &config)?;
    // ...
}

let text = decoder.decode(&generated_ids)?;
```

## Performance
- Prefill: ~50ms (10 tokens)
- Decode: ~100ms/token
- Throughput: ~10 tokens/sec

## References
- Stories: LT-022 to LT-027
```

**5. Phi-3 Integration Guide** (`05_phi3_integration.md`):
```markdown
# Phi-3-mini-4k Integration

## Model Specifications
- Architecture: Llama (with MHA)
- Layers: 32
- Hidden dim: 3072
- Attention: 32 Q heads, 32 KV heads (MHA)
- FFN dim: 8192
- Context: 4096 tokens
- VRAM: ~7.5 GB

## Loading
```rust
let model = Phi3Loader::load("phi-3-mini-4k.gguf")?;
```

## Generation
Similar to Qwen but with Phi3Forward.

## Performance
- Prefill: ~100ms (10 tokens)
- Decode: ~150ms/token
- Throughput: ~6-7 tokens/sec

## References
- Stories: LT-029 to LT-032
```

**6. Adapter Usage Guide** (`06_adapter_usage.md`):
```markdown
# LlamaInferenceAdapter Usage

## Overview
Unified interface for all Llama-family models.

## Creating Adapter
```rust
// From model_ref
let mut adapter = LlamaInferenceAdapter::from_model_ref("qwen2.5-0.5b")?;

// Or explicitly
let mut adapter = LlamaInferenceAdapter::new(LlamaVariant::Qwen);
```

## Using Adapter
```rust
// Load
adapter.load(Path::new("qwen2.5-0.5b.gguf"))?;

// Encode
let ids = adapter.encode("Hello")?;

// Generate
let output_ids = adapter.prefill(&ids)?;
let next_token = adapter.decode_token(current_token)?;

// Decode
let text = adapter.decode(&output_ids)?;

// Unload
adapter.unload()?;
```

## Polymorphic Usage
```rust
fn generate(adapter: &mut dyn InferenceAdapter, prompt: &str) -> String {
    let ids = adapter.encode(prompt).unwrap();
    let output_ids = adapter.prefill(&ids).unwrap();
    adapter.decode(&output_ids).unwrap()
}
```

## References
- Story: LT-033
```

**7. API Reference** (`07_api_reference.md`):
- Complete API documentation for all public interfaces
- Generated from code comments

**8. Examples** (`08_examples.md`):
- Complete working examples
- Common use cases
- Best practices

**9. Troubleshooting Guide** (`09_troubleshooting.md`):
- Common issues and solutions
- Error messages and fixes
- Performance tuning

**10. Performance Guide** (`10_performance.md`):
- Performance characteristics
- Optimization tips
- Benchmarking results

---

## Testing Strategy

### Documentation Review
- Technical accuracy review
- Code example validation
- Link checking

### User Testing
- Follow documentation to use features
- Verify examples work
- Identify gaps

### Manual Verification
1. Review all documentation files
2. Test all code examples
3. Verify links and references
4. Check formatting and clarity

---

## Definition of Done

- [ ] All acceptance criteria met
- [ ] All documentation files created
- [ ] All code examples tested
- [ ] Documentation reviewed and approved
- [ ] Links verified
- [ ] Story marked complete in day-tracker.md

---

## References

- All Llama team stories (LT-001 to LT-037)
- Spec: `bin/.specs/01_M0_worker_orcd.md`

---

**Status**: Ready for execution  
**Owner**: Llama-Beta  
**Created**: 2025-10-04

---

**Final Story**: This completes the Llama team deliverables. All 38 stories (LT-001 to LT-038) are now documented and ready for execution.

---

Detailed by Project Management Team — ready to implement 📋

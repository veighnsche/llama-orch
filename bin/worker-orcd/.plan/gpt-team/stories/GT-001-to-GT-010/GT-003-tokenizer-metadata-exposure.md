# GT-003: Tokenizer Metadata Exposure

**Team**: GPT-Gamma  
**Sprint**: Sprint 1 - HF Tokenizer  
**Size**: M (2 days)  
**Days**: 18 - 19  
**Spec Ref**: M0-W-1361, M0-W-1364

---

## Story Description

Expose tokenizer metadata from tokenizer.json including EOS/BOS token IDs, vocab size, and context length. Integrate this metadata into the /health endpoint for observability and debugging.

---

## Acceptance Criteria

- [ ] `eos_id` extracted from tokenizer.json and exposed via API
- [ ] `bos_id` extracted from tokenizer.json and exposed via API
- [ ] `vocab_size` extracted and validated against GGUF metadata
- [ ] `model_max_context` extracted if available in tokenizer.json
- [ ] Metadata accessible via `HfJsonTokenizer::metadata()` method
- [ ] `/health` endpoint includes `tokenizer_kind: "hf-json"`
- [ ] `/health` endpoint includes `vocab_size` field
- [ ] `/health` endpoint includes `context_length` field (if available)
- [ ] Unit tests validate metadata extraction
- [ ] Integration test validates /health endpoint response

---

## Dependencies

### Upstream (Blocks This Story)
- GT-002: tokenizer.json Loading (needs loaded tokenizer)

### Downstream (This Story Blocks)
- GT-004: HF Tokenizer Conformance Tests (needs metadata for validation)
- FT-029: Support GPT Integration (needs tokenizer metadata)

---

## Technical Details

### Files to Create/Modify
- `bin/worker-orcd/src/tokenizer/hf_json.rs` - Add metadata extraction
- `bin/worker-orcd/src/tokenizer/metadata.rs` - TokenizerMetadata struct
- `bin/worker-orcd/src/http/health.rs` - Add tokenizer fields to response

### Key Interfaces
```rust
#[derive(Debug, Clone)]
pub struct TokenizerMetadata {
    pub eos_id: Option<u32>,
    pub bos_id: Option<u32>,
    pub vocab_size: usize,
    pub model_max_context: Option<usize>,
}

impl HfJsonTokenizer {
    pub fn metadata(&self) -> TokenizerMetadata {
        let eos_id = self.inner.token_to_id("

# GT-001: HF Tokenizers Crate Integration

**Team**: GPT-Gamma  
**Sprint**: Sprint 1 - HF Tokenizer  
**Size**: S (1 day)  
**Days**: 15 - 15  
**Spec Ref**: M0-W-1361, M0-W-1365

---

## Story Description

Integrate the Hugging Face `tokenizers` Rust crate into worker-orcd to enable loading and using tokenizer.json files for GPT-OSS-20B. This provides a pure-Rust tokenization backend with no Python or external binary dependencies.

---

## Acceptance Criteria

- [ ] `tokenizers` crate added to `Cargo.toml` with appropriate version
- [ ] Tokenizer module created at `bin/worker-orcd/src/tokenizer/hf_json.rs`
- [ ] `Tokenizer::from_file()` successfully loads tokenizer.json
- [ ] Basic encode/decode functionality working
- [ ] Error handling for missing or invalid tokenizer.json files
- [ ] Unit tests validate tokenizer loads successfully
- [ ] Integration test validates encode/decode round-trip
- [ ] No Python runtime or external binaries required
- [ ] Documentation added to module explaining HF tokenizer backend

---

## Dependencies

### Upstream (Blocks This Story)
- FT-006: FFI Interface Definition (FFI lock on day 15)
- FT-007: Rust FFI Bindings (FFI lock on day 15)

### Downstream (This Story Blocks)
- GT-002: tokenizer.json Loading (needs crate integration)
- GT-003: Tokenizer Metadata Exposure (needs loaded tokenizer)
- GT-004: HF Tokenizer Conformance Tests (needs working tokenizer)

---

## Technical Details

### Files to Create/Modify
- `bin/worker-orcd/Cargo.toml` - Add `tokenizers = "0.15"` dependency
- `bin/worker-orcd/src/tokenizer/mod.rs` - Tokenizer module exports
- `bin/worker-orcd/src/tokenizer/hf_json.rs` - HF tokenizer implementation
- `bin/worker-orcd/src/tokenizer/backend.rs` - TokenizerBackend enum

### Key Interfaces
```rust
use tokenizers::Tokenizer;
use std::path::Path;

pub struct HfJsonTokenizer {
    inner: Tokenizer,
    vocab_size: usize,
}

impl HfJsonTokenizer {
    /// Load tokenizer from tokenizer.json file
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self, TokenizerError>;
    
    /// Encode text to token IDs
    pub fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<u32>, TokenizerError>;
    
    /// Decode token IDs to text
    pub fn decode(&self, token_ids: &[u32], skip_special_tokens: bool) -> Result<String, TokenizerError>;
    
    /// Get vocabulary size
    pub fn vocab_size(&self) -> usize;
}

#[derive(Debug, thiserror::Error)]
pub enum TokenizerError {
    #[error("Failed to load tokenizer from {path}: {source}")]
    LoadFailed {
        path: String,
        source: Box<dyn std::error::Error>,
    },
    #[error("Encoding failed: {0}")]
    EncodeFailed(String),
    #[error("Decoding failed: {0}")]
    DecodeFailed(String),
}
```

### Implementation Notes
- Use `tokenizers::Tokenizer::from_file()` for loading
- Store tokenizer instance in struct for reuse
- Extract vocab size from tokenizer metadata
- Handle file not found and parse errors gracefully
- Log tokenizer load at INFO level with vocab size
- Ensure no Python runtime dependencies
- Pure Rust implementation only

---

## Testing Strategy

### Unit Tests
- Test tokenizer loads from valid tokenizer.json
- Test error handling for missing file
- Test error handling for invalid JSON
- Test vocab_size() returns correct value
- Test basic encode/decode functionality

### Integration Tests
- Test loading GPT-OSS-20B tokenizer.json
- Test encode/decode round-trip preserves text
- Test special tokens handling (BOS/EOS)
- Test empty string encoding
- Test very long text encoding

### Manual Verification
1. Place tokenizer.json in test directory
2. Run: `cargo test hf_json`
3. Verify tokenizer loads successfully
4. Verify encode/decode works
5. Check no Python dependencies required

---

## Definition of Done

- [ ] All acceptance criteria met
- [ ] Code reviewed (self-review for agents)
- [ ] Unit tests passing (5+ tests)
- [ ] Integration tests passing (3+ tests)
- [ ] Documentation updated (module-level docs)
- [ ] Story marked complete in day-tracker.md

---

## References

- Spec: `bin/.specs/01_M0_worker_orcd.md` §8.2 HF-JSON Backend (M0-W-1361)
- Spec: `bin/.specs/01_M0_worker_orcd.md` §8.6 Implementation Requirements (M0-W-1365)
- HF Tokenizers Crate: https://docs.rs/tokenizers/latest/tokenizers/
- Related Stories: GT-002 (loading), GT-003 (metadata), GT-004 (tests)

---

**Status**: 📋 Ready for execution  
**Owner**: GPT-Gamma  
**Created**: 2025-10-04

---
Detailed by Project Management Team — ready to implement 📋

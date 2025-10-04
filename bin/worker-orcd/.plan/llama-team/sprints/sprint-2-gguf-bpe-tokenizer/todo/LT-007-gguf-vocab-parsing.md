# LT-007: GGUF Vocab Parsing

**Team**: Llama-Beta  
**Sprint**: Sprint 2 - GGUF-BPE Tokenizer  
**Size**: M (2 days)  
**Days**: 27-28  
**Spec Ref**: M0-W-1362

---

## Story Description

Parse vocabulary from GGUF metadata to build token-to-ID and ID-to-token mappings for byte-level BPE tokenizer. Extract vocab entries from GGUF file and construct bidirectional lookup tables required for tokenization and detokenization.

---

## Acceptance Criteria

- [ ] Parse `tokenizer.ggml.tokens` array from GGUF metadata
- [ ] Extract token strings and assign sequential IDs (0, 1, 2, ...)
- [ ] Build token-to-ID map (string → u32)
- [ ] Build ID-to-token map (u32 → string)
- [ ] Validate vocab size matches `tokenizer.ggml.token_count`
- [ ] Handle special tokens (BOS, EOS, PAD) from metadata
- [ ] Extract `tokenizer.ggml.bos_token_id` (beginning of sequence)
- [ ] Extract `tokenizer.ggml.eos_token_id` (end of sequence)
- [ ] Unit tests validate vocab parsing for Qwen2.5-0.5B
- [ ] Unit tests validate special token extraction
- [ ] Error handling for missing vocab metadata
- [ ] Log vocab size and special tokens at INFO level

---

## Dependencies

### Upstream (Blocks This Story)
- LT-001: GGUF Header Parser (needs metadata structure)
- LT-002: GGUF Metadata Extraction (needs metadata access)

### Downstream (This Story Blocks)
- LT-009: Byte-Level BPE Encoder (needs vocab maps)
- LT-010: Byte-Level BPE Decoder (needs vocab maps)

---

## Technical Details

### Files to Create/Modify
- `bin/worker-orcd/cuda/src/tokenizer/vocab.cpp` - Vocab parser
- `bin/worker-orcd/cuda/src/tokenizer/vocab.h` - Vocab struct
- `bin/worker-orcd/src/tokenizer/vocab.rs` - Rust vocab struct

### Key Interfaces
```cpp
struct Vocabulary {
    std::unordered_map<std::string, uint32_t> token_to_id;
    std::unordered_map<uint32_t, std::string> id_to_token;
    uint32_t vocab_size;
    
    // Special tokens
    uint32_t bos_token_id;  // Beginning of sequence
    uint32_t eos_token_id;  // End of sequence
    uint32_t pad_token_id;  // Padding (optional)
};

class VocabParser {
public:
    // Parse vocabulary from GGUF metadata
    static Result<Vocabulary> parse(const GGUFMetadata& metadata);
    
private:
    static std::vector<std::string> extract_tokens(const GGUFMetadata& metadata);
    static uint32_t extract_special_token_id(const GGUFMetadata& metadata, const std::string& key);
};
```

```rust
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct Vocabulary {
    pub token_to_id: HashMap<String, u32>,
    pub id_to_token: HashMap<u32, String>,
    pub vocab_size: u32,
    pub bos_token_id: u32,
    pub eos_token_id: u32,
    pub pad_token_id: Option<u32>,
}

impl VocabParser {
    pub fn parse(metadata: &GGUFMetadata) -> Result<Vocabulary, VocabError>;
}
```

### Implementation Notes
- Parse `tokenizer.ggml.tokens` as array of strings
- Assign IDs sequentially: tokens[0] → ID 0, tokens[1] → ID 1, etc.
- Build bidirectional maps for O(1) lookup
- Extract special token IDs from metadata keys:
  - `tokenizer.ggml.bos_token_id`
  - `tokenizer.ggml.eos_token_id`
  - `tokenizer.ggml.pad_token_id` (optional)
- Validate special token IDs are within vocab range
- Log vocab size and special tokens at INFO level
- Handle UTF-8 token strings correctly

**Parsing Logic**:
```cpp
Result<Vocabulary> parse(const GGUFMetadata& metadata) {
    Vocabulary vocab;
    
    // Extract token array
    auto tokens = extract_tokens(metadata);
    vocab.vocab_size = tokens.size();
    
    // Build bidirectional maps
    for (uint32_t id = 0; id < tokens.size(); ++id) {
        vocab.token_to_id[tokens[id]] = id;
        vocab.id_to_token[id] = tokens[id];
    }
    
    // Extract special tokens
    vocab.bos_token_id = extract_special_token_id(metadata, "tokenizer.ggml.bos_token_id");
    vocab.eos_token_id = extract_special_token_id(metadata, "tokenizer.ggml.eos_token_id");
    
    // Validate special tokens
    if (vocab.bos_token_id >= vocab.vocab_size) {
        return Err("Invalid BOS token ID");
    }
    
    return Ok(vocab);
}
```

---

## Testing Strategy

### Unit Tests
- Test vocab parsing for Qwen2.5-0.5B (vocab_size ~151,936)
- Test token-to-ID lookup (forward mapping)
- Test ID-to-token lookup (reverse mapping)
- Test special token extraction (BOS, EOS)
- Test special token validation (within vocab range)
- Test error handling for missing vocab metadata
- Test UTF-8 token handling

### Integration Tests
- Test full vocab parsing from real GGUF file
- Test vocab struct construction
- Test bidirectional lookup consistency

### Manual Verification
1. Load Qwen2.5-0.5B GGUF
2. Parse vocabulary
3. Verify vocab_size ~151,936
4. Verify BOS token ID (e.g., 151643)
5. Verify EOS token ID (e.g., 151645)
6. Check logs show vocab size and special tokens

---

## Definition of Done

- [ ] All acceptance criteria met
- [ ] Code reviewed
- [ ] Unit tests passing (7+ tests)
- [ ] Integration tests passing
- [ ] Documentation updated
- [ ] Story marked complete in day-tracker.md

---

## References

- Spec: `bin/.specs/01_M0_worker_orcd.md` Section 6.4 (Tokenization)
- GGUF Spec: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md
- Qwen2.5 Tokenizer: https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF
- Related Stories: LT-001, LT-002, LT-009, LT-010

---

**Status**: Ready for execution  
**Owner**: Llama-Beta  
**Created**: 2025-10-04

---

Detailed by Project Management Team — ready to implement 📋

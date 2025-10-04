# LT-010: Byte-Level BPE Decoder

**Team**: Llama-Beta  
**Sprint**: Sprint 2 - GGUF-BPE Tokenizer  
**Size**: M (2 days)  
**Days**: 34-35  
**Spec Ref**: M0-W-1362

---

## Story Description

Implement byte-level BPE decoding algorithm to convert token IDs back into text strings. Reverse the encoding process by mapping token IDs to tokens, concatenating byte-level representations, and converting back to UTF-8 text.

---

## Acceptance Criteria

- [ ] Implement byte-level BPE decoding algorithm
- [ ] Convert token IDs to tokens using vocabulary (ID → string)
- [ ] Concatenate byte-level tokens into byte sequence
- [ ] Convert byte sequence back to UTF-8 text
- [ ] Handle special tokens (skip BOS, EOS in output)
- [ ] Handle byte-level characters (Ġ → space, Ċ → newline)
- [ ] Return decoded UTF-8 string
- [ ] Unit tests validate decoding for simple token sequences
- [ ] Unit tests validate round-trip (encode → decode → original text)
- [ ] Error handling for invalid token IDs
- [ ] Error handling for invalid UTF-8 sequences
- [ ] Log decoding statistics (token count, output length)

---

## Dependencies

### Upstream (Blocks This Story)
- LT-007: GGUF Vocab Parsing (needs ID-to-token map)
- LT-009: Byte-Level BPE Encoder (needs round-trip testing)

### Downstream (This Story Blocks)
- LT-011: UTF-8 Safe Streaming Decode (needs decoder)
- LT-025: Qwen Haiku Generation Test (needs detokenization)

---

## Technical Details

### Files to Create/Modify
- `bin/worker-orcd/cuda/src/tokenizer/bpe_decoder.cpp` - BPE decoder
- `bin/worker-orcd/cuda/src/tokenizer/bpe_decoder.h` - Decoder interface
- `bin/worker-orcd/src/tokenizer/decoder.rs` - Rust decoder wrapper

### Key Interfaces
```cpp
class BPEDecoder {
public:
    BPEDecoder(const Vocabulary& vocab);
    
    // Decode token IDs to text
    std::string decode(const std::vector<uint32_t>& token_ids) const;
    
    // Decode with special token handling
    std::string decode_with_special(
        const std::vector<uint32_t>& token_ids,
        bool skip_special_tokens = true
    ) const;
    
private:
    const Vocabulary& vocab_;
    
    std::vector<std::string> ids_to_tokens(const std::vector<uint32_t>& ids) const;
    std::string from_byte_level(const std::vector<std::string>& tokens) const;
    std::string bytes_to_utf8(const std::vector<uint8_t>& bytes) const;
};
```

```rust
pub struct BPEDecoder {
    vocab: Vocabulary,
}

impl BPEDecoder {
    pub fn new(vocab: Vocabulary) -> Self;
    
    pub fn decode(&self, token_ids: &[u32]) -> Result<String, DecodeError>;
    
    pub fn decode_with_special(
        &self,
        token_ids: &[u32],
        skip_special_tokens: bool,
    ) -> Result<String, DecodeError>;
}
```

### Implementation Notes
- **Step 1: IDs to tokens**
  - Lookup each token ID in vocabulary
  - Skip special tokens (BOS, EOS) if requested
  - Handle unknown IDs (return error or replacement token)
  
- **Step 2: Byte-level to bytes**
  - Convert byte-level tokens to byte sequence
  - Map special characters: "Ġ" → space (0x20), "Ċ" → newline (0x0A)
  - Concatenate all bytes
  
- **Step 3: Bytes to UTF-8**
  - Validate UTF-8 sequence
  - Convert bytes to UTF-8 string
  - Handle invalid UTF-8 (return error or replacement char)

**Decoding Algorithm**:
```cpp
std::string decode(const std::vector<uint32_t>& token_ids) const {
    // 1. Convert IDs to tokens
    auto tokens = ids_to_tokens(token_ids);
    
    // 2. Convert byte-level tokens to bytes
    std::vector<uint8_t> bytes;
    for (const auto& token : tokens) {
        // Handle byte-level characters
        if (token == "Ġ") {
            bytes.push_back(0x20);  // space
        } else if (token == "Ċ") {
            bytes.push_back(0x0A);  // newline
        } else {
            // Regular token: convert to bytes
            for (char c : token) {
                bytes.push_back(static_cast<uint8_t>(c));
            }
        }
    }
    
    // 3. Convert bytes to UTF-8 string
    return bytes_to_utf8(bytes);
}

std::vector<std::string> ids_to_tokens(const std::vector<uint32_t>& ids) const {
    std::vector<std::string> tokens;
    for (uint32_t id : ids) {
        // Skip special tokens
        if (id == vocab_.bos_token_id || id == vocab_.eos_token_id) {
            continue;
        }
        
        // Lookup token
        auto it = vocab_.id_to_token.find(id);
        if (it == vocab_.id_to_token.end()) {
            throw DecodeError("Unknown token ID: " + std::to_string(id));
        }
        tokens.push_back(it->second);
    }
    return tokens;
}
```

---

## Testing Strategy

### Unit Tests
- Test decoding simple token sequences ([123, 456] → "hello")
- Test ID-to-token conversion
- Test byte-level to UTF-8 conversion
- Test special token handling (skip BOS, EOS)
- Test byte-level character mapping (Ġ → space)
- Test round-trip encoding/decoding (text → IDs → text)
- Test error handling for invalid token IDs
- Test error handling for invalid UTF-8

### Integration Tests
- Test full decoding pipeline (token IDs → text)
- Test decoding with real Qwen2.5 vocab
- Test round-trip with various text inputs

### Manual Verification
1. Encode "Hello, world!" → token IDs
2. Decode token IDs → text
3. Verify output matches input
4. Test with emoji, CJK characters
5. Check logs show decoding statistics

---

## Definition of Done

- [ ] All acceptance criteria met
- [ ] Code reviewed
- [ ] Unit tests passing (8+ tests)
- [ ] Integration tests passing
- [ ] Round-trip tests passing
- [ ] Documentation updated
- [ ] Story marked complete in day-tracker.md

---

## References

- Spec: `bin/.specs/01_M0_worker_orcd.md` Section 6.4 (Tokenization)
- BPE Paper: https://arxiv.org/abs/1508.07909
- Byte-Level BPE: https://arxiv.org/abs/1909.03341
- UTF-8 Spec: https://www.rfc-editor.org/rfc/rfc3629
- Related Stories: LT-007, LT-009, LT-011

---

**Status**: Ready for execution  
**Owner**: Llama-Beta  
**Created**: 2025-10-04

---

Detailed by Project Management Team — ready to implement 📋

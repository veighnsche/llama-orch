# LT-009: Byte-Level BPE Encoder

**Team**: Llama-Beta  
**Sprint**: Sprint 2 - GGUF-BPE Tokenizer  
**Size**: M (3 days)  
**Days**: 31-33  
**Spec Ref**: M0-W-1362

---

## Story Description

Implement byte-level BPE encoding algorithm to convert text strings into token IDs. Apply BPE merge rules iteratively to encode input text using vocabulary and merge table extracted from GGUF metadata.

---

## Acceptance Criteria

- [ ] Implement byte-level BPE encoding algorithm
- [ ] Convert input text to byte-level representation (UTF-8 → bytes)
- [ ] Apply BPE merges iteratively (lowest priority first)
- [ ] Convert merged tokens to token IDs using vocabulary
- [ ] Handle special tokens (BOS, EOS) prepending/appending
- [ ] Handle unknown characters (fallback to byte tokens)
- [ ] Return vector of token IDs (Vec<u32>)
- [ ] Unit tests validate encoding for simple strings
- [ ] Unit tests validate encoding matches reference tokenizer
- [ ] Conformance tests with Qwen2.5 test vectors (LT-018)
- [ ] Error handling for encoding failures
- [ ] Log encoding statistics (input length, token count)

---

## Dependencies

### Upstream (Blocks This Story)
- LT-007: GGUF Vocab Parsing (needs vocab maps)
- LT-008: GGUF Merges Parsing (needs merge table)

### Downstream (This Story Blocks)
- LT-018: Tokenizer Conformance Tests (needs encoder)
- LT-024: Qwen Forward Pass (needs tokenization)

---

## Technical Details

### Files to Create/Modify
- `bin/worker-orcd/cuda/src/tokenizer/bpe_encoder.cpp` - BPE encoder
- `bin/worker-orcd/cuda/src/tokenizer/bpe_encoder.h` - Encoder interface
- `bin/worker-orcd/src/tokenizer/encoder.rs` - Rust encoder wrapper

### Key Interfaces
```cpp
class BPEEncoder {
public:
    BPEEncoder(const Vocabulary& vocab, const MergeTable& merges);
    
    // Encode text to token IDs
    std::vector<uint32_t> encode(const std::string& text) const;
    
    // Encode with special tokens
    std::vector<uint32_t> encode_with_special(
        const std::string& text,
        bool add_bos = true,
        bool add_eos = true
    ) const;
    
private:
    const Vocabulary& vocab_;
    const MergeTable& merges_;
    
    std::vector<std::string> to_byte_level(const std::string& text) const;
    std::vector<std::string> apply_merges(std::vector<std::string> tokens) const;
    std::vector<uint32_t> tokens_to_ids(const std::vector<std::string>& tokens) const;
};
```

```rust
pub struct BPEEncoder {
    vocab: Vocabulary,
    merges: MergeTable,
}

impl BPEEncoder {
    pub fn new(vocab: Vocabulary, merges: MergeTable) -> Self;
    
    pub fn encode(&self, text: &str) -> Vec<u32>;
    
    pub fn encode_with_special(
        &self,
        text: &str,
        add_bos: bool,
        add_eos: bool,
    ) -> Vec<u32>;
}
```

### Implementation Notes
- **Step 1: Byte-level conversion**
  - Convert UTF-8 text to byte-level tokens
  - Map bytes to printable characters (e.g., space → "Ġ")
  
- **Step 2: Apply BPE merges**
  - Iterate through merge table (priority 0 → N)
  - Find adjacent token pairs matching merge rule
  - Merge pairs into single token
  - Repeat until no more merges possible
  
- **Step 3: Convert to IDs**
  - Lookup each token in vocabulary
  - Return token IDs
  - Handle unknown tokens (fallback to byte tokens)

**BPE Algorithm**:
```cpp
std::vector<uint32_t> encode(const std::string& text) const {
    // 1. Convert to byte-level tokens
    auto tokens = to_byte_level(text);
    
    // 2. Apply merges iteratively
    tokens = apply_merges(tokens);
    
    // 3. Convert to IDs
    return tokens_to_ids(tokens);
}

std::vector<std::string> apply_merges(std::vector<std::string> tokens) const {
    // Iterate through all merges by priority
    for (const auto& [pair, priority] : merges_.merge_priority) {
        bool merged = true;
        while (merged) {
            merged = false;
            std::vector<std::string> new_tokens;
            
            for (size_t i = 0; i < tokens.size(); ++i) {
                if (i + 1 < tokens.size() && 
                    tokens[i] == pair.left && 
                    tokens[i + 1] == pair.right) {
                    // Merge pair
                    new_tokens.push_back(tokens[i] + tokens[i + 1]);
                    ++i;  // Skip next token
                    merged = true;
                } else {
                    new_tokens.push_back(tokens[i]);
                }
            }
            
            tokens = new_tokens;
        }
    }
    
    return tokens;
}
```

---

## Testing Strategy

### Unit Tests
- Test encoding simple strings ("hello" → token IDs)
- Test byte-level conversion (UTF-8 → byte tokens)
- Test BPE merge application (verify merge order)
- Test token-to-ID conversion
- Test special token handling (BOS, EOS)
- Test unknown character handling
- Test empty string encoding

### Conformance Tests (LT-018)
- Test encoding matches Qwen2.5 reference tokenizer
- Test 20-30 test vectors (text → expected token IDs)
- Test edge cases (emoji, CJK characters, special chars)

### Integration Tests
- Test full encoding pipeline (text → token IDs)
- Test encoding with real Qwen2.5 vocab and merges

### Manual Verification
1. Encode "Hello, world!" with Qwen2.5 tokenizer
2. Verify token IDs match reference implementation
3. Encode with BOS/EOS tokens
4. Check logs show encoding statistics

---

## Definition of Done

- [ ] All acceptance criteria met
- [ ] Code reviewed
- [ ] Unit tests passing (7+ tests)
- [ ] Conformance tests passing (deferred to LT-018)
- [ ] Integration tests passing
- [ ] Documentation updated
- [ ] Story marked complete in day-tracker.md

---

## References

- Spec: `bin/.specs/01_M0_worker_orcd.md` Section 6.4 (Tokenization)
- BPE Paper: https://arxiv.org/abs/1508.07909
- Byte-Level BPE: https://arxiv.org/abs/1909.03341
- Qwen2.5 Tokenizer: https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF
- Related Stories: LT-007, LT-008, LT-018

---

**Status**: Ready for execution  
**Owner**: Llama-Beta  
**Created**: 2025-10-04

---

Detailed by Project Management Team — ready to implement 📋

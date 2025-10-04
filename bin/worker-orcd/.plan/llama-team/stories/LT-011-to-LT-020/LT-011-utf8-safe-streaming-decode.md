# LT-011: UTF-8 Safe Streaming Decode

**Team**: Llama-Beta  
**Sprint**: Sprint 3 - UTF-8 Safety + Llama Kernels  
**Size**: M (2 days)  
**Days**: 36-37  
**Spec Ref**: M0-W-1362

---

## Story Description

Implement UTF-8 safe streaming decoder to handle partial token decoding during SSE streaming. Buffer incomplete UTF-8 sequences at token boundaries to prevent broken characters in streaming output, ensuring all emitted text is valid UTF-8.

---

## Acceptance Criteria

- [ ] Implement streaming decoder that buffers incomplete UTF-8 sequences
- [ ] Detect UTF-8 continuation bytes (0x80-0xBF) at token boundaries
- [ ] Buffer incomplete sequences until complete character received
- [ ] Emit only complete UTF-8 characters in streaming output
- [ ] Handle multi-byte UTF-8 sequences (2-4 bytes)
- [ ] Flush remaining buffer at end of stream
- [ ] Unit tests validate UTF-8 boundary detection
- [ ] Unit tests validate multi-byte character handling (emoji, CJK)
- [ ] Integration tests validate streaming with partial tokens
- [ ] Error handling for invalid UTF-8 sequences
- [ ] Log buffering events at DEBUG level

---

## Dependencies

### Upstream (Blocks This Story)
- LT-010: Byte-Level BPE Decoder (needs decoder)

### Downstream (This Story Blocks)
- LT-025: Qwen Haiku Generation Test (needs streaming decode)
- FT-003: SSE Streaming (needs UTF-8 safe output)

---

## Technical Details

### Files to Create/Modify
- `bin/worker-orcd/cuda/src/tokenizer/streaming_decoder.cpp` - Streaming decoder
- `bin/worker-orcd/cuda/src/tokenizer/streaming_decoder.h` - Decoder interface
- `bin/worker-orcd/src/tokenizer/streaming.rs` - Rust streaming wrapper

### Key Interfaces
```cpp
class StreamingDecoder {
public:
    StreamingDecoder(const Vocabulary& vocab);
    
    // Decode single token with UTF-8 safety
    std::string decode_token(uint32_t token_id);
    
    // Flush remaining buffer (call at end of stream)
    std::string flush();
    
    // Reset decoder state
    void reset();
    
private:
    const Vocabulary& vocab_;
    std::vector<uint8_t> utf8_buffer_;  // Buffer for incomplete sequences
    
    bool is_utf8_continuation(uint8_t byte) const;
    bool is_complete_utf8_sequence(const std::vector<uint8_t>& bytes) const;
    int expected_utf8_length(uint8_t first_byte) const;
};
```

```rust
pub struct StreamingDecoder {
    vocab: Vocabulary,
    utf8_buffer: Vec<u8>,
}

impl StreamingDecoder {
    pub fn new(vocab: Vocabulary) -> Self;
    
    pub fn decode_token(&mut self, token_id: u32) -> String;
    
    pub fn flush(&mut self) -> String;
    
    pub fn reset(&mut self);
}
```

### Implementation Notes

**UTF-8 Encoding Rules**:
- 1-byte: 0xxxxxxx (ASCII, 0x00-0x7F)
- 2-byte: 110xxxxx 10xxxxxx (0xC0-0xDF, 0x80-0xBF)
- 3-byte: 1110xxxx 10xxxxxx 10xxxxxx (0xE0-0xEF, 0x80-0xBF, 0x80-0xBF)
- 4-byte: 11110xxx 10xxxxxx 10xxxxxx 10xxxxxx (0xF0-0xF7, 0x80-0xBF, ...)

**Streaming Algorithm**:
```cpp
std::string decode_token(uint32_t token_id) {
    // 1. Decode token to bytes
    std::string token_str = vocab_.id_to_token[token_id];
    std::vector<uint8_t> token_bytes(token_str.begin(), token_str.end());
    
    // 2. Append to buffer
    utf8_buffer_.insert(utf8_buffer_.end(), token_bytes.begin(), token_bytes.end());
    
    // 3. Extract complete UTF-8 sequences
    std::string output;
    size_t i = 0;
    while (i < utf8_buffer_.size()) {
        int seq_len = expected_utf8_length(utf8_buffer_[i]);
        
        // Check if we have complete sequence
        if (i + seq_len <= utf8_buffer_.size()) {
            // Complete sequence: emit it
            output.append(utf8_buffer_.begin() + i, utf8_buffer_.begin() + i + seq_len);
            i += seq_len;
        } else {
            // Incomplete sequence: keep in buffer
            break;
        }
    }
    
    // 4. Remove emitted bytes from buffer
    utf8_buffer_.erase(utf8_buffer_.begin(), utf8_buffer_.begin() + i);
    
    return output;
}

int expected_utf8_length(uint8_t first_byte) const {
    if ((first_byte & 0x80) == 0x00) return 1;  // 0xxxxxxx
    if ((first_byte & 0xE0) == 0xC0) return 2;  // 110xxxxx
    if ((first_byte & 0xF0) == 0xE0) return 3;  // 1110xxxx
    if ((first_byte & 0xF8) == 0xF0) return 4;  // 11110xxx
    return 1;  // Invalid, treat as single byte
}
```

---

## Testing Strategy

### Unit Tests
- Test ASCII characters (no buffering needed)
- Test 2-byte UTF-8 split across tokens (buffer + emit)
- Test 3-byte UTF-8 split across tokens (emoji)
- Test 4-byte UTF-8 split across tokens (rare emoji)
- Test CJK characters (3-byte sequences)
- Test flush() with remaining buffer
- Test reset() clears buffer
- Test invalid UTF-8 handling

### Integration Tests
- Test streaming decode with real token sequence
- Test decode matches non-streaming decoder (same output)
- Test streaming with Qwen tokenizer output

### Edge Cases
- Token boundary splits 2-byte char: "é" (0xC3 0xA9) → [0xC3] | [0xA9]
- Token boundary splits 3-byte char: "中" (0xE4 0xB8 0xAD) → [0xE4] | [0xB8 0xAD]
- Token boundary splits 4-byte emoji: "😀" (0xF0 0x9F 0x98 0x80) → [0xF0 0x9F] | [0x98 0x80]

### Manual Verification
1. Stream decode "Hello 世界 😀" token by token
2. Verify no broken UTF-8 in output
3. Verify final flush() emits remaining chars
4. Check logs show buffering events

---

## Definition of Done

- [ ] All acceptance criteria met
- [ ] Code reviewed
- [ ] Unit tests passing (8+ tests)
- [ ] Integration tests passing
- [ ] Edge case tests passing
- [ ] Documentation updated
- [ ] Story marked complete in day-tracker.md

---

## References

- Spec: `bin/.specs/01_M0_worker_orcd.md` Section 6.4 (Tokenization)
- UTF-8 Spec: https://www.rfc-editor.org/rfc/rfc3629
- UTF-8 Encoding: https://en.wikipedia.org/wiki/UTF-8
- Related Stories: LT-010, LT-025, FT-003

---

**Status**: Ready for execution  
**Owner**: Llama-Beta  
**Created**: 2025-10-04

---

Detailed by Project Management Team — ready to implement 📋

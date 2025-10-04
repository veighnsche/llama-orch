# LT-018: Tokenizer Conformance Tests (Qwen)

**Team**: Llama-Beta  
**Sprint**: Sprint 4 - GQA Attention + Integration  
**Size**: M (2 days)  
**Days**: 50-51  
**Spec Ref**: M0-W-1363

---

## Story Description

Create comprehensive conformance test suite for Qwen2.5 byte-level BPE tokenizer. Validate encoding and decoding against reference implementation with 20-30 test vectors covering edge cases, multilingual text, and special characters.

---

## Acceptance Criteria

- [ ] Create 20-30 tokenizer test vectors (text → expected token IDs)
- [ ] Test ASCII text encoding/decoding
- [ ] Test UTF-8 multibyte characters (emoji, CJK)
- [ ] Test special characters (punctuation, symbols)
- [ ] Test whitespace handling (spaces, tabs, newlines)
- [ ] Test empty string and single character inputs
- [ ] Test very long sequences (>1000 tokens)
- [ ] Test round-trip encoding/decoding (text → IDs → text)
- [ ] Compare against Qwen2.5 reference tokenizer (HuggingFace)
- [ ] All test vectors pass with exact token ID match
- [ ] Document test vectors in markdown table
- [ ] Error handling for tokenization failures
- [ ] Log test results with pass/fail status

---

## Dependencies

### Upstream (Blocks This Story)
- LT-009: Byte-Level BPE Encoder (needs encoder)
- LT-010: Byte-Level BPE Decoder (needs decoder)

### Downstream (This Story Blocks)
- LT-025: Qwen Haiku Generation Test (needs validated tokenizer)

---

## Technical Details

### Files to Create/Modify
- `bin/worker-orcd/tests/tokenizer_conformance_qwen.cpp` - Conformance tests
- `bin/worker-orcd/tests/test_vectors_qwen.h` - Test vector data
- `bin/worker-orcd/.docs/tokenizer_test_vectors_qwen.md` - Test documentation

### Test Vector Format
```cpp
struct TokenizerTestVector {
    std::string text;
    std::vector<uint32_t> expected_token_ids;
    std::string description;
};

const std::vector<TokenizerTestVector> QWEN_TEST_VECTORS = {
    {
        "Hello, world!",
        {9906, 11, 1917, 0},  // Example IDs (verify with reference)
        "Basic ASCII greeting"
    },
    {
        "你好世界",
        {108386, 53901, 104643, 99489},  // Example IDs
        "Chinese greeting"
    },
    {
        "Hello 😀 world",
        {9906, 62904, 1917},  // Example IDs
        "Emoji in text"
    },
    // ... 17+ more test vectors
};
```

### Test Categories

**1. ASCII Text** (5 vectors):
- Simple greeting: "Hello, world!"
- Sentence with punctuation: "The quick brown fox jumps over the lazy dog."
- Numbers: "The year is 2024."
- Mixed case: "CamelCase and snake_case"
- Special chars: "Email: user@example.com"

**2. UTF-8 Multibyte** (5 vectors):
- Chinese: "你好世界" (Hello world)
- Japanese: "こんにちは" (Hello)
- Korean: "안녕하세요" (Hello)
- Arabic: "مرحبا" (Hello)
- Emoji: "Hello 😀🌍🚀"

**3. Whitespace** (3 vectors):
- Multiple spaces: "Hello    world"
- Tabs: "Hello\tworld"
- Newlines: "Hello\nworld"

**4. Edge Cases** (5 vectors):
- Empty string: ""
- Single char: "a"
- Single emoji: "😀"
- Very long: (1000+ char string)
- Special tokens: "<BOS>Hello<EOS>"

**5. Round-Trip** (2 vectors):
- ASCII round-trip: "Hello, world!" → IDs → "Hello, world!"
- Multilingual round-trip: "Hello 你好 😀" → IDs → "Hello 你好 😀"

### Implementation Notes

**Test Execution**:
```cpp
void run_conformance_tests() {
    int passed = 0;
    int failed = 0;
    
    for (const auto& test : QWEN_TEST_VECTORS) {
        // Encode
        auto actual_ids = encoder.encode(test.text);
        
        // Compare token IDs
        if (actual_ids == test.expected_token_ids) {
            passed++;
            tracing::info!("PASS: {}", test.description);
        } else {
            failed++;
            tracing::error!("FAIL: {}", test.description);
            tracing::error!("  Expected: {:?}", test.expected_token_ids);
            tracing::error!("  Actual:   {:?}", actual_ids);
        }
        
        // Round-trip test
        auto decoded = decoder.decode(actual_ids);
        if (decoded != test.text) {
            tracing::error!("Round-trip FAIL: {} != {}", decoded, test.text);
        }
    }
    
    tracing::info!("Conformance tests: {} passed, {} failed", passed, failed);
}
```

**Reference Tokenizer**:
```python
# Generate test vectors using HuggingFace Qwen tokenizer
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")

test_texts = [
    "Hello, world!",
    "你好世界",
    # ... all test cases
]

for text in test_texts:
    token_ids = tokenizer.encode(text)
    print(f'{{"{text}", {{{", ".join(map(str, token_ids))}}}, "..."}},')
```

---

## Testing Strategy

### Unit Tests
- Test each test vector individually
- Test encoding (text → IDs)
- Test decoding (IDs → text)
- Test round-trip (text → IDs → text)
- Test exact token ID match
- Test decoded text match

### Integration Tests
- Test full conformance suite (all 20-30 vectors)
- Test against HuggingFace reference tokenizer
- Test error handling for invalid inputs

### Manual Verification
1. Run conformance test suite
2. Verify all tests pass
3. Compare with HuggingFace tokenizer output
4. Check logs show test results

---

## Definition of Done

- [ ] All acceptance criteria met
- [ ] Code reviewed
- [ ] 20-30 test vectors created
- [ ] All conformance tests passing
- [ ] Round-trip tests passing
- [ ] Documentation updated
- [ ] Story marked complete in day-tracker.md

---

## References

- Spec: `bin/.specs/01_M0_worker_orcd.md` Section 6.4 (Tokenization)
- Qwen2.5 Tokenizer: https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct
- BPE Paper: https://arxiv.org/abs/1508.07909
- Related Stories: LT-009, LT-010, LT-025

---

**Status**: Ready for execution  
**Owner**: Llama-Beta  
**Created**: 2025-10-04

---

Detailed by Project Management Team — ready to implement 📋

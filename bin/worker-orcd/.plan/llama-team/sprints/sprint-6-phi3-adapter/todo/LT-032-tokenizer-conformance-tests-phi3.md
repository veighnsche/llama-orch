# LT-032: Tokenizer Conformance Tests (Phi-3)

**Team**: Llama-Beta  
**Sprint**: Sprint 6 - Phi-3 + Adapter  
**Size**: M (2 days)  
**Days**: 73-74  
**Spec Ref**: M0-W-1363

---

## Story Description

Create conformance test suite for Phi-3 tokenizer. Validate encoding and decoding against reference implementation, ensuring compatibility with Phi-3's vocabulary and BPE merges.

---

## Acceptance Criteria

- [ ] Create 15-20 tokenizer test vectors for Phi-3
- [ ] Test ASCII text encoding/decoding
- [ ] Test UTF-8 multibyte characters (emoji, CJK)
- [ ] Test special characters and punctuation
- [ ] Test whitespace handling
- [ ] Test Phi-3 special tokens (if different from Qwen)
- [ ] Test round-trip encoding/decoding
- [ ] Compare against Phi-3 reference tokenizer (HuggingFace)
- [ ] All test vectors pass with exact token ID match
- [ ] Document test vectors in markdown table
- [ ] Error handling for tokenization failures
- [ ] Log test results with pass/fail status

---

## Dependencies

### Upstream (Blocks This Story)
- LT-031: Phi-3 Forward Pass (needs working model)
- LT-018: Tokenizer Conformance Tests Qwen (needs test template)

### Downstream (This Story Blocks)
- LT-033: LlamaInferenceAdapter (needs validated tokenizer)
- LT-035: Llama Integration Test Suite (needs both tokenizers)

---

## Technical Details

### Files to Create/Modify
- `bin/worker-orcd/tests/tokenizer_conformance_phi3.cpp` - Phi-3 conformance tests
- `bin/worker-orcd/tests/test_vectors_phi3.h` - Phi-3 test vector data
- `bin/worker-orcd/.docs/tokenizer_test_vectors_phi3.md` - Test documentation

### Test Vector Format
```cpp
struct TokenizerTestVector {
    std::string text;
    std::vector<uint32_t> expected_token_ids;
    std::string description;
};

const std::vector<TokenizerTestVector> PHI3_TEST_VECTORS = {
    {
        "Hello, world!",
        {22557, 29892, 3186, 29991},  // Example IDs (verify with reference)
        "Basic ASCII greeting"
    },
    {
        "你好世界",
        {30919, 31076, 30408, 30967},  // Example IDs
        "Chinese greeting"
    },
    {
        "The quick brown fox",
        {450, 4996, 17354, 1701, 29916},  // Example IDs
        "Common test phrase"
    },
    // ... 12+ more test vectors
};
```

### Test Categories

**1. ASCII Text** (5 vectors):
- Simple greeting: "Hello, world!"
- Common phrase: "The quick brown fox jumps over the lazy dog."
- Numbers: "The year is 2024."
- Mixed case: "CamelCase and snake_case"
- Special chars: "Email: user@example.com"

**2. UTF-8 Multibyte** (4 vectors):
- Chinese: "你好世界"
- Japanese: "こんにちは"
- Emoji: "Hello 😀🌍"
- Arabic: "مرحبا"

**3. Whitespace** (2 vectors):
- Multiple spaces: "Hello    world"
- Newlines: "Hello\nworld"

**4. Edge Cases** (4 vectors):
- Empty string: ""
- Single char: "a"
- Very long: (500+ char string)
- Special tokens: "<BOS>Hello<EOS>"

### Implementation Notes

**Test Execution**:
```cpp
void run_phi3_conformance_tests() {
    int passed = 0;
    int failed = 0;
    
    auto encoder = BPEEncoder::from_gguf("phi-3-mini-4k.gguf");
    auto decoder = BPEDecoder::from_gguf("phi-3-mini-4k.gguf");
    
    for (const auto& test : PHI3_TEST_VECTORS) {
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
    
    tracing::info!("Phi-3 conformance tests: {} passed, {} failed", passed, failed);
}
```

**Reference Tokenizer**:
```python
# Generate test vectors using HuggingFace Phi-3 tokenizer
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")

test_texts = [
    "Hello, world!",
    "你好世界",
    "The quick brown fox",
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
- Test full conformance suite (all 15-20 vectors)
- Test against HuggingFace reference tokenizer
- Test error handling for invalid inputs

### Comparison Tests
- Compare Phi-3 tokenizer with Qwen tokenizer
- Verify both use same BPE algorithm
- Document vocabulary differences

### Manual Verification
1. Run Phi-3 conformance test suite
2. Verify all tests pass
3. Compare with HuggingFace tokenizer output
4. Check logs show test results

---

## Definition of Done

- [ ] All acceptance criteria met
- [ ] Code reviewed
- [ ] 15-20 test vectors created
- [ ] All conformance tests passing
- [ ] Round-trip tests passing
- [ ] Documentation updated
- [ ] Story marked complete in day-tracker.md

---

## References

- Spec: `bin/.specs/01_M0_worker_orcd.md` Section 6.4 (Tokenization)
- Phi-3 Tokenizer: https://huggingface.co/microsoft/Phi-3-mini-4k-instruct
- Related Stories: LT-031, LT-018, LT-033

---

**Status**: Ready for execution  
**Owner**: Llama-Beta  
**Created**: 2025-10-04

---

Detailed by Project Management Team — ready to implement 📋

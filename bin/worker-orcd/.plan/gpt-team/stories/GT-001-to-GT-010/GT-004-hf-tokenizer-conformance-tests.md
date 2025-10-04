# GT-004: HF Tokenizer Conformance Tests

**Team**: GPT-Gamma  
**Sprint**: Sprint 1 (HF Tokenizer)  
**Size**: S (1 day)  
**Days**: 19  
**Spec Ref**: M0-W-1304, M0-W-1305

---

## Story Description

Implement comprehensive conformance tests for the HuggingFace tokenizer integration to validate encode/decode correctness against reference implementations. Tests must verify byte-level BPE behavior, special token handling, and UTF-8 safety for GPT-OSS-20B tokenization.

---

## Acceptance Criteria

- [ ] Test suite validates tokenizer encode matches reference for 10+ test cases
- [ ] Test suite validates tokenizer decode matches reference for 10+ test cases
- [ ] Test cases include ASCII, UTF-8 multibyte, special tokens, edge cases
- [ ] Test validates special token IDs match expected values
- [ ] Test validates vocabulary size matches expected count
- [ ] Test validates unknown token handling
- [ ] Test validates whitespace handling
- [ ] Test validates empty string encode/decode
- [ ] All tests passing with 100% success rate
- [ ] Documentation updated with conformance test results

---

## Dependencies

### Upstream (Blocks This Story)
- GT-001: HF Tokenizers Crate Integration (Expected: Day 15)
- GT-002: tokenizer.json Loading (Expected: Day 17)
- GT-003: Tokenizer Metadata Exposure (Expected: Day 18)

### Downstream (This Story Blocks)
- GT-005: GPT GGUF Metadata Parsing (needs validated tokenizer)

---

## Technical Details

### Files to Create/Modify
- `bin/worker-orcd/src/tokenizer/hf_tokenizer_tests.rs` - Conformance test suite
- `bin/worker-orcd/tests/fixtures/tokenizer_test_cases.json` - Test case data
- `bin/worker-orcd/tests/integration/tokenizer_conformance_test.rs` - Integration tests

### Key Interfaces
```rust
#[cfg(test)]
mod conformance_tests {
    use super::*;
    
    #[test]
    fn test_ascii_roundtrip() {
        let tok = load_test_tokenizer();
        let text = "Hello, world!";
        let ids = tok.encode(text, false).unwrap();
        let decoded = tok.decode(&ids, true).unwrap();
        assert_eq!(decoded, text);
    }
    
    #[test]
    fn test_utf8_multibyte() {
        let tok = load_test_tokenizer();
        let text = "Hello world";
        let ids = tok.encode(text, false).unwrap();
        let decoded = tok.decode(&ids, true).unwrap();
        assert_eq!(decoded, text);
    }
}
```

### Implementation Notes
- Use reference tokenizer outputs from HF transformers library
- Test against known good token sequences
- Validate special token IDs match GPT-OSS-20B expectations
- Test edge cases: empty strings, very long text, special characters
- Ensure UTF-8 safety in all encode/decode operations

---

## Testing Strategy

### Unit Tests
- Test ASCII text encode/decode roundtrip
- Test UTF-8 multibyte character handling
- Test special token insertion and removal
- Test vocabulary size validation
- Test unknown token handling

### Integration Tests
- Test full GPT-OSS-20B tokenizer conformance
- Test against reference implementation outputs
- Test edge cases and boundary conditions

### Manual Verification
1. Run conformance test suite
2. Compare outputs with HF transformers reference
3. Verify 100% test pass rate
4. Document any discrepancies

---

## Definition of Done

- [ ] All acceptance criteria met
- [ ] Code reviewed
- [ ] All tests passing
- [ ] Documentation updated
- [ ] Story marked complete in day-tracker.md

---

## References

- Spec: `bin/.specs/01_M0_worker_orcd.md` Section 8 (Tokenization)
- HF Tokenizers: https://docs.rs/tokenizers/
- Related Stories: GT-001, GT-002, GT-003

---

**Status**: Ready for execution  
**Owner**: GPT-Gamma  
**Created**: 2025-10-04

---
Detailed by Project Management Team — ready to implement 📋

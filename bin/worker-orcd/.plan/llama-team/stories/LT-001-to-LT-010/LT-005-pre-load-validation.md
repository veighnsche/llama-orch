# LT-005: Pre-Load Validation

**Team**: Llama-Beta  
**Sprint**: Sprint 1 - GGUF Foundation  
**Size**: M (2 days)  
**Days**: 24-25  
**Spec Ref**: M0-W-1210

---

## Story Description

Implement comprehensive pre-load validation for GGUF files before attempting to load model weights into VRAM. Validate file integrity, architecture compatibility, VRAM requirements, and security constraints to prevent loading failures and security vulnerabilities.

---

## Acceptance Criteria

- [ ] Validate GGUF file exists and is readable
- [ ] Validate GGUF magic bytes and version
- [ ] Validate architecture is supported ("llama")
- [ ] Calculate total VRAM requirement from tensor sizes
- [ ] Validate total VRAM requirement fits in available VRAM
- [ ] Validate all tensor offsets and sizes (security)
- [ ] Validate tensor count is reasonable (<10,000)
- [ ] Validate metadata is well-formed (no corrupted keys)
- [ ] Return validation report with pass/fail and details
- [ ] Unit tests validate each validation check
- [ ] Integration tests validate full pre-load validation
- [ ] Error handling with clear, actionable error messages
- [ ] Audit log for rejected GGUF files (security)

**Security Criteria (M0-W-1211a)**:
- [ ] All tensor offsets validated before mmap
- [ ] All tensor sizes validated before VRAM allocation
- [ ] Clear error messages for malformed GGUF files
- [ ] Worker exits cleanly on validation failure (no partial load)
- [ ] Audit log entry for rejected GGUF files (include file hash)

---

## Dependencies

### Upstream (Blocks This Story)
- LT-001: GGUF Header Parser (needs header validation)
- LT-002: GGUF Metadata Extraction (needs metadata validation)

### Downstream (This Story Blocks)
- LT-023: Qwen Weight Loading (needs validation pass)
- LT-030: Phi-3 Weight Loading (needs validation pass)

---

## Technical Details

### Files to Create/Modify
- `bin/worker-orcd/cuda/src/validation/pre_load.cpp` - Pre-load validation
- `bin/worker-orcd/cuda/src/validation/pre_load.h` - Validation interface
- `bin/worker-orcd/src/validation/pre_load.rs` - Rust validation wrapper

### Key Interfaces
```cpp
struct ValidationReport {
    bool passed;
    std::string error_message;
    std::vector<std::string> warnings;
    
    // Validation details
    size_t total_vram_required;
    size_t available_vram;
    uint64_t tensor_count;
    std::string architecture;
};

class PreLoadValidator {
public:
    // Validate GGUF file before loading
    static ValidationReport validate(
        const std::string& gguf_path,
        size_t available_vram
    );
    
private:
    static bool validate_file_access(const std::string& path);
    static bool validate_header(const GGUFHeader& header);
    static bool validate_metadata(const GGUFMetadata& metadata);
    static bool validate_vram_requirements(
        const std::vector<GGUFTensor>& tensors,
        size_t available_vram
    );
    static bool validate_tensor_bounds(
        const std::vector<GGUFTensor>& tensors,
        size_t file_size
    );
};
```

```rust
#[derive(Debug, Clone)]
pub struct ValidationReport {
    pub passed: bool,
    pub error_message: Option<String>,
    pub warnings: Vec<String>,
    pub total_vram_required: usize,
    pub available_vram: usize,
    pub tensor_count: u64,
    pub architecture: String,
}

impl PreLoadValidator {
    pub fn validate(
        gguf_path: &Path,
        available_vram: usize,
    ) -> ValidationReport;
}
```

### Implementation Notes
- Run all validation checks in order
- Fail fast on critical errors (file not found, invalid magic)
- Collect warnings for non-critical issues
- Calculate VRAM requirement: sum of all tensor sizes + overhead (10%)
- Validate architecture is "llama" (reject "gpt", "mpt", etc.)
- Log validation result at INFO level (pass) or ERROR level (fail)
- Include file hash in audit log for security tracking

**Validation Sequence**:
```cpp
ValidationReport validate(const std::string& path, size_t available_vram) {
    // 1. File access
    if (!validate_file_access(path)) {
        return {false, "File not found or not readable", ...};
    }
    
    // 2. Parse header
    auto header = parse_gguf_header(path);
    if (!validate_header(header)) {
        return {false, "Invalid GGUF header", ...};
    }
    
    // 3. Parse metadata
    auto metadata = parse_gguf_metadata(path);
    if (!validate_metadata(metadata)) {
        return {false, "Invalid metadata", ...};
    }
    
    // 4. Security: Validate tensor bounds
    if (!validate_tensor_bounds(header.tensors, file_size)) {
        audit_log("GGUF validation failed: tensor bounds", path);
        return {false, "Security: Invalid tensor offsets", ...};
    }
    
    // 5. VRAM requirements
    size_t total_vram = calculate_vram_requirement(header.tensors);
    if (total_vram > available_vram) {
        return {false, "Insufficient VRAM", ...};
    }
    
    return {true, "", {}, total_vram, available_vram, ...};
}
```

---

## Testing Strategy

### Unit Tests
- Test file access validation (file exists, readable)
- Test header validation (magic bytes, version)
- Test metadata validation (required keys present)
- Test VRAM requirement calculation
- Test VRAM availability check (pass/fail)
- Test tensor bounds validation (security)
- Test architecture validation (llama only)
- Test error message clarity

### Integration Tests
- Test full validation with valid Qwen2.5-0.5B GGUF
- Test validation failure with insufficient VRAM
- Test validation failure with unsupported architecture
- Test validation failure with malformed GGUF (security)
- Test validation report structure

### Manual Verification
1. Validate Qwen2.5-0.5B GGUF (should pass)
2. Validate with insufficient VRAM (should fail with clear message)
3. Validate GPT GGUF (should fail: unsupported architecture)
4. Validate malicious GGUF (should fail: security)
5. Check audit log for rejected files

---

## Definition of Done

- [ ] All acceptance criteria met
- [ ] Code reviewed
- [ ] Unit tests passing (8+ tests)
- [ ] Integration tests passing
- [ ] Security validation tests passing
- [ ] Documentation updated
- [ ] Story marked complete in day-tracker.md

---

## References

- Spec: `bin/.specs/01_M0_worker_orcd.md` Section 6.2 (Model Validation)
- Security Alert: `bin/worker-orcd/.security/SECURITY_ALERT_GGUF_PARSING.md`
- Related Stories: LT-001, LT-002, LT-023, LT-030

---

**Status**: Ready for execution  
**Owner**: Llama-Beta  
**Created**: 2025-10-04

---

Detailed by Project Management Team — ready to implement 📋

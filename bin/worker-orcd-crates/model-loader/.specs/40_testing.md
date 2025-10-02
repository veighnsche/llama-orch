# Model Loader — Testing Specification

**Status**: Draft  
**Security Tier**: TIER 1 (Critical)  
**Last Updated**: 2025-10-02

---

## 0. Executive Summary

**Key Finding**: Property testing is CRITICAL for parser security.

**Why Property Testing Matters for model-loader**:
- Parses untrusted binary data (GGUF files from network/disk)
- Single malformed byte can trigger buffer overflow → RCE
- Traditional unit tests miss edge cases (only test known scenarios)
- Property tests find parser bugs via randomized fuzzing
- **Security requirement**: Parser MUST NOT panic on ANY input

**Testing Strategy**: Multi-layered defense (unit + property + BDD + fuzz)

---

## 1. Testing Architecture

### 1.1 Why This Crate Needs Extensive Testing

**Risk Profile**:
- **Untrusted input**: Model files from network, disk, or compromised pool-managerd
- **Binary format parsing**: GGUF headers with variable-length fields
- **Memory safety boundary**: Parser bugs → buffer overflow → RCE
- **Security-critical**: First validation gate before VRAM loading

**Threat Model**:
- Attacker crafts malicious GGUF file
- Parser has buffer overflow bug
- Malicious payload triggers RCE on worker

**Testing Goal**: Prove parser is robust against ALL inputs (not just valid ones).

---

### 1.2 Four-Layer Testing Strategy

| Layer | Purpose | Coverage | Tools |
|-------|---------|----------|-------|
| **Unit Tests** | Known valid/invalid inputs | Basic paths | `cargo test` |
| **Property Tests** | Parser invariants hold for ALL inputs | Edge cases, fuzzing | `proptest` |
| **BDD Tests** | Observable behaviors | Integration scenarios | `cucumber` |
| **Fuzz Tests** | Crash detection | Random mutation | `cargo-fuzz` (future) |

**Why all four?**
- Unit: Fast feedback on known cases
- Property: Find unknown edge cases
- BDD: Verify end-to-end behaviors
- Fuzz: Deep mutation-based exploration

---

## 2. Property Testing Strategy

### 2.1 Why Property Testing is Critical

**Traditional Unit Test Problem**:
```rust
#[test]
fn test_valid_gguf() {
    let gguf = create_valid_gguf();
    assert!(loader.validate_bytes(&gguf, None).is_ok());
}

#[test]
fn test_invalid_magic() {
    let gguf = vec![0x00, 0x00, 0x00, 0x00];
    assert!(loader.validate_bytes(&gguf, None).is_err());
}

// Problem: Only tests 2 specific byte sequences!
// Misses: truncated files, oversized fields, malformed strings, etc.
```

**Property Test Solution**:
```rust
proptest! {
    #[test]
    fn parser_never_panics(bytes: Vec<u8>) {
        let loader = ModelLoader::new();
        let _ = loader.validate_bytes(&bytes, None);
        // Tests THOUSANDS of random byte sequences
        // Finds edge cases unit tests miss
    }
}
```

---

### 2.2 Critical Properties to Test

#### Property 1: Parser Never Panics

**Property**: `∀ bytes, validate_bytes(bytes) returns Result (never panics)`

```rust
proptest! {
    #[test]
    fn parser_never_panics_on_any_input(bytes: Vec<u8>) {
        let loader = ModelLoader::new();
        let _ = loader.validate_bytes(&bytes, None);
        // If this panics, we found a security bug
    }
}
```

**Why critical**: Panic → DoS or exploitable crash

---

#### Property 2: Valid GGUF Always Accepted

**Property**: `∀ valid_gguf, validate_bytes(valid_gguf) = Ok`

```rust
// Strategy: Generate valid GGUF files
fn valid_gguf_strategy() -> impl Strategy<Value = Vec<u8>> {
    (1usize..100, 0usize..10).prop_map(|(tensor_count, kv_count)| {
        create_valid_gguf(tensor_count, kv_count)
    })
}

proptest! {
    #[test]
    fn valid_gguf_always_accepted(gguf in valid_gguf_strategy()) {
        let loader = ModelLoader::new();
        assert!(loader.validate_bytes(&gguf, None).is_ok());
    }
}
```

**Why critical**: False negatives break functionality

---

#### Property 3: Bounds Checks Always Hold

**Property**: `∀ offset, length, read(offset, length) checks bounds`

```rust
proptest! {
    #[test]
    fn read_u32_never_overflows(
        bytes in prop::collection::vec(any::<u8>(), 0..1000),
        offset in 0usize..1000,
    ) {
        let parser = GgufParser::new(&bytes);
        let result = parser.read_u32(offset);
        
        // If offset + 4 > len, MUST return error
        if offset.saturating_add(4) > bytes.len() {
            assert!(result.is_err());
        }
        // Never panic
    }
}
```

**Why critical**: Bounds check failure → buffer overflow

---

#### Property 4: String Length Validation

**Property**: `∀ string_len, string_len > MAX ⇒ reject`

```rust
proptest! {
    #[test]
    fn oversized_strings_rejected(
        string_len in (MAX_STRING_LEN + 1)..1_000_000usize
    ) {
        let mut gguf = valid_gguf_header();
        gguf.extend_from_slice(&(string_len as u32).to_le_bytes());
        
        let loader = ModelLoader::new();
        let result = loader.validate_bytes(&gguf, None);
        
        assert!(matches!(result, Err(LoadError::StringTooLong { .. })));
    }
}
```

**Why critical**: Unbounded allocation → OOM DoS

---

#### Property 5: Tensor Count Limits

**Property**: `∀ count, count > MAX_TENSORS ⇒ reject`

```rust
proptest! {
    #[test]
    fn excessive_tensor_count_rejected(
        count in (MAX_TENSORS + 1)..100_000usize
    ) {
        let gguf = create_gguf_with_tensor_count(count);
        let loader = ModelLoader::new();
        let result = loader.validate_bytes(&gguf, None);
        
        assert!(matches!(result, Err(LoadError::TensorCountExceeded { .. })));
    }
}
```

**Why critical**: Resource exhaustion → DoS

---

#### Property 6: Hash Verification Correctness

**Property**: `∀ bytes, hash(bytes) = recompute(hash(bytes))`

```rust
proptest! {
    #[test]
    fn hash_verification_is_deterministic(bytes: Vec<u8>) {
        let hash1 = compute_hash(&bytes);
        let hash2 = compute_hash(&bytes);
        assert_eq!(hash1, hash2);
    }
    
    #[test]
    fn hash_verification_detects_tampering(
        mut bytes: Vec<u8>,
        flip_bit_idx in 0usize..10000,
    ) {
        if bytes.is_empty() {
            return Ok(());
        }
        
        let original_hash = compute_hash(&bytes);
        
        // Flip one bit
        let byte_idx = flip_bit_idx % bytes.len();
        bytes[byte_idx] ^= 0x01;
        
        let tampered_hash = compute_hash(&bytes);
        
        // Hashes MUST differ (collision would be a bug)
        assert_ne!(original_hash, tampered_hash);
    }
}
```

**Why critical**: Hash collision → integrity bypass

---

### 2.3 Property Test Configuration

```rust
// tests/property_tests.rs

use proptest::prelude::*;
use model_loader::*;

// Configure proptest
proptest! {
    #![proptest_config(ProptestConfig {
        cases: 1000,        // Run 1000 random tests per property
        max_shrink_iters: 10000,  // Shrink failing cases
        ..ProptestConfig::default()
    })]
    
    // Tests here...
}
```

**Performance**:
- 1000 cases per property = good coverage
- Takes ~5-10 seconds per property
- Runs in CI on every commit

---

## 3. Unit Testing Strategy

### 3.1 Required Unit Test Coverage

**Core Validation**:
- ✅ Valid GGUF file (magic, version, header)
- ✅ Invalid magic number
- ✅ Truncated file (< MIN_HEADER_SIZE)
- ✅ Oversized file (> max_size)
- ✅ Hash verification (correct, mismatch)
- ✅ Bounds checking (offset + length > size)

**Error Handling**:
- ✅ File not found (Io error)
- ✅ Permission denied (Io error)
- ✅ Hash format validation
- ✅ Path validation (traversal, symlinks)

**Edge Cases**:
- ✅ Empty file (0 bytes)
- ✅ Minimum valid file (12 bytes)
- ✅ Maximum string length (exactly MAX_STRING_LEN)
- ✅ Maximum tensor count (exactly MAX_TENSORS)

---

### 3.2 Security-Focused Unit Tests

```rust
#[test]
fn test_buffer_overflow_protection() {
    // Attempt to read past end of buffer
    let bytes = vec![0u8; 10];
    let parser = GgufParser::new(&bytes);
    
    let result = parser.read_u32(7);  // Would read bytes 7..11 (overflow!)
    
    assert!(matches!(result, Err(LoadError::BufferOverflow { .. })));
}

#[test]
fn test_integer_overflow_in_tensor_dims() {
    // Craft tensor with dims that overflow when multiplied
    let mut gguf = valid_gguf_header();
    // dims = [2^32, 2^32] → overflow
    gguf.extend_from_slice(&[0xFF, 0xFF, 0xFF, 0xFF]);
    gguf.extend_from_slice(&[0xFF, 0xFF, 0xFF, 0xFF]);
    
    let loader = ModelLoader::new();
    let result = loader.validate_bytes(&gguf, None);
    
    assert!(matches!(result, Err(LoadError::InvalidFormat(_))));
}

#[test]
fn test_path_traversal_rejected() {
    let loader = ModelLoader::new();
    let result = loader.load_and_validate(LoadRequest::new(
        Path::new("../../../../etc/passwd")
    ));
    
    // Should fail (exact error depends on input-validation integration)
    assert!(result.is_err());
}
```

---

## 4. BDD Testing Strategy

### 4.1 Observable Behaviors

**What BDD tests verify**:
- End-to-end workflows (not implementation details)
- Integration with filesystem, temp files
- Error messages are user-friendly
- Validation happens in correct order

**Example scenarios**:
```gherkin
Scenario: Load model with correct hash
  Given a GGUF model file with known hash
  When I load the model with hash verification
  Then the model loads successfully
  And the loaded bytes match the file contents

Scenario: Reject malformed GGUF
  Given a file with invalid magic number
  When I load and validate the model
  Then the load fails with invalid format error
```

**See**: `bdd/tests/features/` for complete scenarios

---

### 4.2 BDD vs Unit Tests

| Aspect | Unit Tests | BDD Tests |
|--------|-----------|-----------|
| **Focus** | Functions, methods | User-facing behaviors |
| **Language** | Rust code | Gherkin (Given/When/Then) |
| **Integration** | Isolated components | Full integration |
| **Audience** | Developers | Stakeholders, auditors |
| **Speed** | Very fast | Slower (I/O, setup) |

**Both are needed**: Unit tests for coverage, BDD for behavior verification.

---

## 5. Fuzzing Strategy (Future)

### 5.1 Why Fuzzing Matters

**Fuzzing** = Mutation-based randomized testing

**Difference from property testing**:
- Property tests: Generate inputs from scratch
- Fuzzing: Start with valid input, mutate randomly
- Fuzzing finds deeper bugs (coverage-guided)

---

### 5.2 Fuzz Targets

```rust
// fuzz/fuzz_targets/gguf_parser.rs

#![no_main]
use libfuzzer_sys::fuzz_target;
use model_loader::ModelLoader;

fuzz_target!(|data: &[u8]| {
    let loader = ModelLoader::new();
    let _ = loader.validate_bytes(data, None);
    // Fuzzer will try MILLIONS of mutations
    // Looking for panics, crashes, hangs
});
```

**Fuzzing goals**:
- Run for 24+ hours
- Find edge cases property tests miss
- Coverage-guided (explores all code paths)

**Status**: Post-M0 (requires `cargo-fuzz` setup)

---

## 6. Test Organization

### 6.1 File Structure

```
model-loader/
├── src/
│   └── (implementation)
├── tests/
│   ├── property_tests.rs        # Property-based tests
│   ├── unit_tests.rs            # Unit tests
│   └── integration_tests.rs     # Integration tests
├── bdd/
│   ├── src/steps/               # BDD step definitions
│   └── tests/features/          # Cucumber scenarios
└── fuzz/                        # Fuzz targets (future)
    └── fuzz_targets/
        └── gguf_parser.rs
```

---

### 6.2 Test Execution

```bash
# Unit tests (fast, runs every commit)
cargo test -p model-loader

# Property tests (runs in CI, ~30 seconds)
cargo test -p model-loader --test property_tests

# BDD tests (runs in CI, ~10 seconds)
cargo run -p model-loader-bdd

# Fuzz tests (runs in dedicated fuzz runs, 24+ hours)
cargo fuzz run gguf_parser  # Future
```

---

## 7. Test Coverage Requirements

### 7.1 Coverage Targets

| Category | Target | Rationale |
|----------|--------|-----------|
| **Line coverage** | ≥ 90% | Most code paths tested |
| **Branch coverage** | ≥ 85% | Most error paths tested |
| **Property tests** | 5+ properties | Core invariants verified |
| **BDD scenarios** | 10+ scenarios | Observable behaviors covered |

---

### 7.2 Critical Paths MUST Be Tested

**Security-critical code paths** (100% coverage required):
- ✅ Bounds checking in all read functions
- ✅ Hash verification logic
- ✅ Path validation logic
- ✅ String length validation
- ✅ Tensor count validation
- ✅ Integer overflow checks

**Test verification**:
```bash
# Generate coverage report
cargo tarpaulin -p model-loader --out Html

# Check security-critical paths
# (Manual review of coverage report)
```

---

## 8. Security Test Requirements

### 8.1 Required Security Tests

**Per 20_security.md**, these MUST be tested:

#### Buffer Overflow Tests
```rust
#[test]
fn test_gguf_001_buffer_overflow() {
    // Attempt to trigger buffer overflow in parser
    let malicious = craft_oversized_string_header();
    assert!(loader.validate_bytes(&malicious, None).is_err());
}
```

#### Path Traversal Tests
```rust
#[test]
fn test_path_001_traversal() {
    assert!(loader.load("../../../etc/passwd").is_err());
}
```

#### Resource Exhaustion Tests
```rust
#[test]
fn test_limit_001_tensor_count() {
    let malicious = craft_gguf_with_huge_tensor_count();
    assert!(matches!(
        loader.validate_bytes(&malicious, None),
        Err(LoadError::TensorCountExceeded { .. })
    ));
}
```

**See**: `20_security.md` for complete vulnerability list

---

### 8.2 Negative Testing

**Principle**: Test what the parser REJECTS (not just what it accepts)

```rust
#[test]
fn test_rejects_all_invalid_magic_numbers() {
    for magic in 0x00000000..=0xFFFFFFFF {
        if magic == 0x46554747 {
            continue;  // Skip valid magic
        }
        
        let mut bytes = magic.to_le_bytes().to_vec();
        bytes.extend_from_slice(&[0; 8]);  // Padding
        
        let result = loader.validate_bytes(&bytes, None);
        assert!(result.is_err(), "Should reject magic: 0x{:x}", magic);
    }
}
```

---

## 9. Continuous Testing

### 9.1 CI Pipeline

```yaml
# .github/workflows/model-loader-tests.yml

name: model-loader Tests
on: [push, pull_request]

jobs:
  unit-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions-rs/toolchain@v1
        with:
          toolchain: stable
      - name: Run unit tests
        run: cargo test -p model-loader
  
  property-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Run property tests
        run: cargo test -p model-loader --test property_tests
  
  bdd-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Run BDD tests
        run: cargo run -p model-loader-bdd
  
  coverage:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Generate coverage
        run: cargo tarpaulin -p model-loader --out Xml
      - name: Upload to codecov
        uses: codecov/codecov-action@v3
```

---

### 9.2 Pre-commit Hooks

```bash
# .git/hooks/pre-commit

#!/bin/bash
# Run fast tests before commit

echo "Running unit tests..."
cargo test -p model-loader || exit 1

echo "Running property tests (1000 cases)..."
cargo test -p model-loader --test property_tests || exit 1

echo "✅ All tests passed"
```

---

## 10. Test-Driven Development Workflow

### 10.1 Red-Green-Refactor for Parser

**Step 1: Write failing test**
```rust
#[test]
fn test_parse_gguf_version_3() {
    let gguf = create_gguf_v3();
    assert!(loader.validate_bytes(&gguf, None).is_ok());
}
// ❌ FAILS (parser not implemented yet)
```

**Step 2: Implement minimal code**
```rust
fn validate_gguf(&self, bytes: &[u8]) -> Result<()> {
    let version = read_u32(bytes, 4)?;
    if version != 3 {
        return Err(LoadError::InvalidFormat("Unsupported version".into()));
    }
    Ok(())
}
// ✅ PASSES
```

**Step 3: Refactor**
```rust
const SUPPORTED_VERSIONS: &[u32] = &[2, 3];

fn validate_gguf(&self, bytes: &[u8]) -> Result<()> {
    let version = read_u32(bytes, 4)?;
    if !SUPPORTED_VERSIONS.contains(&version) {
        return Err(LoadError::InvalidFormat(
            format!("Unsupported version: {}", version)
        ));
    }
    Ok(())
}
// ✅ STILL PASSES (better code)
```

---

### 10.2 Property-First Development

**Workflow**:
1. Write property test FIRST (defines invariant)
2. Implement code to satisfy property
3. Property test continuously validates invariant

**Example**:
```rust
// 1. Define property FIRST
proptest! {
    #[test]
    fn read_never_overflows(bytes: Vec<u8>, offset: usize) {
        let result = read_u32(&bytes, offset);
        // Never panics (property to uphold)
    }
}
// ❌ FAILS (read_u32 doesn't exist)

// 2. Implement with bounds checking
fn read_u32(bytes: &[u8], offset: usize) -> Result<u32> {
    if offset.saturating_add(4) > bytes.len() {
        return Err(LoadError::BufferOverflow { ... });
    }
    Ok(u32::from_le_bytes([...]))
}
// ✅ PASSES (property holds)
```

---

## 11. Refinement Opportunities

### 11.1 Advanced Testing Techniques

**M0+ Enhancements**:

1. **Mutation Testing**
   - Use `cargo-mutants` to verify test quality
   - Kill mutants with property tests
   - Target: 90%+ mutation score

2. **Coverage-Guided Fuzzing**
   - Integrate `cargo-fuzz` with libFuzzer
   - Run 24+ hour fuzz campaigns
   - Target: 100% code coverage via fuzzing

3. **Symbolic Execution**
   - Use KLEE or angr for path exploration
   - Find corner cases property tests miss
   - Prove bounds checks are exhaustive

4. **Differential Testing**
   - Compare against reference GGUF parser
   - Validate identical behavior on same inputs
   - Find spec compliance issues

---

### 11.2 Performance Testing

**Benchmark suite**:
```rust
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn bench_hash_verification(c: &mut Criterion) {
    let model = create_10gb_model();
    
    c.bench_function("hash_10gb_model", |b| {
        b.iter(|| compute_hash(black_box(&model)));
    });
}

criterion_group!(benches, bench_hash_verification);
criterion_main!(benches);
```

**Target**: Hash verification < 1s per GB

---

### 11.3 Integration with Proof Bundles

**Future**: Generate test artifacts for auditing
- Property test execution logs
- Coverage reports
- Fuzz corpus (interesting inputs found)
- Mutation testing results

---

## 12. Summary

### 12.1 Why This Testing Strategy Matters

**For Security**:
- Property testing finds bugs unit tests miss
- Fuzzing explores deep corner cases
- 100% coverage of security-critical paths

**For Reliability**:
- BDD verifies end-to-end behaviors
- Regression tests prevent bugs from returning
- CI catches issues before merge

**For Maintainability**:
- Tests document expected behavior
- Property tests are self-documenting invariants
- Safe refactoring (tests catch regressions)

---

### 12.2 Testing Checklist

**Before M0 release**:
- ✅ Unit tests for all public APIs
- ✅ Property tests for parser robustness (5+ properties)
- ✅ BDD tests for observable behaviors (10+ scenarios)
- ✅ Security tests for all vulnerabilities (20_security.md)
- ✅ ≥90% line coverage
- ✅ CI pipeline runs all tests
- ⬜ Fuzzing (post-M0)
- ⬜ Mutation testing (post-M0)

---

**Property testing is NOT optional for this crate** — it's a security requirement for any parser of untrusted binary data.

---

**End of Testing Specification**

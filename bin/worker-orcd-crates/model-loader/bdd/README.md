# model-loader BDD Tests

**Behavior-Driven Development tests for model-loader validation**

## Running Tests

```bash
# Run all BDD tests
cd bin/worker-orcd-crates/model-loader/bdd
cargo run --bin bdd-runner

# Run specific feature
cargo run --bin bdd-runner -- tests/features/hash_verification.feature

# Run with tags
cargo run --bin bdd-runner -- --tags "not @skip"
```

## Test Coverage

### ✅ Hash Verification (`hash_verification.feature`)
- Load model with correct hash → Success
- Load model with wrong hash → HashMismatch error
- Load model without hash → Success (optional verification)

### ✅ GGUF Format Validation (`gguf_validation.feature`)
- Load valid GGUF file → Success
- Reject invalid magic number → InvalidFormat error
- Validate valid bytes in memory → Success
- Reject invalid bytes in memory → InvalidFormat error

### ✅ Resource Limits (`resource_limits.feature`)
- Reject file exceeding max size → TooLarge error
- TODO: Reject excessive tensor count
- TODO: Reject oversized strings

### 🚧 Path Security (`path_security.feature`)
- TODO: Reject path traversal sequence (blocked on input-validation)
- TODO: Reject symlink escape
- TODO: Reject null byte in path

## Test Organization

```
bdd/
├── src/
│   ├── main.rs              # BDD runner entry point
│   └── steps/
│       ├── mod.rs           # Step module exports
│       ├── given.rs         # Setup fixtures
│       ├── when.rs          # Execute actions
│       └── then.rs          # Verify outcomes
└── tests/
    └── features/
        ├── hash_verification.feature
        ├── gguf_validation.feature
        ├── resource_limits.feature
        └── path_security.feature
```

## Adding New Tests

1. **Create feature file** in `tests/features/`
2. **Write scenarios** in Gherkin syntax
3. **Implement steps** in `src/steps/` (given/when/then)
4. **Run tests** to verify

## Security Testing

All security requirements from `20_security.md` are tested:
- ✅ HASH-001 to HASH-007: Hash verification
- ✅ GGUF-010: Magic number validation
- ✅ LIMIT-001: File size limits
- 🚧 PATH-001 to PATH-008: Path security (pending input-validation)
- 🚧 GGUF-001 to GGUF-012: Full GGUF validation (M0 work)

## CI Integration

```yaml
# .github/workflows/model-loader-bdd.yml
name: model-loader BDD Tests
on: [push, pull_request]
jobs:
  bdd:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions-rs/toolchain@v1
        with:
          toolchain: stable
      - name: Run BDD tests
        run: |
          cd bin/worker-orcd-crates/model-loader/bdd
          cargo run --bin bdd-runner
```

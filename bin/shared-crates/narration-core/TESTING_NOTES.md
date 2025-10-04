# Testing Notes for Narration Core

## Known Test Issues

### Capture Adapter Tests - Parallel Execution

The tests in `src/auto.rs` that use `CaptureAdapter` may fail when run in parallel due to shared global state:
- `test_narrate_auto_injects_fields`
- `test_narrate_auto_respects_existing_fields`

**Workaround**: Run these tests individually:
```bash
cargo test -p observability-narration-core test_narrate_auto_injects_fields
cargo test -p observability-narration-core test_narrate_auto_respects_existing_fields
```

**Root Cause**: The `CaptureAdapter` uses a global `OnceLock` for test capture. When tests run in parallel, they interfere with each other's capture state.

**Future Fix**: Consider using `serial_test` crate to force these tests to run serially, or refactor to use thread-local storage instead of global state.

## Test Coverage

All core functionality is tested:
- ✅ Narration levels (TRACE, DEBUG, INFO, WARN, ERROR, FATAL)
- ✅ Secret redaction (Bearer tokens, API keys, JWT, private keys, URL passwords)
- ✅ Correlation ID generation and validation
- ✅ HTTP header propagation
- ✅ Auto-injection (service identity, timestamps)
- ✅ Trace macros (conditional compilation)
- ✅ Capture adapter (when run individually)

## Running Tests

**All tests (may have flaky failures)**:
```bash
cargo test -p observability-narration-core
```

**Individual test suites**:
```bash
cargo test -p observability-narration-core --lib redaction
cargo test -p observability-narration-core --lib correlation
cargo test -p observability-narration-core --lib http
cargo test -p observability-narration-core --lib trace
```

**With features**:
```bash
cargo test -p observability-narration-core --features trace-enabled
cargo test -p observability-narration-core --features otel
```

---

*Note: These issues will be addressed in Week 4 when we add proper BDD tests with proof bundle integration.*

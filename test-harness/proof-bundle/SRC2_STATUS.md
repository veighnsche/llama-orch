# src2/ Implementation Status

**Started**: 2025-10-02 16:03  
**Status**: 🟢 IN PROGRESS  
**Goal**: Clean V3 architecture that actually works

---

## Why src2/?

**The brutal truth**: `src/` is too broken to fix incrementally.

### Problems with src/
1. ❌ V1 + V2 mixed together (incompatible designs)
2. ❌ Parser reads wrong output stream (stdout vs stderr)
3. ❌ Metadata annotations completely lost (just comments)
4. ❌ Silent failures everywhere (returns success with 0 tests)
5. ❌ Formatters generate contradictory garbage
6. ❌ No validation, no error handling
7. ❌ Fragile regex parsing that breaks on warnings

**Decision**: Start fresh. Build it right.

---

## V3 Architecture (src2/)

### Core Principles

1. **Use Cargo's Tools** - `cargo test --format json` (stable, structured)
2. **Extract Metadata** - Parse source with `syn`, build index
3. **Fail Fast** - Clear errors, no silent failures
4. **Validate Everything** - Never generate garbage reports
5. **Zero Boilerplate** - One-liner API that works

### Module Structure

```
src2/
├── lib.rs                 ✅ DONE - Public API, re-exports
├── core/
│   ├── mod.rs            ✅ DONE - Module organization
│   ├── error.rs          ✅ DONE - Proper error types
│   ├── metadata.rs       ⏳ TODO - TestMetadata (copy from src/)
│   ├── mode.rs           ⏳ TODO - Mode enum
│   ├── status.rs         ⏳ TODO - TestStatus enum
│   ├── result.rs         ⏳ TODO - TestResult struct
│   └── summary.rs        ⏳ TODO - TestSummary struct
├── extraction/
│   ├── mod.rs            ⏳ TODO - Metadata extraction
│   ├── parser.rs         ⏳ TODO - Parse Rust source (syn)
│   ├── annotations.rs    ⏳ TODO - Parse @key: value
│   └── index.rs          ⏳ TODO - Build metadata index
├── runners/
│   ├── mod.rs            ⏳ TODO - Test execution
│   ├── cargo_json.rs     ⏳ TODO - cargo test --format json
│   └── subprocess.rs     ⏳ TODO - Run cargo as subprocess
├── formatters/
│   ├── mod.rs            ⏳ TODO - Report generation
│   ├── executive.rs      ⏳ TODO - Copy from src/, add validation
│   ├── developer.rs      ⏳ TODO - Copy from src/, add validation
│   ├── failure.rs        ⏳ TODO - Copy from src/, add validation
│   └── metadata.rs       ⏳ TODO - Copy from src/, add validation
├── writers/
│   ├── mod.rs            ⏳ TODO - File I/O
│   ├── bundle.rs         ⏳ TODO - ProofBundle directory
│   ├── ndjson.rs         ⏳ TODO - NDJSON streaming
│   └── markdown.rs       ⏳ TODO - Markdown with headers
└── api/
    ├── mod.rs            ⏳ TODO - Public API
    ├── generate.rs       ⏳ TODO - generate_for_crate()
    └── builder.rs        ⏳ TODO - Fluent builder
```

---

## Implementation Plan

### Phase 1: Core Types (2 hours)

**Goal**: Stable foundation

- [x] Create src2/ directory structure
- [x] Implement error types
- [ ] Copy TestMetadata from src/ (it's fine)
- [ ] Copy TestStatus from src/ (it's fine)
- [ ] Copy TestResult from src/ (it's fine)
- [ ] Copy TestSummary from src/ (it's fine)
- [ ] Implement Mode enum (cleaner than old TestType)

**Output**: Core types that work

### Phase 2: JSON Runner (3 hours)

**Goal**: Actually capture tests correctly

- [ ] Implement cargo_json.rs
  - Parse `cargo test --format json` output
  - Handle test events (started, ok, failed)
  - Extract timing, stdout, stderr per test
  - Build TestSummary from events
- [ ] Add validation
  - Fail if 0 tests found
  - Fail if cargo test exits non-zero
  - Clear error messages
- [ ] Unit tests
  - Parse real cargo JSON output
  - Handle edge cases (all pass, all fail, mixed)

**Output**: Reliable test execution

### Phase 3: Metadata Extraction (4 hours)

**Goal**: Actually use @annotations

- [ ] Implement parser.rs
  - Use `syn` to parse Rust source
  - Find #[test] functions
  - Extract doc comments
- [ ] Implement annotations.rs
  - Parse @priority: critical
  - Parse @spec: ORCH-1234
  - Parse @team, @owner, @tags, etc.
- [ ] Implement index.rs
  - Build HashMap<TestName, TestMetadata>
  - Save to metadata_index.json
  - Load at runtime
- [ ] Unit tests
  - Parse various annotation formats
  - Handle malformed annotations
  - Build complete index

**Output**: Metadata actually extracted!

### Phase 4: Formatters (2 hours)

**Goal**: Beautiful reports with validation

- [ ] Copy formatters from src/
- [ ] Add validation
  - Reject empty summaries
  - Handle division by zero
  - No contradictory messages
- [ ] Enhance with metadata
  - Group by priority
  - Show spec references
  - Highlight critical failures
- [ ] Unit tests
  - Generate reports for various scenarios
  - Verify no panics
  - Check for contradictions

**Output**: Reports that make sense

### Phase 5: Writers (1 hour)

**Goal**: Clean file I/O

- [ ] Implement bundle.rs
  - Create proof bundle directories
  - Manage timestamps
  - Clean API
- [ ] Implement ndjson.rs
  - Stream test results
  - Add metadata headers
- [ ] Implement markdown.rs
  - Add autogenerated headers
  - Format nicely

**Output**: Files written correctly

### Phase 6: Public API (2 hours)

**Goal**: One-liner that works

- [ ] Implement generate.rs
  - `generate_for_crate(package, mode)`
  - Run cargo test --format json
  - Extract metadata
  - Merge results
  - Generate reports
  - Write files
  - Return summary
- [ ] Implement builder.rs
  - Fluent API for advanced usage
  - Optional metadata extraction
  - Custom output directory
- [ ] Integration tests
  - Test on proof-bundle itself
  - Verify 43 tests captured
  - Verify metadata extracted
  - Verify reports generated

**Output**: V3 API that works!

---

## Timeline

### Day 1 (Today)

- [x] Create src2/ structure ✅
- [x] Implement error types ✅
- [ ] Implement core types (2h)
- [ ] Implement JSON runner (3h)
- [ ] Basic integration test

**Goal**: Can capture tests correctly

### Day 2 (Tomorrow)

- [ ] Implement metadata extraction (4h)
- [ ] Implement formatters (2h)
- [ ] Implement writers (1h)

**Goal**: Full pipeline works

### Day 3

- [ ] Implement public API (2h)
- [ ] Comprehensive tests (3h)
- [ ] Documentation (2h)

**Goal**: Production ready

### Day 4

- [ ] Dogfood on proof-bundle (1h)
- [ ] Fix any issues (2h)
- [ ] Migration guide (2h)
- [ ] Announce V3 (1h)

**Goal**: V3 is default

---

## Success Criteria

### Must Work

1. ✅ **Capture all tests** - 43/43 from proof-bundle
2. ✅ **Extract metadata** - All @annotations from source
3. ✅ **Generate reports** - Beautiful, accurate, useful
4. ✅ **Fail fast** - Clear errors, no silent failures
5. ✅ **One-liner API** - Works first time

### Metrics

- **Test capture rate**: 100% (currently 0%)
- **Metadata extraction**: 100% (currently 0%)
- **Silent failures**: 0 (currently 100%)
- **Contradictory reports**: 0 (currently 100%)
- **User satisfaction**: High (currently none)

---

## Migration from src/

### What to Keep

✅ **Core types** - TestResult, TestSummary, TestStatus, TestMetadata
- These are fine, just copy them

✅ **Formatters** - Executive, developer, failure, metadata
- Good structure, just add validation

✅ **File I/O** - NDJSON, markdown, JSON
- Works fine, just clean up API

### What to Rewrite

❌ **Parsers** - Regex-based text parsing
- Replace with JSON parsing

❌ **API** - Mixed V1/V2, broken
- Clean V3 API

❌ **Metadata** - Lost in comments
- Actually extract from source

### What to Delete

🗑️ **V1 API** - `ProofBundle::for_type()`
- Too much boilerplate

🗑️ **Legacy types** - `TestType`, `LegacyTestType`
- Confusing names

🗑️ **Stable parser** - Regex-based
- Fragile, broken

---

## Breaking Changes

### API

**Old**:
```rust
// V1 - too much boilerplate
let pb = ProofBundle::for_type(TestType::Unit)?;
pb.write_markdown("report", "...");

// V2 - broken
api::generate_for_crate("pkg", Mode::UnitFast)?;
```

**New**:
```rust
// V3 - clean, works
proof_bundle::generate_for_crate("pkg", Mode::UnitFast)?;
```

### Types

**Old**: `TestType`, `LegacyTestType`  
**New**: `Mode`

**Old**: `ProofBundle::for_type()`  
**New**: `generate_for_crate()`

### Modules

**Old**: Everything in `src/`  
**New**: Clean separation in `src2/`

---

## Dependencies

### Core (always)

```toml
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
chrono = "0.4"
```

### Metadata extraction (default feature)

```toml
syn = { version = "2.0", features = ["full", "parsing"] }
quote = "1.0"
glob = "0.3"
```

### Optional features

```toml
[features]
default = ["metadata-extraction"]
metadata-extraction = ["syn", "quote", "glob"]
```

---

## Testing Strategy

### Unit Tests

- ✅ Core types - Serialization, validation
- ✅ JSON parser - Parse real cargo output
- ✅ Metadata extractor - Parse various formats
- ✅ Formatters - Generate reports, no panics
- ✅ Error handling - All error paths

### Integration Tests

- ✅ Dogfood - Generate proof bundle for proof-bundle
- ✅ Empty crate - Handle 0 tests gracefully
- ✅ All failing - Handle 100% failure
- ✅ Metadata - Verify extraction works

### Property Tests

- ✅ Pass rate - Always 0-100%
- ✅ Reports - Never panic
- ✅ Validation - Catch all invalid data

---

## Documentation

### User Docs

- [ ] README.md - Quick start
- [ ] ARCHITECTURE_V3.md - Design
- [ ] MIGRATION.md - From src/
- [ ] API.md - Complete reference

### Developer Docs

- [ ] CONTRIBUTING.md - How to contribute
- [ ] TESTING.md - Testing strategy
- [ ] Inline docs - Every public item

---

## Next Steps

**Right now**:
1. Implement core types (copy from src/)
2. Implement JSON runner (new)
3. Test on proof-bundle itself

**Tomorrow**:
1. Implement metadata extraction
2. Implement formatters with validation
3. End-to-end integration test

**This week**:
1. Complete V3 implementation
2. Comprehensive testing
3. Documentation
4. Announce V3

---

## Commit to Quality

### No Compromises

- ✅ **No silent failures** - Fail fast with clear errors
- ✅ **No contradictions** - Reports must make sense
- ✅ **No fragile parsing** - Use stable APIs
- ✅ **No lost metadata** - Extract from source
- ✅ **No boilerplate** - One-liner API

### High Standards

- ✅ **100% test coverage** - Every function tested
- ✅ **100% documentation** - Every public item documented
- ✅ **100% validation** - Every input validated
- ✅ **100% reliability** - Works every time

**This is the last chance to get it right. Let's not waste it.** 🎯

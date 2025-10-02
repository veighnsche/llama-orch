# src2/ - Clean V3 Implementation

**Status**: 🟢 ACTIVE DEVELOPMENT  
**Version**: v0.3.0 (V3 Architecture)  
**Started**: 2025-10-02

---

## Why src2/?

The original `src/` has accumulated too many design flaws:

1. ❌ **V1 + V2 mixed together** - Two incompatible APIs in one codebase
2. ❌ **Broken parser** - Reads wrong output stream, fragile regex
3. ❌ **No metadata extraction** - Annotations are just comments
4. ❌ **Silent failures** - Returns success with 0 tests
5. ❌ **Contradictory formatters** - Generate garbage for empty data

**Decision**: Start fresh rather than fix a broken foundation.

---

## V3 Architecture Principles

### 1. Discover Tests Properly

```
❌ OLD: Guess where test files are
✅ NEW: Use cargo_metadata to find all test targets
```

### 2. Extract Metadata from Source

```
❌ OLD: Metadata annotations are lost (just doc comments)
✅ NEW: Parse source files with syn at runtime, cache results
```

### 3. Parse Output Correctly

```
❌ OLD: Parse stdout (wrong stream!), fragile regex
✅ NEW: Parse stderr (correct stream), validate results
```

### 4. Fail Fast

```
❌ OLD: Return success with 0 tests, generate garbage reports
✅ NEW: Validate everything, fail with clear errors
```

### 5. One-Liner API

```
❌ OLD: Manual API calls, easy to use wrong
✅ NEW: generate_for_crate("pkg", Mode::UnitFast) - works first time
```

---

## Module Structure

```
src2/
├── lib.rs                  # Public API, re-exports
├── core/
│   ├── mod.rs             # Module organization
│   ├── types.rs           # TestResult, TestSummary, TestStatus
│   ├── metadata.rs        # TestMetadata struct
│   └── error.rs           # ProofBundleError (thiserror)
├── discovery/              # NEW - Find tests using cargo_metadata
│   ├── mod.rs
│   ├── cargo_meta.rs      # Use cargo_metadata crate
│   └── targets.rs         # Find test targets and source paths
├── extraction/
│   ├── mod.rs             # Metadata extraction from source
│   ├── parser.rs          # Parse Rust source with syn (runtime)
│   ├── annotations.rs     # Parse @key: value syntax
│   └── cache.rs           # Cache metadata index for performance
├── runners/
│   ├── mod.rs             # Test execution
│   └── subprocess.rs      # cargo test as subprocess (parse stderr!)
├── formatters/
│   ├── mod.rs             # Report generation
│   ├── executive.rs       # Management summary (with validation)
│   ├── developer.rs       # Technical report (with validation)
│   ├── failure.rs         # Failure analysis (with validation)
│   └── metadata.rs        # Metadata report (with validation)
├── bundle/                 # Simplified I/O
│   ├── mod.rs
│   └── writer.rs          # Write proof bundle files (use std lib)
└── api/
    ├── mod.rs             # Public API
    └── generate.rs        # generate_for_crate() - the one-liner
```

---

## Design Decisions

### Decision 1: Use cargo_metadata for Test Discovery

**Rationale**: Only reliable way to find all test targets
- ✅ Knows about all packages in workspace
- ✅ Knows about all targets (lib, tests, benches)
- ✅ Provides source file paths
- ✅ Works for all project structures

**Implementation**:
```rust
use cargo_metadata::MetadataCommand;

let metadata = MetadataCommand::new().exec()?;
for package in metadata.packages {
    for target in package.targets {
        if target.kind.contains(&"test".into()) {
            // Found a test target!
        }
    }
}
```

### Decision 2: Runtime Metadata Extraction with Caching

**Rationale**: More flexible than build-time, sees all tests
- ✅ Simpler (no build.rs needed)
- ✅ Sees proc-macro generated tests
- ✅ Can be cached for performance
- ✅ Works in all environments

**Implementation**:
```rust
// Try to load cache first
if let Some(cached) = load_metadata_cache()? {
    return Ok(cached);
}

// Parse source files
let metadata = extract_from_source_files()?;

// Cache for next run
save_metadata_cache(&metadata)?;
```

### Decision 3: Parse stderr Not stdout

**Rationale**: cargo test writes test output to stderr, not stdout
- ✅ Works on stable Rust (no nightly features)
- ✅ Simple subprocess execution
- ✅ Reliable (just fix the stream!)

**Implementation**:
```rust
let output = Command::new("cargo")
    .args(&["test", "--package", package])
    .output()?;

// BUG FIX: Parse stderr, not stdout!
let test_output = String::from_utf8_lossy(&output.stderr);
let summary = parse_test_output(&test_output)?;
```

### Decision 4: Validation Everywhere

**Rationale**: Fail fast with clear errors, never silent failures.

**Implementation**:
```rust
// After parsing
if summary.total == 0 {
    return Err(ProofBundleError::NoTestsFound {
        package: package.to_string(),
        hint: "Check package name or test filters",
    });
}

// After metadata extraction
if metadata_index.is_empty() {
    warn!("No metadata found in source files");
}

// Before report generation
if summary.total == 0 {
    return Err(ProofBundleError::CannotGenerateReports {
        reason: "No tests to report on",
    });
}
```

---

## Migration from src/

### Phase 1: Core Types (Day 1)

Move stable types to `src2/core/`:
- ✅ `TestResult`
- ✅ `TestSummary`
- ✅ `TestStatus`
- ✅ `TestMetadata`

**No changes** - these are fine.

### Phase 2: Formatters (Day 1)

Move formatters to `src2/formatters/`:
- ✅ Executive summary
- ✅ Developer report
- ✅ Failure report
- ✅ Metadata report

**Changes**: Add validation for empty data.

### Phase 3: New Runners (Day 2)

Implement in `src2/runners/`:
- ✅ JSON parser (cargo test --format json)
- ✅ Metadata extractor (syn-based)
- ⏳ Custom harness (libtest-mimic) - optional

### Phase 4: New API (Day 2)

Implement in `src2/api/`:
- ✅ `generate_for_crate()` - works correctly
- ✅ Builder pattern for advanced usage
- ⏳ Proc macro - optional

### Phase 5: Deprecate src/ (v1.0.0)

- Mark `src/` as deprecated
- Point to `src2/` in docs
- Remove in v2.0.0

---

## Success Criteria

### Must Work

1. ✅ **Capture all tests** - 43/43 tests from proof-bundle itself
2. ✅ **Extract metadata** - All @annotations from source
3. ✅ **Generate reports** - Beautiful, accurate, useful
4. ✅ **Fail fast** - Clear errors, no silent failures
5. ✅ **One-liner API** - `generate_for_crate("pkg", Mode::UnitFast)?`

### Nice to Have

1. ⏳ **Real-time recording** - Custom test harness
2. ⏳ **Proc macro** - Zero-boilerplate usage
3. ⏳ **Parallel execution** - Fast proof bundle generation
4. ⏳ **Incremental updates** - Only re-run changed tests

---

## Timeline

### Week 1 (Now)

- [x] Create src2/ structure
- [ ] Implement core types
- [ ] Implement JSON parser
- [ ] Implement metadata extractor
- [ ] Implement new API
- [ ] Test with proof-bundle itself

**Goal**: V3 API works, captures all tests, extracts metadata

### Week 2

- [ ] Custom test harness (optional)
- [ ] Proc macro (optional)
- [ ] Performance optimization
- [ ] Comprehensive tests
- [ ] Documentation

**Goal**: Production-ready V3

### Week 3

- [ ] Migration guide
- [ ] Deprecate src/
- [ ] Update all examples
- [ ] Update all docs

**Goal**: V3 is default, src/ is legacy

---

## Breaking Changes

### API Changes

**Old (src/)**:
```rust
// V1
let pb = ProofBundle::for_type(TestType::Unit)?;
pb.write_markdown("report", "...");

// V2 (broken)
api::generate_for_crate("pkg", Mode::UnitFast)?;
```

**New (src2/)**:
```rust
// V3
proof_bundle::generate_for_crate("pkg", Mode::UnitFast)?;

// Or with builder
proof_bundle::Builder::new("pkg")
    .mode(Mode::UnitFast)
    .with_metadata_extraction()
    .generate()?;
```

### Type Changes

**Old**: `TestType` (legacy enum)  
**New**: `Mode` (clearer name)

**Old**: `ProofBundle::for_type()`  
**New**: `generate_for_crate()`

### Module Changes

**Old**: Everything in `src/`  
**New**: Clean separation in `src2/`

---

## Compatibility

### Rust Version

- **Minimum**: 1.70 (for `syn` and `serde`)
- **Recommended**: 1.75+

### Cargo Version

- **Minimum**: 1.70 (for `--format json`)
- **Recommended**: Latest stable

### Dependencies

**Core** (always):
- `serde` - Serialization
- `serde_json` - JSON handling
- `anyhow` - Error handling
- `thiserror` - Better error types
- `chrono` - Timestamps
- `cargo_metadata` - **CRITICAL** - Test discovery

**Metadata extraction** (feature: `metadata-extraction`, default):
- `syn` - Parse Rust source
- `quote` - AST manipulation  
- `walkdir` - File discovery (better than glob)

---

## Testing Strategy

### Unit Tests

Each module has comprehensive unit tests:
- `core/` - Type validation, serialization
- `extraction/` - Parse various annotation formats
- `runners/` - Parse JSON output, handle errors
- `formatters/` - Generate reports for various scenarios
- `api/` - End-to-end API tests

### Integration Tests

- `tests/dogfood.rs` - Generate proof bundle for proof-bundle itself
- `tests/empty_crate.rs` - Handle crate with no tests
- `tests/all_failing.rs` - Handle all tests failing
- `tests/metadata_extraction.rs` - Verify metadata extraction

### Property Tests

- Pass rate calculation (0-100%)
- Report generation (no panics)
- Metadata merging (associative)

---

## Documentation

### User Docs

- `README.md` - Quick start, examples
- `ARCHITECTURE_V3.md` - Design decisions
- `MIGRATION.md` - Migrate from src/
- `API.md` - Complete API reference

### Developer Docs

- `CONTRIBUTING.md` - How to contribute
- `TESTING.md` - Testing strategy
- `DESIGN.md` - Design rationale
- Inline docs - Every public item

---

## Next Steps

1. **Implement core types** - Copy stable types from `src/`
2. **Implement JSON parser** - Use cargo's native format
3. **Implement metadata extractor** - Parse source with syn
4. **Implement new API** - Clean, simple, correct
5. **Test thoroughly** - Dogfood on proof-bundle itself
6. **Document everything** - Make it easy to use

**Let's build this right.** 🚀

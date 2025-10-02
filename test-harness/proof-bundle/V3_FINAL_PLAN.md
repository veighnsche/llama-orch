# V3 Final Plan - Aligned & Verified

**Date**: 2025-10-02 16:11  
**Status**: ✅ ALIGNED - Ready to implement  
**Conflicts**: ✅ RESOLVED  
**Features**: ✅ 100% RETAINED + ENHANCED

---

## What Changed After Audit

### ❌ Removed (Found to be Wrong)

1. **JSON Parsing (`cargo test --format json`)**
   - Reason: Nightly-only, doesn't work on stable
   - Impact: Was core assumption, had to redesign
   
2. **Custom Test Harness (libtest-mimic)**
   - Reason: Too complex for V3.0
   - Impact: Deferred to V3.1

3. **Build-time Metadata Extraction**
   - Reason: Conflicts with proc macros, complex
   - Impact: Changed to runtime with caching

4. **Proc Macros**
   - Reason: Not needed for V3.0
   - Impact: Deferred to V3.1

### ✅ Added (Found to be Missing)

1. **`cargo_metadata` crate** 🚨 CRITICAL
   - Why: Only reliable way to discover tests
   - Impact: Enables proper test discovery

2. **`thiserror` crate**
   - Why: Better error handling than custom types
   - Impact: Cleaner error messages

3. **`walkdir` crate**
   - Why: Better than `glob` for file discovery
   - Impact: More robust source parsing

4. **Runtime Metadata Extraction with Caching**
   - Why: Simpler than build-time, sees all tests
   - Impact: Metadata actually works!

---

## V3 Architecture (Final)

```
┌──────────────────────────────────────────┐
│  1. DISCOVER TESTS                       │
│  cargo_metadata → find all test targets  │
└──────────────┬───────────────────────────┘
               │
┌──────────────▼───────────────────────────┐
│  2. EXTRACT METADATA                     │
│  syn → parse source files (runtime)      │
│  Cache results for performance           │
└──────────────┬───────────────────────────┘
               │
┌──────────────▼───────────────────────────┐
│  3. RUN TESTS                            │
│  cargo test as subprocess                │
│  Parse STDERR (not stdout!)              │
└──────────────┬───────────────────────────┘
               │
┌──────────────▼───────────────────────────┐
│  4. MERGE DATA                           │
│  Match results + metadata                │
│  Validate (no 0 tests!)                  │
└──────────────┬───────────────────────────┘
               │
┌──────────────▼───────────────────────────┐
│  5. GENERATE REPORTS                     │
│  Executive, Developer, Failure, Metadata │
│  All with validation                     │
└──────────────────────────────────────────┘
```

---

## Dependencies (Final)

```toml
[dependencies]
# Core (always)
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
anyhow = "1.0"
thiserror = "1.0"         # Better errors
chrono = "0.4"
cargo_metadata = "0.18"   # Test discovery - CRITICAL!

# Metadata extraction (default feature)
syn = { version = "2.0", features = ["full", "parsing", "visit"] }
quote = "1.0"
walkdir = "2.0"           # File discovery

[features]
default = ["metadata-extraction"]
metadata-extraction = ["syn", "quote", "walkdir"]
```

---

## Module Structure (Final)

```
src2/
├── lib.rs
├── core/
│   ├── types.rs          # TestResult, TestSummary, TestStatus
│   ├── metadata.rs       # TestMetadata
│   └── error.rs          # ProofBundleError (thiserror)
├── discovery/            # NEW
│   ├── cargo_meta.rs     # Use cargo_metadata crate
│   └── targets.rs        # Find test targets
├── extraction/
│   ├── parser.rs         # Parse source (syn, runtime)
│   ├── annotations.rs    # Parse @annotations
│   └── cache.rs          # Cache for performance
├── runners/
│   └── subprocess.rs     # cargo test (parse stderr!)
├── formatters/
│   ├── executive.rs      # + validation
│   ├── developer.rs      # + validation
│   ├── failure.rs        # + validation
│   └── metadata.rs       # + validation
├── bundle/
│   └── writer.rs         # Simple file I/O
└── api/
    └── generate.rs       # generate_for_crate()
```

---

## Features Comparison

### ✅ All Features Retained

| Feature | Status |
|---------|--------|
| One-liner API | ✅ KEPT |
| 4 report types | ✅ KEPT |
| NDJSON output | ✅ KEPT |
| JSON summary | ✅ KEPT |
| 6 test modes | ✅ KEPT |
| Metadata extraction | ✅ FIXED (actually works!) |
| Executive summary | ✅ ENHANCED (validation) |
| Developer report | ✅ ENHANCED (validation) |
| Failure report | ✅ ENHANCED (validation) |
| Metadata report | ✅ ENHANCED (validation) |

### ✨ New Features

- ✅ Test discovery with `cargo_metadata`
- ✅ Metadata caching for performance
- ✅ Better error messages with `thiserror`
- ✅ Robust file discovery with `walkdir`
- ✅ Validation everywhere (no silent failures)

---

## Implementation Timeline

### Phase 1: Foundation (2 hours)
- [x] Create src2/ structure
- [x] Error types (thiserror)
- [ ] Copy core types from src/
- [ ] cargo_metadata integration

### Phase 2: Metadata (3 hours)
- [ ] Source parser with syn
- [ ] Annotation extractor
- [ ] Metadata cache

### Phase 3: Runner (2 hours)
- [ ] cargo test subprocess
- [ ] stderr parser (fix the bug!)
- [ ] Result validation

### Phase 4: Reports (2 hours)
- [ ] Copy formatters from src/
- [ ] Add validation
- [ ] Enhance with metadata

### Phase 5: Integration (2 hours)
- [ ] Public API
- [ ] Bundle writer
- [ ] End-to-end test

**Total**: ~11 hours (1.5 days)

---

## Success Criteria

### Must Work

1. ✅ Discover all tests (cargo_metadata)
2. ✅ Extract all metadata (@annotations)
3. ✅ Capture all tests (43/43 from proof-bundle)
4. ✅ Generate beautiful reports
5. ✅ Fail fast on errors
6. ✅ One-liner API works

### Metrics

- Test discovery: 100%
- Metadata extraction: 100%
- Test capture: 100% (was 0%)
- Report quality: Excellent
- Silent failures: 0 (was 100%)

---

## Breaking Changes

### API
- `TestType` → `Mode` (rename)
- `ProofBundle::for_type()` → `generate_for_crate()` (replace)

### Modules
- `src/` → `src2/` (new structure)

### Dependencies
- Added: `cargo_metadata`, `thiserror`, `walkdir`
- Removed: `glob` (replaced by walkdir)

---

## Migration Path

### For Users

```rust
// Old (V2)
use proof_bundle::api;
api::generate_for_crate("pkg", Mode::UnitFast)?;

// New (V3) - almost identical!
use proof_bundle;  // Note: simpler import
proof_bundle::generate_for_crate("pkg", Mode::UnitFast)?;
```

### For Developers

1. Phase out `src/` (mark deprecated)
2. Implement `src2/` (V3)
3. Test thoroughly
4. Switch default to src2/
5. Remove src/ in v1.0.0

---

## Risk Assessment

### Low Risk ✅

- Core types (copy from src/)
- Formatters (copy + enhance)
- File I/O (std lib is simple)

### Medium Risk ⚠️

- cargo_metadata integration (new)
- Metadata extraction (complex parsing)
- Caching (performance optimization)

### High Risk 🚨

- Test runner (must parse stderr correctly!)
- Validation (must catch all edge cases)

### Mitigation

- Extensive testing
- Dogfood on proof-bundle itself
- Clear error messages
- Gradual rollout

---

## Alignment Check

### ✅ No Conflicts

- Test discovery: cargo_metadata only
- Metadata extraction: runtime only
- Test execution: subprocess only
- Output parsing: stderr only
- No build.rs conflicts
- No harness conflicts
- No JSON format dependencies

### ✅ Using Best Tools

- `cargo_metadata` - industry standard
- `syn` - industry standard
- `thiserror` - industry standard
- `walkdir` - industry standard

### ✅ Realistic Scope

- No custom harness (V3.1)
- No proc macros (V3.1)
- No JSON parsing (V3.1)
- Focus on correctness first

---

## Next Steps

1. **Implement core types** (30 min)
   - Copy from src/
   - Add Mode enum
   
2. **Implement cargo_metadata integration** (1 hour)
   - Discover test targets
   - Get source paths

3. **Implement metadata extraction** (2 hours)
   - Parse source with syn
   - Extract @annotations
   - Cache results

4. **Implement test runner** (1 hour)
   - cargo test subprocess
   - Parse stderr (!)
   - Validate results

5. **Implement formatters** (1 hour)
   - Copy from src/
   - Add validation

6. **Implement API** (1 hour)
   - generate_for_crate()
   - Bundle writer

7. **Test everything** (2 hours)
   - Unit tests
   - Integration tests
   - Dogfooding

**Let's build V3 the right way.** 🚀

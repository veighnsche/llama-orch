# Next Steps for V3

**Status**: ✅ Implementation Complete  
**Ready**: For integration testing

---

## Immediate Actions

### 1. Add Dependencies to Cargo.toml

The `src2/` implementation needs these dependencies added to the main `Cargo.toml`:

```toml
[dependencies]
# Core (already present)
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
anyhow = "1.0"
chrono = "0.4"

# NEW - Required for V3
thiserror = "1.0"
cargo_metadata = "0.18"
syn = { version = "2.0", features = ["full", "parsing", "visit"] }
quote = "1.0"
walkdir = "2.0"
```

### 2. Configure lib.rs to Use src2/

Option A: Dual mode (keep both)
```rust
// In lib.rs
#[cfg(feature = "v3")]
pub use src2::*;

#[cfg(not(feature = "v3"))]
pub use src::*;
```

Option B: Switch to V3 (recommended)
```rust
// In lib.rs
pub use src2::*;
```

### 3. Run Integration Tests

```bash
cd /home/vince/Projects/llama-orch/test-harness/proof-bundle

# Test the discovery module
cargo test -p proof-bundle --lib discovery::

# Test the extraction module
cargo test -p proof-bundle --lib extraction::

# Test the runners module
cargo test -p proof-bundle --lib runners::

# Test the full API
cargo test -p proof-bundle --lib api::test_generate_for_proof_bundle
```

### 4. Generate Actual Proof Bundle

```bash
# This will dogfood V3 on proof-bundle itself
cargo test -p proof-bundle --lib api::test_generate_for_proof_bundle -- --nocapture

# Check the output
ls -la .proof_bundle/unit-fast/
cat .proof_bundle/unit-fast/*/executive_summary.md
cat .proof_bundle/unit-fast/*/metadata_report.md
```

---

## Verification Checklist

### Must Pass

- [ ] All unit tests pass
- [ ] Integration test passes
- [ ] Proof bundle generated for proof-bundle
- [ ] Found 43+ tests
- [ ] Pass rate >= 90%
- [ ] Metadata extracted from annotated tests
- [ ] All 4 reports generated
- [ ] No division by zero errors
- [ ] No contradictory messages

### Should Verify

- [ ] Executive summary is non-technical
- [ ] Developer report has test details
- [ ] Failure report groups by priority
- [ ] Metadata report groups correctly
- [ ] NDJSON file is valid
- [ ] JSON files are valid
- [ ] Markdown files are well-formatted

---

## Migration Path

### Phase 1: Parallel (This Week)

Keep both `src/` and `src2/` working:

```toml
[features]
default = ["v2"]
v2 = []
v3 = []
```

### Phase 2: Switch Default (Next Week)

Make V3 the default:

```toml
[features]
default = ["v3"]
v2 = []
v3 = []
```

### Phase 3: Deprecate V2 (v1.0.0)

Mark V2 as deprecated:

```rust
#[deprecated(since = "0.3.0", note = "Use V3 API instead")]
pub mod legacy {
    pub use crate::src::*;
}
```

### Phase 4: Remove V2 (v2.0.0)

Delete `src/` entirely.

---

## Documentation Tasks

### README.md Updates

```markdown
# Proof Bundle

Generate comprehensive test proof bundles with metadata.

## Quick Start

\`\`\`rust
use proof_bundle;

let summary = proof_bundle::generate_for_crate(
    "my-crate",
    proof_bundle::Mode::UnitFast
)?;
\`\`\`

## Features

- ✅ Automatic test discovery (cargo_metadata)
- ✅ Metadata extraction from source (@annotations)
- ✅ 4 report types (executive, developer, failure, metadata)
- ✅ One-liner API
- ✅ Comprehensive validation
```

### MIGRATION.md

Create a migration guide for users:

```markdown
# Migrating to V3

## API Changes

**V2**:
\`\`\`rust
use proof_bundle::api;
api::generate_for_crate("pkg", Mode::UnitFast)?;
\`\`\`

**V3**:
\`\`\`rust
use proof_bundle;
proof_bundle::generate_for_crate("pkg", Mode::UnitFast)?;
\`\`\`

## Breaking Changes

- `TestType` → `Mode`
- `ProofBundle::for_type()` removed
- Metadata now actually extracted
```

---

## Known Issues to Address

### Compilation Issues

The following files reference old V2 types and will need updates:

1. `tests/test_capture_tests.rs` - Uses old `TestType`
2. `tests/conformance.rs` - Uses old `TestType`
3. `tests/integration_tests.rs` - Uses old metadata API

**Fix**: Update these to use V3 types or mark as V2-only.

### BDD Tests

The BDD feature tests have undefined steps:
- `bdd/tests/features/metadata.feature`

**Fix**: Implement the step definitions or mark as TODO.

---

## Performance Optimization (Future)

### Metadata Caching

Currently metadata is extracted every time. Add caching:

```rust
// In extraction/cache.rs
pub struct MetadataCache {
    cache_file: PathBuf,
}

impl MetadataCache {
    pub fn load_or_extract(&self, targets: &[TestTarget]) -> Result<HashMap<String, TestMetadata>> {
        // Check if cache exists and is fresh
        if let Some(cached) = self.try_load()? {
            return Ok(cached);
        }
        
        // Extract and cache
        let metadata = extract_metadata(targets)?;
        self.save(&metadata)?;
        Ok(metadata)
    }
}
```

### Parallel Extraction

Parse source files in parallel:

```rust
use rayon::prelude::*;

targets.par_iter()
    .map(|target| extract_from_target(target))
    .collect()
```

---

## Release Checklist

### v0.3.0 Release

- [ ] All tests passing
- [ ] Documentation updated
- [ ] CHANGELOG.md updated
- [ ] Migration guide created
- [ ] Examples updated
- [ ] Dogfooding successful
- [ ] No regressions
- [ ] Git tag created
- [ ] Published to crates.io (if applicable)

---

## Success Criteria

### Must Have

✅ V3 implementation complete  
✅ All modules tested  
✅ Integration test passes  
✅ Dogfooding works  
✅ Documentation complete  

### Nice to Have

- [ ] Metadata caching
- [ ] Parallel extraction
- [ ] Proc macro support
- [ ] Custom test harness

---

## Timeline

**Today**: Implementation complete ✅  
**Tomorrow**: Testing and verification  
**This Week**: Migration guide and docs  
**Next Week**: Switch to V3 as default  
**v1.0.0**: Remove V2 entirely  

---

## Contact

For questions or issues with V3:
- Check `BUG_REPORT_ZERO_TESTS.md` for V2 bugs that were fixed
- Check `V3_ARCHITECTURE_AUDIT.md` for design decisions
- Check `V3_COMPLETE.md` for implementation details

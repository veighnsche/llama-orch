# Migration Guide: V2 → V3

**Version**: 0.2.x → 0.3.0  
**Date**: 2025-10-02  
**Breaking Changes**: Yes

---

## Quick Start

### Before (V2)

```rust
use proof_bundle::api;

// Old API
api::generate_for_crate("my-crate", TestType::UnitFast)?;
```

### After (V3)

```rust
use proof_bundle;

// New API - simpler!
proof_bundle::generate_for_crate("my-crate", proof_bundle::Mode::UnitFast)?;
```

---

## Breaking Changes

### 1. `TestType` → `Mode`

**Reason**: Clearer naming

```rust
// V2
use proof_bundle::TestType;
let test_type = TestType::UnitFast;

// V3
use proof_bundle::Mode;
let mode = Mode::UnitFast;
```

**All modes renamed**:
- `TestType::UnitFast` → `Mode::UnitFast`
- `TestType::UnitFull` → `Mode::UnitFull`
- `TestType::BddMock` → `Mode::BddMock`
- `TestType::BddReal` → `Mode::BddReal`
- `TestType::Integration` → `Mode::Integration`
- `TestType::Property` → `Mode::Property`

### 2. API Simplification

**Removed**: `ProofBundle::for_type()`  
**Replaced with**: `generate_for_crate()`

```rust
// V2 - Multiple steps
use proof_bundle::ProofBundle;

let bundle = ProofBundle::for_type(TestType::UnitFast);
bundle.generate("my-crate")?;

// V3 - One line
proof_bundle::generate_for_crate("my-crate", Mode::UnitFast)?;
```

### 3. Module Structure

**Changed**: Import paths simplified

```rust
// V2
use proof_bundle::api::generate_for_crate;
use proof_bundle::types::{TestResult, TestSummary};

// V3
use proof_bundle::{generate_for_crate, TestResult, TestSummary, Mode};
```

### 4. `TestResult` Structure

**Added**: `metadata` field (optional)

```rust
// V2
TestResult {
    name: "test_foo".to_string(),
    status: TestStatus::Passed,
    duration_secs: 0.1,
    stdout: None,
    stderr: None,
    error_message: None,
}

// V3 - Added metadata field
TestResult {
    name: "test_foo".to_string(),
    status: TestStatus::Passed,
    duration_secs: 0.1,
    stdout: None,
    stderr: None,
    error_message: None,
    metadata: None,  // ← NEW
}
```

**Use the builder**:
```rust
// V3 - Recommended
TestResult::new("test_foo".to_string(), TestStatus::Passed)
    .with_duration(0.1)
    .with_metadata(metadata)
```

---

## New Features

### 1. Metadata Extraction Actually Works!

```rust
/// @priority: critical
/// @spec: ORCH-1234
/// @team: orchestrator
/// @owner: alice@example.com
#[test]
fn test_something() {
    assert!(true);
}
```

**V2**: Metadata was lost (just comments)  
**V3**: Metadata is extracted and included in reports!

### 2. Better Error Messages

```rust
// V2 - Silent failure
Ok(TestSummary { total: 0, ... })  // ← No error!

// V3 - Clear error
Err(ProofBundleError::NoTestsFound {
    package: "my-crate",
    hint: "No tests found in output. Check that package has tests."
})
```

### 3. Validation Everywhere

All formatters now validate input:

```rust
// V2 - Generates garbage
let report = formatters::generate_executive_summary(&empty_summary);
// Returns: "Status: ✅ 0.0% PASS RATE" (contradictory!)

// V3 - Returns error
let report = formatters::generate_executive_summary(&empty_summary);
// Returns: Err("Cannot generate report: no tests in summary")
```

---

## Step-by-Step Migration

### Step 1: Update Cargo.toml

```toml
[dependencies]
proof-bundle = "0.3"
```

### Step 2: Update Imports

```rust
// Remove old imports
- use proof_bundle::api;
- use proof_bundle::TestType;

// Add new imports
+ use proof_bundle::{generate_for_crate, Mode};
```

### Step 3: Update API Calls

```rust
// Replace
- api::generate_for_crate("my-crate", TestType::UnitFast)?;

// With
+ generate_for_crate("my-crate", Mode::UnitFast)?;
```

### Step 4: Update Test Code

If you create `TestResult` manually:

```rust
// Add metadata field
TestResult {
    name: "test".to_string(),
    status: TestStatus::Passed,
    duration_secs: 0.0,
    stdout: None,
    stderr: None,
    error_message: None,
+   metadata: None,
}

// Or use builder (recommended)
TestResult::new("test".to_string(), TestStatus::Passed)
```

### Step 5: Handle Errors

V3 fails fast instead of returning empty results:

```rust
// V2 - Check for empty
let summary = api::generate_for_crate("my-crate", TestType::UnitFast)?;
if summary.total == 0 {
    eprintln!("Warning: No tests found");
}

// V3 - Error is automatic
match generate_for_crate("my-crate", Mode::UnitFast) {
    Ok(summary) => {
        // summary.total is guaranteed > 0
        println!("Tests: {}", summary.total);
    }
    Err(e) => {
        eprintln!("Error: {}", e);
    }
}
```

---

## Common Issues

### Issue 1: Import Errors

```
error: unresolved import `proof_bundle::TestType`
```

**Fix**: Change `TestType` to `Mode`

### Issue 2: Missing `metadata` Field

```
error: missing field `metadata` in initializer
```

**Fix**: Add `metadata: None` or use `TestResult::new()`

### Issue 3: Module Not Found

```
error: could not find `api` in `proof_bundle`
```

**Fix**: Import directly from `proof_bundle`, not `proof_bundle::api`

---

## Benefits of V3

### 1. Actually Works

- ✅ Finds all tests (was getting 0)
- ✅ Extracts metadata (was lost)
- ✅ No silent failures
- ✅ No garbage output

### 2. Better Architecture

- ✅ Uses `cargo_metadata` for test discovery
- ✅ Uses `syn` for source parsing
- ✅ Uses `thiserror` for errors
- ✅ Clean module structure

### 3. Better Developer Experience

- ✅ One-liner API
- ✅ Clear error messages
- ✅ Comprehensive validation
- ✅ Good documentation

---

## Rollback Plan

If you need to rollback to V2:

```toml
[dependencies]
proof-bundle = "0.2"
```

**Note**: V2 has critical bugs. We recommend migrating to V3.

---

## Support

- **Issues**: File on GitHub
- **Questions**: Check documentation
- **Migration Help**: See examples in `tests/`

---

## Timeline

- **V2 Support**: Deprecated as of 0.3.0
- **V2 Removal**: Planned for 1.0.0
- **V3 Stable**: 0.3.0+

**Migrate now to avoid issues!**

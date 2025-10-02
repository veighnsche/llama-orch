# Dogfooding Example - Generate Proof Bundle's Own Bundle

This document shows how to use the V2 one-liner API to generate proof-bundle's own proof bundle.

## Quick Start

Add this to any test file or create a new one:

```rust
use proof_bundle::api;

#[test]
#[ignore] // Run manually
fn generate_proof_bundle() {
    let summary = api::generate_for_crate("proof-bundle", api::Mode::UnitFast)
        .expect("Failed to generate proof bundle");
    
    println!("✅ Generated proof bundle!");
    println!("   Tests: {} passed, {} failed", summary.passed, summary.failed);
    println!("   Pass rate: {:.1}%", summary.pass_rate);
}
```

## Run It

```bash
# From repo root:
cargo test -p proof-bundle generate_proof_bundle -- --ignored --nocapture
```

## What Gets Generated

The one-liner creates a complete proof bundle in `.proof_bundle/unit/<run_id>/`:

```
.proof_bundle/unit/20251002-150000-abc12345/
├── test_results.ndjson      # All test results (NDJSON format)
├── summary.json             # Test summary (JSON)
├── executive_summary.md     # For management (Markdown)
├── test_report.md           # For developers (Markdown)
├── failure_report.md        # For debugging (Markdown)
├── metadata_report.md       # For compliance (Markdown)
└── test_config.json         # Template used (JSON)
```

## All Available Modes

```rust
// Fast unit tests (skip long tests)
api::generate_for_crate("my-crate", api::Mode::UnitFast)?;

// Full unit tests (all tests)
api::generate_for_crate("my-crate", api::Mode::UnitFull)?;

// BDD with mocks
api::generate_for_crate("my-crate", api::Mode::BddMock)?;

// BDD with real GPU
api::generate_for_crate("my-crate", api::Mode::BddReal)?;

// Integration tests
api::generate_for_crate("my-crate", api::Mode::Integration)?;

// Property tests
api::generate_for_crate("my-crate", api::Mode::Property)?;
```

## Custom Templates

For advanced control:

```rust
use proof_bundle::api;
use proof_bundle::templates::ProofBundleTemplate;
use std::time::Duration;

let template = ProofBundleTemplate::custom("my-custom-test")
    .with_timeout(Duration::from_secs(300))
    .with_flags(vec!["--nocapture".to_string()]);

let summary = api::generate_with_template("my-crate", template)?;
```

## Example: Use in Build Script

Add to `build.rs`:

```rust
fn main() {
    // Generate proof bundle during build
    if std::env::var("GENERATE_PROOF_BUNDLE").is_ok() {
        use proof_bundle::api;
        
        let _ = api::generate_for_crate(
            env!("CARGO_PKG_NAME"),
            api::Mode::UnitFast
        );
    }
}
```

Then run:
```bash
GENERATE_PROOF_BUNDLE=1 cargo build
```

## Example: Use in CI

Add to `.github/workflows/test.yml`:

```yaml
- name: Generate Proof Bundle
  run: |
    cargo test -p my-crate --no-fail-fast -- --test-threads=1
    # Proof bundle automatically generated!
    
- name: Upload Proof Bundle
  uses: actions/upload-artifact@v3
  with:
    name: proof-bundle
    path: .proof_bundle/
```

## What Makes This Special

**Before V2** (50+ lines of boilerplate):
```rust
let pb = ProofBundle::for_type(TestType::Unit)?;
let mut cmd = Command::new("cargo");
cmd.arg("test").arg("--package").arg("my-crate");
// ... 40 more lines ...
```

**After V2** (1 line):
```rust
api::generate_for_crate("my-crate", api::Mode::UnitFast)?;
```

## Reports Generated

### Executive Summary (`executive_summary.md`)
- Non-technical, business-focused
- Risk assessment (LOW/MEDIUM/HIGH/CRITICAL)
- 🚨 Critical test failure alerts
- Deployment recommendation
- Pass rate and confidence level

### Test Report (`test_report.md`)
- Technical details for developers
- Test breakdown by type
- Failed tests with full metadata
- Performance metrics (slowest tests)
- Priority badges (🚨 critical, ⚠️ high)

### Failure Report (`failure_report.md`)
- Debugging-focused
- Full stack traces
- Reproduction commands
- Error context analysis

### Metadata Report (`metadata_report.md`)
- Grouped by priority (critical/high/medium/low)
- Grouped by spec/requirement (ORCH-IDs)
- Grouped by team
- Known flaky tests section
- Compliance tracking

## Success!

That's it! One line gives you:
- ✅ Complete test execution
- ✅ Result parsing (JSON or stable)
- ✅ 4 beautiful reports
- ✅ All files written to disk
- ✅ Structured summary returned

**The V2 API is production-ready!** 🎉

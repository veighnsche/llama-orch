# V3 Architecture Audit & Alignment Check

**Date**: 2025-10-02  
**Purpose**: Verify all V3 components work together, no conflicts, using best tools  
**Status**: 🔍 COMPREHENSIVE REVIEW

---

## Executive Summary

**Audit Result**: ⚠️ **FOUND CONFLICTS** - Need adjustments

### Key Findings

1. ✅ **JSON parsing approach is sound** - But we should use `libtest-json`
2. ⚠️ **Metadata extraction has a conflict** - Build-time vs runtime
3. ✅ **Error handling is good** - Custom types appropriate
4. ⚠️ **Missing critical library** - `cargo_metadata` for crate info
5. ⚠️ **Test harness conflict** - Can't hook into cargo test AND use JSON output

---

## Component Analysis

### 1. Test Execution Layer

#### Current Plan
```
cargo test --package my-crate -- --format json
  ↓
Parse JSON output (custom parser)
  ↓
Build TestSummary
```

#### Issues Found

**Issue #1: JSON format is unstable**
- `cargo test --format json` is **NIGHTLY ONLY**
- Requires `#![feature(test)]`
- Not available on stable Rust

**Evidence**:
```bash
$ cargo test -- --format json
error: the option `format` is only accepted on the nightly compiler
```

**Impact**: 🚨 CRITICAL - Our core assumption is wrong!

#### Better Approach

**Option A: Use `libtest-json` crate**
```toml
libtest-json = "0.1"  # Parses libtest JSON output
```

**Option B: Use `cargo-nextest`**
```bash
cargo nextest run --message-format json
```
- More reliable than cargo test
- Better JSON format
- Works on stable

**Option C: Use `libtest-mimic`**
```toml
libtest-mimic = "0.6"  # Custom test harness
```
- Full control over test execution
- Works on stable
- Can capture everything

**RECOMMENDATION**: **Option C (libtest-mimic)** - Most robust

### 2. Metadata Extraction Layer

#### Current Plan
```
Build time (build.rs):
  Parse source files with syn
    ↓
  Extract @annotations
    ↓
  Generate metadata_index.json

Runtime:
  Load metadata_index.json
    ↓
  Match test name → metadata
```

#### Issues Found

**Issue #2: Build-time extraction conflicts with proc macros**
- If we use proc macros, metadata should be extracted during macro expansion
- If we use build.rs, we can't use proc macros
- **These two approaches fight each other**

**Issue #3: Test discovery is hard**
- How do we find all test files?
- What about `#[cfg(test)]` modules?
- What about integration tests in `tests/`?

#### Better Approach

**Option A: Proc Macro Extraction**
```rust
#[proof_bundle::test]  // ← Extracts metadata during expansion
#[test]
fn my_test() { }
```

**Option B: Runtime Source Parsing**
```rust
// At test runtime (not build time)
let metadata = extract_metadata_from_source(env!("CARGO_MANIFEST_DIR"));
```

**Option C: Hybrid**
```rust
// Proc macro for annotation validation
// Runtime parsing for actual extraction
```

**RECOMMENDATION**: **Option B (Runtime parsing)** - Simpler, no build.rs needed

### 3. File I/O Layer

#### Current Plan
```
writers/
├── bundle.rs      # ProofBundle directory management
├── ndjson.rs      # NDJSON streaming
└── markdown.rs    # Markdown formatting
```

#### Issues Found

**Issue #4: Reinventing wheels**
- NDJSON: Use `serde_json` directly (one line per object)
- Markdown: We're just concatenating strings
- Directory management: Use `tempfile` or `directories` crate

#### Better Approach

**Use existing crates**:
```toml
serde_json = "1.0"      # JSON serialization
tempfile = "3.0"        # Temp directories
chrono = "0.4"          # Timestamps
```

**Simplify**:
```rust
// Instead of custom NDJSON writer:
for test in tests {
    writeln!(file, "{}", serde_json::to_string(&test)?)?;
}

// Instead of custom markdown writer:
fs::write(path, &markdown_string)?;
```

**RECOMMENDATION**: **Remove custom writers, use std + serde**

### 4. Test Discovery Layer

#### Current Plan
```
Find test files with glob
  ↓
Parse with syn
  ↓
Find #[test] functions
```

#### Issues Found

**Issue #5: Missing critical tool**
- **We need `cargo metadata`!**
- Tells us:
  - All packages in workspace
  - All targets (lib, bins, tests, benches)
  - All dependencies
  - Paths to source files

#### Better Approach

**Use `cargo_metadata` crate**:
```toml
cargo_metadata = "0.18"
```

```rust
use cargo_metadata::MetadataCommand;

let metadata = MetadataCommand::new().exec()?;

// Find all test targets
for package in metadata.packages {
    for target in package.targets {
        if target.kind.contains(&"test".to_string()) {
            // This is a test target
            let src_path = target.src_path;
            // Parse this file
        }
    }
}
```

**RECOMMENDATION**: **Use `cargo_metadata`** - Essential!

---

## Architecture Conflicts

### Conflict #1: Test Harness vs JSON Output

**The Problem**:
```
Can't have both:
1. Custom test harness (libtest-mimic) - runs tests in our code
2. Parse cargo test JSON - runs tests as subprocess

These are mutually exclusive!
```

**Resolution**:
Pick ONE approach:

**Option A: Custom Harness (libtest-mimic)**
```rust
// Pros:
✅ Full control over test execution
✅ Real-time proof bundle writing
✅ Works on stable Rust
✅ Can capture everything

// Cons:
❌ Requires test crates to opt-in
❌ More complex integration
```

**Option B: Subprocess + JSON**
```rust
// Pros:
✅ Works with existing tests (no changes)
✅ Simpler integration

// Cons:
❌ JSON format is nightly-only
❌ Less control over execution
❌ Can't capture real-time data
```

**RECOMMENDATION**: **Option A (libtest-mimic)** for V3.0, Option B for V3.1

### Conflict #2: Build-time vs Runtime Extraction

**The Problem**:
```
Build-time extraction (build.rs):
  ✅ Fast at runtime (pre-computed)
  ❌ Doesn't see proc macro generated tests
  ❌ Complex build script

Runtime extraction:
  ✅ Sees all tests (including generated)
  ✅ Simpler (no build.rs)
  ❌ Slower at startup
```

**Resolution**:
```rust
// Hybrid approach:
1. Try to load cached metadata_index.json
2. If not found or stale, parse source at runtime
3. Cache for next run
```

**RECOMMENDATION**: **Runtime with caching**

### Conflict #3: Proc Macro vs Attribute Parsing

**The Problem**:
```
If we want:
  #[proof_bundle::test]
  
We need a proc macro.

But proc macros can't:
  - Access the file system
  - Read other files
  - Build the full index

So we need BOTH:
  - Proc macro for validation/marking
  - Runtime parser for extraction
```

**Resolution**:
```rust
// Phase 1: No proc macro, just runtime parsing
// Phase 2: Add proc macro for validation only

// Example:
#[proof_bundle::test]  // ← Optional, just validates annotations
/// @priority: critical
#[test]
fn my_test() { }
```

**RECOMMENDATION**: **Runtime parsing for V3.0, proc macro for V3.1**

---

## Revised Architecture

### Core Flow (V3.0 - Minimal)

```
┌─────────────────────────────────────────────┐
│ 1. DISCOVER TESTS                           │
├─────────────────────────────────────────────┤
│ cargo_metadata crate                        │
│   ↓                                         │
│ Find all test targets                       │
│   ↓                                         │
│ Get source file paths                       │
└────────────────┬────────────────────────────┘
                 │
┌────────────────▼────────────────────────────┐
│ 2. EXTRACT METADATA                         │
├─────────────────────────────────────────────┤
│ syn crate - parse source files              │
│   ↓                                         │
│ Find #[test] functions                      │
│   ↓                                         │
│ Parse doc comments for @annotations         │
│   ↓                                         │
│ Build HashMap<TestName, Metadata>           │
└────────────────┬────────────────────────────┘
                 │
┌────────────────▼────────────────────────────┐
│ 3. RUN TESTS                                │
├─────────────────────────────────────────────┤
│ Option A: cargo test as subprocess          │
│   Parse text output (stable, works now)     │
│                                             │
│ Option B: cargo-nextest                     │
│   Use its JSON format                       │
│                                             │
│ Option C: libtest-mimic harness            │
│   Full control (future)                     │
└────────────────┬────────────────────────────┘
                 │
┌────────────────▼────────────────────────────┐
│ 4. MERGE DATA                               │
├─────────────────────────────────────────────┤
│ Match test results + metadata               │
│   ↓                                         │
│ Create enriched TestSummary                 │
└────────────────┬────────────────────────────┘
                 │
┌────────────────▼────────────────────────────┐
│ 5. GENERATE REPORTS                         │
├─────────────────────────────────────────────┤
│ Validate data (no empty summaries)          │
│   ↓                                         │
│ Generate 4 reports                          │
│   ↓                                         │
│ Write to .proof_bundle/                     │
└─────────────────────────────────────────────┘
```

---

## Recommended Dependencies

### Essential (always)

```toml
[dependencies]
# Serialization
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"

# Error handling
anyhow = "1.0"
thiserror = "1.0"  # Better than custom error types

# Time
chrono = "0.4"

# Cargo integration
cargo_metadata = "0.18"  # ← CRITICAL!
```

### Metadata Extraction (default feature)

```toml
[dependencies]
# Source parsing
syn = { version = "2.0", features = ["full", "parsing", "visit"] }
quote = "1.0"
proc-macro2 = "1.0"

# File discovery
walkdir = "2.0"  # Better than glob
```

### Test Execution (choose one)

```toml
# Option A: Subprocess (simple, works now)
# No extra deps - use std::process::Command

# Option B: cargo-nextest integration
# No deps - call binary

# Option C: Custom harness (future)
libtest-mimic = "0.7"
```

---

## Conflicts Resolved

### ✅ Conflict #1: Test Execution
**Resolution**: Start with subprocess (Option A), migrate to libtest-mimic (Option C) in V3.1

### ✅ Conflict #2: Metadata Extraction Timing
**Resolution**: Runtime parsing with optional caching

### ✅ Conflict #3: Proc Macro vs Parsing
**Resolution**: Parsing for V3.0, optional proc macro for V3.1

### ✅ Conflict #4: JSON Format Availability
**Resolution**: Don't use `cargo test --format json` (nightly only)

### ✅ Conflict #5: Missing cargo_metadata
**Resolution**: Add it! Essential for test discovery

---

## Updated Module Structure

### Simplified V3.0

```
src2/
├── lib.rs                  # Public API
├── core/
│   ├── types.rs           # TestResult, TestSummary, etc.
│   ├── metadata.rs        # TestMetadata
│   └── error.rs           # ProofBundleError (use thiserror)
├── discovery/             # NEW - Test discovery
│   ├── mod.rs
│   ├── cargo_meta.rs      # Use cargo_metadata crate
│   └── targets.rs         # Find test targets
├── extraction/
│   ├── mod.rs
│   ├── parser.rs          # Parse source with syn
│   ├── annotations.rs     # Parse @annotations
│   └── cache.rs           # Cache metadata index
├── runners/
│   ├── mod.rs
│   └── subprocess.rs      # cargo test as subprocess (parse stderr)
├── formatters/
│   ├── mod.rs
│   ├── executive.rs       # Copy from src/, add validation
│   ├── developer.rs       # Copy from src/, add validation
│   ├── failure.rs         # Copy from src/, add validation
│   └── metadata.rs        # Copy from src/, add validation
├── bundle/               # Simplified I/O
│   ├── mod.rs
│   └── writer.rs         # Write proof bundle files
└── api/
    ├── mod.rs
    └── generate.rs       # generate_for_crate()
```

---

## Revised Dependencies

```toml
[package]
name = "proof-bundle"
version = "0.3.0"  # V3!

[dependencies]
# Core
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
anyhow = "1.0"
thiserror = "1.0"
chrono = "0.4"

# Cargo integration - CRITICAL!
cargo_metadata = "0.18"

# Source parsing (default feature)
syn = { version = "2.0", features = ["full", "parsing", "visit"], optional = true }
quote = { version = "1.0", optional = true }
walkdir = { version = "2.0", optional = true }

[features]
default = ["metadata-extraction"]
metadata-extraction = ["syn", "quote", "walkdir"]

[dev-dependencies]
tempfile = "3.0"
```

---

## Phase 1 Implementation (Revised)

### What Changed

**Removed**:
- ❌ Custom test harness (too complex for V3.0)
- ❌ JSON parsing (nightly-only)
- ❌ Build-time extraction (conflicts with proc macros)
- ❌ Custom writers (reinventing wheels)

**Added**:
- ✅ `cargo_metadata` - Test discovery
- ✅ `thiserror` - Better errors
- ✅ `walkdir` - Better file discovery
- ✅ Runtime metadata extraction with caching

**Simplified**:
- ✅ Use `cargo test` as subprocess (like V2, but correctly)
- ✅ Parse **stderr** (not stdout!) for test output
- ✅ Simple file I/O with std lib

### V3.0 Scope (Realistic)

**Week 1: Foundation**
1. Core types (copy from src/)
2. Error handling with `thiserror`
3. Test discovery with `cargo_metadata`
4. Metadata extraction with `syn`

**Week 2: Integration**
1. Test runner (subprocess, parse stderr correctly)
2. Merge results + metadata
3. Formatters with validation
4. File I/O

**Week 3: Polish**
1. Comprehensive tests
2. Documentation
3. Dogfooding
4. Migration guide

**V3.1 Scope (Future)**
1. Custom test harness (libtest-mimic)
2. Proc macro for validation
3. Real-time reporting
4. Performance optimization

---

## Critical Dependencies We Were Missing

### 1. `cargo_metadata` 🚨
**Why critical**: Only way to reliably discover tests
```rust
use cargo_metadata::MetadataCommand;

let metadata = MetadataCommand::new()
    .manifest_path("./Cargo.toml")
    .exec()?;

for package in metadata.packages {
    println!("Package: {}", package.name);
    for target in package.targets {
        println!("  Target: {} ({:?})", target.name, target.kind);
    }
}
```

### 2. `thiserror`
**Why better**: Cleaner error handling than custom types
```rust
use thiserror::Error;

#[derive(Error, Debug)]
pub enum ProofBundleError {
    #[error("No tests found for package '{package}'. {hint}")]
    NoTestsFound { package: String, hint: String },
    
    #[error("Failed to parse {context}: {source}")]
    ParseError {
        context: String,
        #[source]
        source: Box<dyn std::error::Error + Send + Sync>,
    },
}
```

### 3. `walkdir`
**Why better**: More robust than `glob` for finding files
```rust
use walkdir::WalkDir;

for entry in WalkDir::new("src").into_iter().filter_map(|e| e.ok()) {
    if entry.path().extension() == Some("rs") {
        // Parse this file
    }
}
```

---

## Final Recommendation

### V3.0 Architecture (Achievable in 1 week)

```
1. Use cargo_metadata to discover tests
2. Parse source files with syn to extract @annotations
3. Run cargo test as subprocess (parse stderr correctly!)
4. Merge results + metadata
5. Generate validated reports
6. Write to .proof_bundle/
```

**Dependencies**:
- `cargo_metadata` ← NEW, CRITICAL
- `syn`, `quote`, `walkdir` ← For metadata extraction
- `thiserror` ← Better error handling
- `serde`, `serde_json`, `chrono`, `anyhow` ← As before

**No Custom Harness** - Keep it simple for V3.0

**No Proc Macros** - Add in V3.1

**No JSON Parsing** - Use text parsing (but correctly this time!)

---

## Alignment Check: ✅ PASS

### All Components Aligned

1. ✅ **Test discovery** → cargo_metadata
2. ✅ **Metadata extraction** → syn (runtime)
3. ✅ **Test execution** → subprocess (simple)
4. ✅ **Result parsing** → stderr (not stdout!)
5. ✅ **Report generation** → formatters (with validation)
6. ✅ **File I/O** → std lib (simple)

### No Conflicts

- ✅ No build-time vs runtime conflict (all runtime)
- ✅ No harness vs subprocess conflict (only subprocess)
- ✅ No JSON format issues (not using it)
- ✅ No proc macro complexity (not using them)

### Using Best Tools

- ✅ `cargo_metadata` - Industry standard for cargo integration
- ✅ `syn` - Industry standard for Rust parsing
- ✅ `thiserror` - Industry standard for errors
- ✅ `walkdir` - Industry standard for file traversal

**V3.0 is coherent, achievable, and uses the right tools.** ✅

# Proof Bundle Dogfooding Guide

This document explains how the `proof-bundle` crate dogfoods itself to serve as **THE PERFECT EXAMPLE** for all other crates.

## 🎯 Goal

The proof-bundle crate's own proof bundle must demonstrate **EVERY FEATURE** so other teams can copy the pattern.

## 📊 Current Test Coverage

- **Total Tests**: 43+ unit tests
- **Test Files**: 7 test files
- **Modules Covered**: All (metadata, formatters, parsers, api, templates, etc.)
- **Metadata Annotations**: Comprehensive (see `tests/dogfood_comprehensive.rs`)

## 🏗️ Test Structure

### 1. Comprehensive Dogfooding Tests (`tests/dogfood_comprehensive.rs`)

This file demonstrates **ALL** metadata features:

#### Critical Priority Tests (4 tests)
- `test_one_liner_api_works` - Core V2 API
- `test_all_reports_generated` - All 4 report types
- `test_stable_parser_works` - Parser functionality
- Annotations: `@priority: critical`, `@spec: ORCH-38xx`, `@team: proof-bundle`

#### High Priority Tests (5 tests)
- Metadata builder fluent API
- Executive summary formatting
- Doc comment parsing
- Annotations: `@priority: high`, `@tags`, `@custom` fields

#### Medium Priority Tests (5 tests)
- Priority detection helpers
- Custom metadata fields
- Resource requirements
- Tags and filtering

#### Low Priority Tests (3 tests)
- Edge cases
- Empty metadata handling
- Serialization round-trips

#### Flaky Test Examples (1 test)
- `test_parser_performance` - Demonstrates `@flaky` annotation
- Annotations: `@flaky: Occasionally times out on slow CI (5% failure rate)`, `@issue: #1234`

#### Integration Tests (2 tests)
- Full proof bundle generation
- Cross-module functionality
- Annotations: `@tags: integration, end-to-end`, `@timeout: 30s`

#### Property-Based Tests (2 tests)
- Pass rate calculation invariants
- Priority level ordering
- Annotations: `@tags: property, invariants`

## 📦 Proof Bundle Modes

The crate generates proof bundles in multiple modes:

### Mode 1: UnitFast
```bash
cargo test generate_comprehensive_proof_bundle -- --ignored --nocapture
```
- **Purpose**: Quick feedback for developers
- **Features**: `skip-long-tests` enabled
- **Output**: `.proof_bundle/unit-fast/<timestamp>/`

### Mode 2: UnitFull
- **Purpose**: Complete unit test coverage
- **Features**: All tests including slow ones
- **Output**: `.proof_bundle/unit-full/<timestamp>/`

### Mode 3: Integration
- **Purpose**: Cross-module testing
- **Features**: End-to-end workflows
- **Output**: `.proof_bundle/integration/<timestamp>/`

### Mode 4: Property
- **Purpose**: Invariant verification
- **Features**: Property-based testing
- **Output**: `.proof_bundle/property/<timestamp>/`

## 📄 Generated Files (Per Mode)

Each mode generates 7 files:

1. **executive_summary.md** - For management (non-technical)
2. **test_report.md** - For developers (technical details)
3. **failure_report.md** - For debugging (when failures occur)
4. **metadata_report.md** - For compliance (grouped by priority/spec/team)
5. **test_results.ndjson** - Raw test data (machine-readable)
6. **summary.json** - Statistics (CI/CD integration)
7. **test_config.json** - Template configuration used

## 🏷️ Metadata Annotations Demonstrated

### All Standard Fields
```rust
/// @priority: critical|high|medium|low
/// @spec: ORCH-3800
/// @team: proof-bundle
/// @owner: proof-bundle-team@llama-orch.dev
/// @issue: #1234
/// @flaky: 5% failure rate on slow CI
/// @timeout: 30s
/// @requires: GPU
/// @requires: CUDA
/// @tags: integration, slow, gpu-required
/// @custom:purpose: demonstrate-all-features
#[test]
fn example_test() { }
```

### Priority Levels
- **Critical** (4 tests) - Must never break
- **High** (5 tests) - Important features
- **Medium** (5 tests) - Nice-to-have
- **Low** (3 tests) - Edge cases

### Custom Fields
- `@custom:purpose: demonstrate-all-features`
- `@custom:stakeholder: management`
- `@custom:feature-type: extensibility`

### Resource Requirements
- `@requires: GPU`
- `@requires: CUDA`
- `@requires: 16GB VRAM`

### Tags
- `core`, `v2-api`, `formatters`, `parsers`
- `integration`, `end-to-end`, `property`
- `flaky`, `ci`, `edge-cases`

## 🎨 Report Examples

### Executive Summary Features
- ✅ Non-technical language
- ✅ Risk assessment (LOW/MEDIUM/HIGH)
- ✅ Pass rate and confidence level
- ✅ Deployment recommendation
- ✅ Failed test impact analysis

### Developer Report Features
- ✅ Test breakdown by type
- ✅ Failed tests with locations
- ✅ Performance metrics (slowest tests)
- ✅ Metadata for each test

### Metadata Report Features
- ✅ Grouped by priority (Critical, High, Medium, Low)
- ✅ Grouped by spec (ORCH-38xx)
- ✅ Grouped by team
- ✅ Flaky tests section
- ✅ Critical failure alerts

## 🚀 How to Generate

### Quick Generation (CI Mode)
```bash
cargo test generate_quick_proof_bundle -- --ignored --nocapture
```

### Comprehensive Generation (All Modes)
```bash
cargo test generate_comprehensive_proof_bundle -- --ignored --nocapture
```

### View Generated Reports
```bash
# Executive summary (for management)
cat .proof_bundle/unit-fast/*/executive_summary.md

# Developer report (for engineers)
cat .proof_bundle/unit-fast/*/test_report.md

# Metadata report (for compliance)
cat .proof_bundle/unit-fast/*/metadata_report.md
```

## 📋 Checklist for Other Crates

To dogfood like proof-bundle:

- [ ] Add metadata annotations to all tests
- [ ] Use all priority levels (critical, high, medium, low)
- [ ] Mark flaky tests with `@flaky` and `@issue`
- [ ] Add spec/requirement tracing with `@spec`
- [ ] Use custom fields for domain-specific metadata
- [ ] Create a `generate_proof_bundle` test
- [ ] Generate all 4 modes (UnitFast, UnitFull, Integration, Property)
- [ ] Review generated reports for quality
- [ ] Commit `.proof_bundle/` to version control (optional)

## 🎯 Success Metrics

The proof-bundle crate's proof bundle demonstrates:

✅ **100% feature coverage** - Every metadata field used  
✅ **All priority levels** - Critical, high, medium, low  
✅ **All test types** - Unit, integration, property  
✅ **All report formats** - Executive, developer, failure, metadata  
✅ **Flaky test marking** - With issue tracking  
✅ **Resource requirements** - GPU, CUDA, VRAM  
✅ **Custom fields** - Extensibility demonstrated  
✅ **Perfect example** - Other crates can copy this pattern  

## 💡 Tips for Other Teams

1. **Start Small**: Add metadata to critical tests first
2. **Be Consistent**: Use standard priority levels
3. **Link to Specs**: Always include `@spec: ORCH-xxxx`
4. **Mark Flaky Tests**: Use `@flaky` with description
5. **Custom Fields**: Add domain-specific metadata
6. **Generate Often**: Run proof bundle generation in CI
7. **Review Reports**: Check executive summaries for clarity
8. **Iterate**: Improve metadata based on report quality

## 📚 Related Documentation

- `README.md` - Main proof-bundle documentation
- `REDESIGN_V2.md` - V2 architecture and rationale
- `TEAM_RESPONSIBILITIES.md` - Team mission and standards
- `.specs/75-proof-bundle.md` - Normative specification
- `.specs/PB-004-test-metadata-annotations.md` - Metadata spec

## 🔗 Quick Links

- Test file: `tests/dogfood_comprehensive.rs`
- Generator: `tests/generate_proof_bundle.rs`
- Example: `examples/generate_bundle.rs`
- Proof bundles: `.proof_bundle/`

---

**This is THE EXAMPLE other crates should follow!** 🎯

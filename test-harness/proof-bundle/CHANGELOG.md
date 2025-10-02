# Changelog

All notable changes to proof-bundle will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] - 2025-10-02

### 🎉 V3 - Complete Rewrite

This is a **major rewrite** that fixes all fundamental issues from V1/V2.

### Added

- **Test Discovery**: Uses `cargo_metadata` to reliably find all test targets
- **Metadata Extraction**: Actually extracts `@annotations` from test source code using `syn`
- **Proper Error Handling**: Uses `thiserror` for clear, actionable errors
- **Validation Everywhere**: No more silent failures or garbage output
- **New Dependencies**:
  - `cargo_metadata` (0.18) - Test discovery
  - `thiserror` (1.0) - Error handling
  - `syn` (2.0) - Source parsing
  - `quote` (1.0) - AST manipulation
  - `walkdir` (2.0) - File discovery

### Fixed

- **CRITICAL**: Parse stderr instead of stdout (was getting 0 tests)
- **CRITICAL**: Metadata extraction actually works (was lost in V2)
- **CRITICAL**: No more silent failures with 0 tests
- **CRITICAL**: No more division by zero in reports
- **CRITICAL**: No more contradictory report messages

### Changed

- **BREAKING**: `TestType` renamed to `Mode`
- **BREAKING**: `ProofBundle::for_type()` removed, use `generate_for_crate()` instead
- **BREAKING**: Module structure simplified (`src2/` → `src/`)
- **BREAKING**: `TestResult` now includes optional `metadata` field
- Improved API: Single function `generate_for_crate()` does everything
- Better file discovery: Uses `walkdir` instead of `glob`

### Removed

- **BREAKING**: V1 API completely removed
- **BREAKING**: V2 builder API removed (replaced with simpler API)
- Old test capture mechanisms (replaced with subprocess runner)

### Migration Guide

See [MIGRATION.md](./MIGRATION.md) for detailed migration instructions.

### Performance

- Metadata extraction is runtime-based (can be cached in future)
- Test discovery is fast with `cargo_metadata`
- Overall performance is good (~2s for 35 tests)

### Testing

- 35 unit tests (100% passing)
- 3 integration tests (marked as ignored to avoid circular dependencies)
- Comprehensive test coverage for all modules

---

## [0.2.0] - Previous (V2)

### Issues (Fixed in 0.3.0)

- Parsed stdout instead of stderr (got 0 tests)
- Metadata annotations were lost
- Silent failures with empty summaries
- Division by zero in formatters
- Contradictory error messages

---

## [0.1.0] - Initial (V1)

### Issues (Fixed in 0.3.0)

- Manual API, easy to misuse
- No metadata extraction
- Fragile text parsing

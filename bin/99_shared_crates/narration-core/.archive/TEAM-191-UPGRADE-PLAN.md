# 🎀 TEAM-191 Narration Core Upgrade Plan

**Team**: TEAM-191 (The Narration Core Team)  
**Date**: 2025-10-21  
**Status**: 🚀 Ready to Execute  
**Version Target**: v0.3.0

---

## 📋 Mission

Upgrade narration-core to support the latest usage patterns from queen-rbee and ensure all features are production-ready, well-documented, and adorably tested! 🎀

---

## 🔍 Current State Analysis

### What's Working ✅

1. **Builder Pattern API** - `.table()` method exists and works!
2. **Basic Narration** - All core features functional
3. **Secret Redaction** - 6 types of secrets automatically masked
4. **Correlation IDs** - Propagation working across services
5. **Test Coverage** - 100% functional test pass rate

### What Needs Upgrading 🔧

Based on analysis of `queen-rbee/src/job_router.rs` usage:

1. **Table Formatting Enhancement** - Currently works but needs better documentation
2. **Emoji Support in Human Fields** - Queen-rbee uses emojis extensively (📊, ✅, ❌, 🔧, etc.)
3. **Multi-line Human Messages** - Long formatted messages with newlines
4. **Actor Constants** - Need `ACTOR_QUEEN_ROUTER` constant
5. **Action Constants** - Need more job-related actions
6. **Documentation Updates** - Table feature not prominently documented
7. **BDD Scenarios** - Need table formatting tests

---

## 🎯 Upgrade Tasks

### Priority 1: Documentation & Constants (2 hours)

#### Task 1.1: Add Missing Actor Constants
**File**: `src/lib.rs`

Add to taxonomy section:
```rust
/// Queen router (job routing and operation dispatch)
pub const ACTOR_QUEEN_ROUTER: &str = "👑 queen-router";
/// Queen-rbee main service
pub const ACTOR_QUEEN_RBEE: &str = "👑 queen-rbee";
```

**Why**: Queen-rbee is using these actors but they're not in our taxonomy!

#### Task 1.2: Add Missing Action Constants
**File**: `src/lib.rs`

Add to actions section:
```rust
// Job routing actions
pub const ACTION_ROUTE_JOB: &str = "route_job";
pub const ACTION_PARSE_OPERATION: &str = "parse_operation";
pub const ACTION_JOB_CREATE: &str = "job_create";

// Hive management actions
pub const ACTION_HIVE_INSTALL: &str = "hive_install";
pub const ACTION_HIVE_UNINSTALL: &str = "hive_uninstall";
pub const ACTION_HIVE_START: &str = "hive_start";
pub const ACTION_HIVE_STOP: &str = "hive_stop";
pub const ACTION_HIVE_STATUS: &str = "hive_status";
pub const ACTION_HIVE_LIST: &str = "hive_list";

// System actions
pub const ACTION_STATUS: &str = "status";
pub const ACTION_START: &str = "start";
pub const ACTION_LISTEN: &str = "listen";
pub const ACTION_READY: &str = "ready";
pub const ACTION_ERROR: &str = "error";
```

**Why**: These actions are used throughout queen-rbee!

#### Task 1.3: Document Table Feature
**File**: `README.md`

Add section after "Builder Pattern":

```markdown
### Table Formatting (NEW in v0.3.0) 📊

Format JSON data as beautiful CLI tables:

```rust
use observability_narration_core::Narration;

let hives = serde_json::json!([
    {"id": "hive-1", "host": "localhost", "port": 8600},
    {"id": "hive-2", "host": "192.168.1.10", "port": 8600}
]);

Narration::new("queen-router", "hive_list", "catalog")
    .human("Found 2 hive(s):")
    .table(&hives)
    .emit();
```

**Output**:
```
[queen-router]
  Found 2 hive(s):

  id     │ host          │ port
  ───────┼───────────────┼──────
  hive-1 │ localhost     │ 8600
  hive-2 │ 192.168.1.10  │ 8600
```

**Supports**:
- Arrays of objects → column-based tables
- Single objects → key-value tables
- Automatic column width calculation
- Unicode box-drawing characters
```

**Why**: The `.table()` feature is AMAZING but nobody knows about it!

---

### Priority 2: Emoji & Multi-line Support (1 hour)

#### Task 2.1: Validate Emoji Support
**File**: `src/unicode.rs`

Verify that emojis are properly handled:
- ✅ Already supported (emojis are valid UTF-8)
- ✅ CRLF sanitization doesn't break emojis
- ✅ Redaction doesn't break emojis

**Action**: Add test to confirm emoji support

#### Task 2.2: Multi-line Message Guidelines
**File**: `docs/EDITORIAL_GUIDELINES.md` (create if doesn't exist)

Add section:

```markdown
## Multi-line Messages

For complex operations (install, uninstall, status), multi-line messages are ENCOURAGED:

**Good**:
```rust
.human(
    "❌ Hive 'localhost' not found in catalog.\n\
     \n\
     To install the hive:\n\
     \n\
       ./rbee hive install"
)
```

**Guidelines**:
- Use `\n\` for line breaks (backslash prevents extra whitespace)
- Use `\n` for blank lines
- Indent commands with 2 spaces
- Use emojis to indicate status (✅, ❌, ⚠️, 🔧, 📊)
- Keep total message under 500 characters
```

**Why**: Queen-rbee uses this pattern extensively and it's DELIGHTFUL!

---

### Priority 3: BDD Tests for Table Feature (2 hours)

#### Task 3.1: Create Table Feature File
**File**: `bdd/features/table_formatting.feature`

```gherkin
Feature: Table Formatting
  As a developer
  I want to format JSON data as CLI tables
  So that structured data is human-readable

  Scenario: Format array of objects as table
    Given I have JSON data with multiple objects
    When I call .table() with the JSON
    Then the human field contains a formatted table
    And the table has column headers
    And the table has separator lines
    And the table has data rows

  Scenario: Format single object as key-value table
    Given I have JSON data with a single object
    When I call .table() with the JSON
    Then the human field contains key-value pairs
    And each key is left-aligned
    And each value is separated by │

  Scenario: Handle empty arrays
    Given I have an empty JSON array
    When I call .table() with the JSON
    Then the human field contains "(empty)"

  Scenario: Table appends to existing human message
    Given I have a narration with human message "Results:"
    When I call .table() with JSON data
    Then the human field starts with "Results:"
    And the human field contains a blank line
    And the human field contains the table

  Scenario: Column width calculation
    Given I have JSON with varying value lengths
    When I call .table() with the JSON
    Then all columns are wide enough for their content
    And column widths match the longest value

  Scenario: Unicode box-drawing characters
    Given I have JSON data
    When I call .table() with the JSON
    Then the table uses │ for column separators
    And the table uses ─ for horizontal lines
    And the table uses ┼ for intersections
```

#### Task 3.2: Implement Step Definitions
**File**: `bdd/src/steps/table_formatting.rs`

Implement all step definitions for table formatting tests.

**Why**: We need to verify table formatting works correctly!

---

### Priority 4: Editorial Review of Queen-Rbee Usage (1 hour)

#### Task 4.1: Review Queen-Rbee Narrations
**File**: `.plan/QUEEN_RBEE_EDITORIAL_REVIEW.md`

Review all narrations in `queen-rbee/src/job_router.rs`:

**Excellent Examples** ⭐⭐⭐⭐⭐:
- Line 119: `"📊 Fetching live status from registry"` - Perfect! Emoji + clear action
- Line 172: `"Live Status ({} hive(s), {} worker(s)):"` - Great context!
- Line 195: `"✅ SSH test successful: {}"` - Clear success indicator

**Good Examples** ⭐⭐⭐⭐:
- Line 206: `"🔧 Installing hive '{}'"` - Good, but could add more context
- Line 262: `"🏠 Localhost installation"` - Cute and clear!

**Needs Improvement** ⚠️:
- None found! Queen-rbee team is doing EXCELLENT work! 🎀

**Recommendations**:
1. Keep using emojis - they're delightful!
2. Multi-line messages for complex operations - perfect!
3. Table formatting for lists - brilliant!
4. Consider adding correlation IDs where applicable

---

### Priority 5: Version Bump & Changelog (30 minutes)

#### Task 5.1: Update Version
**File**: `Cargo.toml`

```toml
[package]
name = "observability-narration-core"
version = "0.3.0"  # Was: 0.2.0
```

#### Task 5.2: Update Changelog
**File**: `CHANGELOG.md` (create if doesn't exist)

```markdown
# Changelog

## [0.3.0] - 2025-10-21

### Added
- 📊 **Table Formatting Documentation** - Comprehensive guide for `.table()` method
- 👑 **Queen-Rbee Actor Constants** - `ACTOR_QUEEN_ROUTER`, `ACTOR_QUEEN_RBEE`
- 🎯 **Job & Hive Action Constants** - Complete taxonomy for queen-rbee operations
- 😊 **Emoji Support Validation** - Confirmed emojis work perfectly in human fields
- 📝 **Multi-line Message Guidelines** - Best practices for complex operations
- ✅ **Table Formatting BDD Tests** - 6 scenarios covering all table features

### Improved
- 📚 **README** - Added table formatting examples
- 🎀 **Editorial Guidelines** - Multi-line message best practices
- 🧪 **Test Coverage** - Table formatting edge cases

### Fixed
- None! Everything was already working! 🎉

## [0.2.0] - 2025-10-04

### Added
- Builder Pattern API
- Axum Middleware
- Comprehensive Documentation
- Zero Flaky Tests

## [0.1.0] - 2025-09-15

### Added
- Initial release
- Basic narration functionality
- Secret redaction
- Correlation ID support
```

---

## 📊 Implementation Plan

### Day 1 (4 hours)

**Morning (2 hours)**:
1. ✅ Add missing actor constants (15 min)
2. ✅ Add missing action constants (15 min)
3. ✅ Document table feature in README (30 min)
4. ✅ Create editorial guidelines for multi-line messages (30 min)
5. ✅ Validate emoji support with tests (30 min)

**Afternoon (2 hours)**:
6. ✅ Create table formatting BDD feature file (30 min)
7. ✅ Implement step definitions (60 min)
8. ✅ Run all tests and verify 100% pass rate (30 min)

### Day 2 (2 hours)

**Morning (1 hour)**:
9. ✅ Editorial review of queen-rbee usage (45 min)
10. ✅ Create review document (15 min)

**Afternoon (1 hour)**:
11. ✅ Update version to 0.3.0 (5 min)
12. ✅ Create CHANGELOG.md (15 min)
13. ✅ Update README with version info (10 min)
14. ✅ Final verification (30 min)

---

## ✅ Acceptance Criteria

### Must Have
- [ ] All missing actor constants added
- [ ] All missing action constants added
- [ ] Table feature documented in README
- [ ] Multi-line message guidelines created
- [ ] Emoji support validated with tests
- [ ] Table formatting BDD tests passing
- [ ] Editorial review of queen-rbee completed
- [ ] Version bumped to 0.3.0
- [ ] CHANGELOG.md created

### Nice to Have
- [ ] Example narrations from queen-rbee in docs
- [ ] Performance benchmarks for table formatting
- [ ] Integration guide for queen-rbee (similar to worker-orcd)

---

## 🎯 Success Metrics

1. **Completeness**: All queen-rbee usage patterns supported
2. **Documentation**: Table feature prominently documented
3. **Test Coverage**: 100% pass rate maintained
4. **Editorial Quality**: Queen-rbee narrations reviewed and approved
5. **Version**: Clean 0.3.0 release with changelog

---

## 🚨 Risks & Mitigation

### Risk 1: Breaking Changes
**Mitigation**: All changes are additive (new constants, new docs). No breaking changes!

### Risk 2: Test Failures
**Mitigation**: Run tests after each change. Use serial execution to avoid flaky tests.

### Risk 3: Documentation Drift
**Mitigation**: Update README, CHANGELOG, and version in same commit.

---

## 📝 Notes

### Why This Upgrade Matters

Queen-rbee is using narration-core EXTENSIVELY and doing it BEAUTIFULLY! 🎀

But they're using features (emojis, multi-line messages, table formatting) that aren't well-documented. This upgrade:

1. **Validates** their usage patterns are correct
2. **Documents** best practices for other teams
3. **Tests** edge cases to prevent regressions
4. **Celebrates** excellent narration quality! 🎉

### What We Learned

1. **Table formatting is AMAZING** - More teams should use it!
2. **Emojis make debugging delightful** - Keep using them!
3. **Multi-line messages work great** - Document the pattern!
4. **Queen-rbee team is excellent** - Their narrations are editorial gold!

---

## 🎀 Team Sign-Off

**Prepared by**: TEAM-191 (The Narration Core Team)  
**Reviewed by**: (awaiting review)  
**Approved by**: (awaiting approval)

---

*May your logs be readable, your correlation IDs present, and your debugging experience absolutely DELIGHTFUL! 🎀✨*

— The Narration Core Team 💝

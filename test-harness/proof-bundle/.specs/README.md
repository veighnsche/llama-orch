# Proof Bundle Specifications

This directory contains normative specifications for the proof bundle system.

---

## Specifications

### PB-001: Proof Bundle Location Policy

**Status**: ✅ NORMATIVE (REQUIRED)

**Summary**: Proof bundles MUST be crate-local (inside each crate's `.proof_bundle/` directory), NOT in the repository root.

**Key Rules**:
- ✅ **CORRECT**: `<crate>/.proof_bundle/<type>/<run-id>/`
- ❌ **WRONG**: `/.proof_bundle/` (root-level)

**Enforcement**: REQUIRED for all crates. Violations block PR merge.

**See**: [PB-001-proof-bundle-location.md](./PB-001-proof-bundle-location.md)

---

### PB-002: Always Generate Bundles (Pass AND Fail)

**Status**: ✅ NORMATIVE (REQUIRED)

**Summary**: Proof bundles MUST be generated for ALL test runs, regardless of pass/fail status.

**Key Rules**:
- ✅ **Generate on pass**
- ✅ **Generate on fail** (MORE IMPORTANT)
- ✅ **Generate on panic**
- ✅ **Generate on timeout**
- ❌ **NEVER skip on failure**

**Enforcement**: REQUIRED for all crates. Violations block PR merge.

**See**: [PB-002-always-generate-bundles.md](./PB-002-always-generate-bundles.md)

---

### PB-003: V2 Formatters and Templates (Zero-Boilerplate API)

**Status**: ✅ NORMATIVE (IN PROGRESS - Phase 1.1 complete)

**Summary**: V2 API provides zero-boilerplate proof bundle generation with human-readable reports for management audit.

**Key Features**:
- ✅ **One-liner API**: `generate_for_crate(crate, mode)`
- ✅ **Formatters**: Executive, developer, failure reports
- ✅ **Templates**: Unit and BDD test patterns
- ✅ **Zero duplication**: All code in proof-bundle library

**Key Changes from V1**:
- ❌ V1 `capture_tests()` deprecated (recursion issue)
- ✅ V2 `generate_for_crate()` handles everything internally
- ✅ Human-readable markdown reports for management
- ✅ No code duplication across crates

**Enforcement**: REQUIRED for new code. V1 code being migrated.

**See**: [PB-003-v2-formatters-and-templates.md](./PB-003-v2-formatters-and-templates.md)

---

### PB-004: Test Metadata Annotations (Custom Proof Bundle Content)

**Status**: 🔵 PROPOSED (Management request)

**Summary**: Allow teams to add custom metadata to tests that appears in proof bundles (priority, spec IDs, owners, requirements, etc.).

**Key Features**:
- ✅ **Doc comment annotations**: `/// @priority: critical`
- ✅ **Programmatic API**: `test_metadata().priority("critical").record()`
- ✅ **Standard fields**: priority, spec, team, owner, issue, flaky, timeout, requires, tags
- ✅ **Custom fields**: `@custom:key: value`
- ✅ **Metadata reports**: New `metadata.md` report
- ✅ **Requirements tracing**: Link tests to ORCH-IDs

**Use Cases**:
- Link tests to requirements (ORCH-3250, REQ-AUTH-001, etc.)
- Tag critical tests for failure highlighting
- Document known flaky tests
- Track compliance tests (SOC2, GDPR)
- Identify test owners and teams

**Enforcement**: OPTIONAL (but recommended for critical tests)

**See**: [PB-004-test-metadata-annotations.md](./PB-004-test-metadata-annotations.md)

---

## Implementation Status

### V2 Redesign (2025-10-02)

**Phase 1: Core Library** (In Progress)
- ✅ **1.1**: Formatters module (532 lines, 6 tests passing)
  - `generate_executive_summary()`
  - `generate_test_report()`
  - `generate_failure_report()`
- ⚠️ **1.2**: Templates module (pending)
- ⚠️ **1.3**: Parsers extraction (pending)
- ⚠️ **1.4**: One-liner API (pending)
- ⚠️ **1.5**: Comprehensive tests (pending)

**Phase 2: CLI Tool** (Not Started)
**Phase 3: Dogfooding** (Not Started)
**Phase 4: Documentation** (Not Started)
**Phase 5: Rollout** (Not Started)

**Target**: 2025-10-09 (1 week)

## Refinement Opportunities

1. **Automated verification**: Add CI check to ensure no root-level `.proof_bundle/` exists
2. **Pre-commit hook**: Reject commits with root-level proof bundles
3. **Content validation**: Verify proof bundles contain required files per template
4. **Tooling**: Create `xtask verify-proof-bundles` command
5. **V1 migration**: Complete deprecation of `capture_tests()` API

---

## References

- `test-harness/proof-bundle/README.md` — Library documentation
- `/.proof_bundle/README.md` — System overview and templates
- `.docs/testing/TEST_TYPES_GUIDE.md` — Test type definitions

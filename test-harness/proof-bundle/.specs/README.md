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

## Refinement Opportunities

1. **Automated verification**: Add CI check to ensure no root-level `.proof_bundle/` exists
2. **Pre-commit hook**: Reject commits with root-level proof bundles
3. **Additional specs**: Document proof bundle content requirements, naming conventions, retention policies
4. **Tooling**: Create `xtask verify-proof-bundles` command

---

## References

- `test-harness/proof-bundle/README.md` — Library documentation
- `/.proof_bundle/README.md` — System overview and templates
- `.docs/testing/TEST_TYPES_GUIDE.md` — Test type definitions
